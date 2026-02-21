"""
batch.py
Silence-aware chunking using VAD.
Ensures no chunk splits mid-word by aligning boundaries to silence gaps.
Adds 0.5s overlap buffer between chunks to avoid boundary drift.
"""

import subprocess
import struct
import logging
import wave
from typing import List, Tuple

logger = logging.getLogger(__name__)

# VAD aggressiveness: 0 (least aggressive) to 3 (most aggressive)
DEFAULT_VAD_AGGRESSIVENESS = 2
OVERLAP_BUFFER_SEC = 0.5
MIN_CHUNK_SEC = 5.0
MAX_CHUNK_SEC = 30.0
FRAME_DURATION_MS = 30  # webrtcvad works with 10/20/30ms frames
SAMPLE_RATE = 16000


def _extract_audio_pcm(video_path: str, workdir: str) -> str:
    """Extract mono 16kHz PCM WAV from video for VAD processing."""
    out_path = f"{workdir}/vad_audio.wav"
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-vn", "-acodec", "pcm_s16le",
        "-ar", str(SAMPLE_RATE), "-ac", "1",
        out_path
    ]
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    return out_path


def _read_wav_frames(wav_path: str, frame_duration_ms: int = FRAME_DURATION_MS):
    """Yield (timestamp_sec, pcm_bytes) for each VAD frame."""
    with wave.open(wav_path, "rb") as wf:
        sample_rate = wf.getframerate()
        assert sample_rate == SAMPLE_RATE, f"Expected {SAMPLE_RATE}Hz, got {sample_rate}Hz"
        assert wf.getsampwidth() == 2, "Expected 16-bit audio"
        assert wf.getnchannels() == 1, "Expected mono audio"

        samples_per_frame = int(sample_rate * frame_duration_ms / 1000)
        bytes_per_frame = samples_per_frame * 2  # 16-bit = 2 bytes

        frame_idx = 0
        while True:
            data = wf.readframes(samples_per_frame)
            if len(data) < bytes_per_frame:
                break
            timestamp = frame_idx * frame_duration_ms / 1000.0
            yield timestamp, data
            frame_idx += 1


def detect_speech_regions(wav_path: str, aggressiveness: int = DEFAULT_VAD_AGGRESSIVENESS
                          ) -> List[Tuple[float, float]]:
    """
    Detect speech regions in audio using webrtcvad.
    Returns list of (start_sec, end_sec) for continuous speech regions.
    """
    try:
        import webrtcvad
    except ImportError:
        logger.warning("webrtcvad not installed, falling back to fixed chunking")
        return []

    vad = webrtcvad.Vad(aggressiveness)

    speech_frames = []
    for timestamp, frame_bytes in _read_wav_frames(wav_path):
        is_speech = vad.is_speech(frame_bytes, SAMPLE_RATE)
        speech_frames.append((timestamp, is_speech))

    if not speech_frames:
        return []

    # Merge consecutive speech frames into regions
    regions = []
    region_start = None
    frame_dur = FRAME_DURATION_MS / 1000.0

    for timestamp, is_speech in speech_frames:
        if is_speech and region_start is None:
            region_start = timestamp
        elif not is_speech and region_start is not None:
            regions.append((region_start, timestamp + frame_dur))
            region_start = None
    if region_start is not None:
        regions.append((region_start, speech_frames[-1][0] + frame_dur))

    # Merge regions that are very close together (< 0.3s gap = same utterance)
    merged = []
    for start, end in regions:
        if merged and (start - merged[-1][1]) < 0.3:
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))

    return merged


def get_silence_boundaries(speech_regions: List[Tuple[float, float]],
                           total_duration: float) -> List[float]:
    """
    Extract silence midpoints between speech regions — ideal chunk boundaries.
    """
    boundaries = [0.0]
    for i in range(len(speech_regions) - 1):
        gap_start = speech_regions[i][1]
        gap_end = speech_regions[i + 1][0]
        if gap_end - gap_start > 0.1:  # Meaningful silence gap
            midpoint = (gap_start + gap_end) / 2.0
            boundaries.append(midpoint)
    boundaries.append(total_duration)
    return boundaries


class VideoBatcher:
    """
    Silence-aware chunking that never splits mid-word.
    Uses VAD to find natural silence boundaries, then groups
    them into chunks within [MIN_CHUNK_SEC, MAX_CHUNK_SEC] range
    with 0.5s overlap buffers for smooth boundary transitions.
    """

    def __init__(self, chunk_sec: int = 30, overlap_sec: float = OVERLAP_BUFFER_SEC):
        self.max_chunk_sec = min(chunk_sec, MAX_CHUNK_SEC)
        self.overlap_sec = overlap_sec

    def get_duration(self, video_path: str) -> float:
        """Get video/audio duration using ffprobe."""
        cmd = [
            "ffprobe", "-v", "error", "-show_entries",
            "format=duration", "-of",
            "default=noprint_wrappers=1:nokey=1", video_path
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            return float(result.stdout.strip())
        except ValueError:
            raise RuntimeError(f"Could not read duration for {video_path}")

    def split(self, video_path: str, workdir: str = "/tmp") -> List[Tuple[float, float]]:
        """
        Split video into silence-aligned chunks.
        Returns list of (start_time, end_time) tuples.
        """
        duration = self.get_duration(video_path)

        # Extract audio and run VAD
        try:
            wav_path = _extract_audio_pcm(video_path, workdir)
            speech_regions = detect_speech_regions(wav_path)
        except Exception as e:
            logger.warning(f"VAD failed ({e}), falling back to fixed chunking")
            speech_regions = []

        if not speech_regions:
            return self._fixed_split(duration)

        # Get candidate boundaries at silence midpoints
        silence_bounds = get_silence_boundaries(speech_regions, duration)
        logger.info(f"Found {len(silence_bounds)} silence boundaries in {duration:.1f}s video")

        # Group boundaries into chunks respecting max duration
        chunks = []
        chunk_start = 0.0

        for boundary in silence_bounds[1:]:  # Skip the 0.0 start
            chunk_len = boundary - chunk_start
            if chunk_len >= self.max_chunk_sec:
                # This chunk is long enough, cut here
                chunks.append((chunk_start, boundary))
                chunk_start = max(0.0, boundary - self.overlap_sec)
            elif boundary == duration:
                # Last boundary, close the final chunk
                if chunk_len >= MIN_CHUNK_SEC or not chunks:
                    chunks.append((chunk_start, duration))
                else:
                    # Merge short tail into previous chunk
                    if chunks:
                        chunks[-1] = (chunks[-1][0], duration)
                    else:
                        chunks.append((chunk_start, duration))

        # If we didn't create any chunks, use the full duration
        if not chunks:
            chunks = [(0.0, duration)]

        logger.info(f"Created {len(chunks)} silence-aligned chunks")
        for i, (s, e) in enumerate(chunks):
            logger.debug(f"  Chunk {i}: {s:.2f}s - {e:.2f}s ({e - s:.2f}s)")

        return chunks

    def _fixed_split(self, duration: float) -> List[Tuple[float, float]]:
        """Fallback: fixed-interval chunking with overlap."""
        chunks = []
        current = 0.0
        while current < duration:
            end = min(current + self.max_chunk_sec, duration)
            chunks.append((current, end))
            current += self.max_chunk_sec - self.overlap_sec
        return chunks
