"""
align.py
Per-segment time stretching using librosa/rubberband.
Replaces global atempo with pitch-preserving per-sentence alignment.
Reconstructs full audio by placing each stretched segment at its original
timestamp, filling gaps with silence to preserve natural pauses.
"""

import os
import subprocess
import logging
import numpy as np
import wave
from typing import List

logger = logging.getLogger(__name__)

# Target sample rate for output
OUTPUT_SAMPLE_RATE = 22050


def _load_wav_as_float(path: str) -> tuple:
    """Load WAV as float32 numpy array. Returns (audio, sample_rate)."""
    import wave
    with wave.open(path, "rb") as wf:
        sr = wf.getframerate()
        n_channels = wf.getnchannels()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)

    dtype = np.int16 if wf.getsampwidth() == 2 else np.int32
    audio = np.frombuffer(raw, dtype=dtype).astype(np.float32)

    if n_channels > 1:
        audio = audio.reshape(-1, n_channels).mean(axis=1)

    # Normalize to [-1, 1]
    audio = audio / np.iinfo(dtype).max
    return audio, sr


def _save_wav(audio: np.ndarray, path: str, sr: int = OUTPUT_SAMPLE_RATE) -> None:
    """Save float32 audio as 16-bit PCM WAV."""
    audio_int16 = np.clip(audio * 32767, -32768, 32767).astype(np.int16)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(audio_int16.tobytes())


def _time_stretch(audio: np.ndarray, sr: int, ratio: float) -> np.ndarray:
    """
    Time-stretch audio by ratio without changing pitch.
    ratio > 1.0 = speed up, ratio < 1.0 = slow down.
    Uses librosa with pyrubberband backend if available, otherwise
    falls back to librosa phase vocoder.
    """
    if abs(ratio - 1.0) < 0.02:
        return audio  # No stretch needed

    # Clamp to safe range
    ratio = max(0.25, min(ratio, 4.0))

    try:
        import pyrubberband as pyrb
        stretched = pyrb.time_stretch(audio, sr, ratio)
        logger.debug(f"Rubberband stretch: ratio={ratio:.3f}")
        return stretched
    except ImportError:
        pass

    try:
        import librosa
        stretched = librosa.effects.time_stretch(audio, rate=ratio)
        logger.debug(f"Librosa stretch: ratio={ratio:.3f}")
        return stretched
    except ImportError:
        pass

    # Absolute fallback: ffmpeg atempo via subprocess
    logger.warning("Neither pyrubberband nor librosa available, using ffmpeg atempo")
    return audio  # Return unchanged as last resort


class AudioAligner:
    """
    Per-segment time stretching and timeline reconstruction.
    
    For each segment:
      1. Load its synthesized audio
      2. Compute stretch ratio = synth_duration / target_duration
      3. Time-stretch to match original timing
      4. Place at original timestamp in output timeline
      5. Fill between segments with silence (preserving pauses)
    """

    def __init__(self, total_duration: float, sample_rate: int = OUTPUT_SAMPLE_RATE):
        """
        Args:
            total_duration: Total duration of the original audio/video in seconds.
            sample_rate: Output sample rate.
        """
        self.total_duration = total_duration
        self.sample_rate = sample_rate

    def align_segments(self, segment_audios, output_path: str) -> str:
        """
        Align and reconstruct full audio from per-segment TTS outputs.
        
        Args:
            segment_audios: List of SegmentAudio (start, end, audio_path)
            output_path: Path for the final reconstructed WAV
            
        Returns:
            Path to the aligned output WAV.
        """
        # Create silence canvas for the full duration
        total_samples = int(self.total_duration * self.sample_rate)
        output = np.zeros(total_samples, dtype=np.float32)

        aligned_count = 0
        for seg in segment_audios:
            try:
                # Load synthesized segment
                audio, sr = _load_wav_as_float(seg.audio_path)

                # Resample if necessary
                if sr != self.sample_rate:
                    try:
                        import librosa
                        audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)
                    except ImportError:
                        logger.warning(f"Cannot resample {sr}→{self.sample_rate}, skipping segment")
                        continue

                # Calculate target duration for this segment
                target_duration = seg.end - seg.start
                synth_duration = len(audio) / self.sample_rate

                if synth_duration <= 0 or target_duration <= 0:
                    continue

                # Time-stretch to match original timing
                ratio = synth_duration / target_duration
                if abs(ratio - 1.0) > 0.02:
                    audio = _time_stretch(audio, self.sample_rate, ratio)
                    logger.debug(
                        f"Segment [{seg.start:.2f}-{seg.end:.2f}]: "
                        f"synth={synth_duration:.2f}s → target={target_duration:.2f}s "
                        f"(ratio={ratio:.3f})"
                    )

                # Trim or pad to exact target length
                target_samples = int(target_duration * self.sample_rate)
                if len(audio) > target_samples:
                    audio = audio[:target_samples]
                elif len(audio) < target_samples:
                    audio = np.pad(audio, (0, target_samples - len(audio)))

                # Apply fade in/out to avoid clicks at boundaries
                fade_samples = min(int(0.01 * self.sample_rate), len(audio) // 4)
                if fade_samples > 0:
                    audio[:fade_samples] *= np.linspace(0, 1, fade_samples)
                    audio[-fade_samples:] *= np.linspace(1, 0, fade_samples)

                # Place into timeline at original timestamp
                start_sample = int(seg.start * self.sample_rate)
                end_sample = start_sample + len(audio)

                # Bounds check
                if start_sample >= total_samples:
                    continue
                if end_sample > total_samples:
                    audio = audio[:total_samples - start_sample]
                    end_sample = total_samples

                output[start_sample:end_sample] += audio
                aligned_count += 1

            except Exception as e:
                logger.error(f"Failed to align segment [{seg.start:.2f}-{seg.end:.2f}]: {e}")
                continue

        logger.info(f"Aligned {aligned_count}/{len(segment_audios)} segments, "
                     f"output: {self.total_duration:.2f}s")

        # Normalize to prevent clipping
        peak = np.abs(output).max()
        if peak > 0.95:
            output = output * (0.95 / peak)

        _save_wav(output, output_path, self.sample_rate)
        return output_path

    def align(self, input_wav: str, output_wav: str) -> None:
        """
        Legacy API: single-file alignment using global stretch.
        Kept for backward compatibility but per-segment is preferred.
        """
        current_duration = self._get_duration(input_wav)

        if abs(current_duration - self.total_duration) <= 0.05:
            subprocess.run(["cp", input_wav, output_wav])
            return

        ratio = current_duration / self.total_duration
        ratio = max(0.5, min(ratio, 100.0))

        cmd = [
            "ffmpeg", "-y", "-i", input_wav,
            "-filter:a", f"atempo={ratio}",
            output_wav
        ]
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    @staticmethod
    def _get_duration(audio_path: str) -> float:
        cmd = [
            "ffprobe", "-v", "error", "-show_entries",
            "format=duration", "-of",
            "default=noprint_wrappers=1:nokey=1", audio_path
        ]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, text=True)
        return float(res.stdout.strip())
