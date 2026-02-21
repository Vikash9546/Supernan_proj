"""
tts.py
Per-segment TTS synthesis with individual voice cloning.
Each sentence is synthesized independently for precise micro-alignment.
"""

import os
import logging
import numpy as np
import subprocess
from typing import List, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class SegmentAudio:
    """A synthesized audio segment with its temporal position."""
    start: float          # Original timeline position (seconds)
    end: float            # Original timeline end (seconds)
    audio_path: str       # Path to WAV file for this segment
    sample_rate: int = 22050


class XTTSGenerator:
    """
    Synthesizes Hindi audio per-segment for precise micro-alignment.
    Model is loaded once and reused across all segments.
    """

    def __init__(self, checkpoint: str = "tts_models/multilingual/multi-dataset/xtts_v2",
                 device: str = "cuda"):
        self.device = device
        self.checkpoint = checkpoint
        self._tts = None

    def load(self):
        """Load the TTS model into VRAM."""
        from TTS.api import TTS
        logger.info(f"Loading XTTS v2 on {self.device}")
        self._tts = TTS(self.checkpoint).to(self.device)
        return self

    def synthesize(self, hindi_segs: List[dict], workdir: str,
                   speaker_wav: Optional[str] = None) -> List[SegmentAudio]:
        """
        Synthesize each Hindi segment individually.
        
        Args:
            hindi_segs: List of dicts with 'text', 'start', 'end'
            workdir: Directory for intermediate WAV files
            speaker_wav: Reference audio for voice cloning
            
        Returns:
            List of SegmentAudio with individual WAV paths and timestamps.
        """
        assert self._tts is not None, "Call .load() first"

        segment_audios = []

        for i, seg in enumerate(hindi_segs):
            text = seg["text"].strip()
            if not text:
                continue

            out_path = os.path.join(workdir, f"tts_seg_{i:04d}.wav")

            try:
                target_duration = seg["end"] - seg["start"]
                
                def _generate(speed=1.0):
                    # XTTS has a hard 250 character limit per generation.
                    # We split long texts into chunks of ~200 chars.
                    words = text.split()
                    chunks = []
                    current_chunk = []
                    current_len = 0
                    for w in words:
                        if current_len + len(w) + 1 > 200:
                            chunks.append(" ".join(current_chunk))
                            current_chunk = [w]
                            current_len = len(w)
                        else:
                            current_chunk.append(w)
                            current_len += len(w) + 1
                    if current_chunk:
                        chunks.append(" ".join(current_chunk))

                    import soundfile as sf
                    all_audio = []
                    sr = 22050
                    
                    for i, chunk in enumerate(chunks):
                        temp_chunk = out_path.replace(".wav", f"_part{i}.wav")
                        if speaker_wav and os.path.exists(speaker_wav):
                            self._tts.tts_to_file(
                                text=chunk,
                                speaker_wav=speaker_wav,
                                language="hi",
                                file_path=temp_chunk,
                                speed=speed
                            )
                        else:
                            self._tts.tts_to_file(
                                text=chunk,
                                language="hi",
                                file_path=temp_chunk,
                                speed=speed
                            )
                        if os.path.exists(temp_chunk):
                            audio_data, sr = sf.read(temp_chunk)
                            all_audio.append(audio_data)
                            os.remove(temp_chunk)
                    
                    if all_audio:
                        import numpy as np
                        combined = np.concatenate(all_audio)
                        sf.write(out_path, combined, sr)
                
                # First pass: generate at normal speed
                _generate(speed=1.0)
                duration = self._get_duration(out_path)

                # Second pass: adjust speed if duration mismatch is significant
                tolerance = 0.1  # seconds
                if abs(duration - target_duration) > tolerance:
                    speed_ratio = duration / target_duration
                    logger.debug(f"Adjusting speech rate for segment {i}: speed={speed_ratio:.2f}")
                    # XTTS speed range is typically limited (e.g., 0.5 to 2.0)
                    speed_ratio = max(0.5, min(2.0, speed_ratio))
                    _generate(speed=speed_ratio)
                    duration = self._get_duration(out_path)

                segment_audios.append(SegmentAudio(
                    start=seg["start"],
                    end=seg["end"],
                    audio_path=out_path,
                    sample_rate=22050,
                ))

                logger.debug(
                    f"Segment {i}: [{seg['start']:.2f}-{seg['end']:.2f}] "
                    f"target={target_duration:.2f}s, "
                    f"synth={duration:.2f}s → {out_path}"
                )

            except Exception as e:
                logger.error(f"TTS failed for segment {i}: {e}")
                continue

        logger.info(f"Synthesized {len(segment_audios)}/{len(hindi_segs)} segments")
        return segment_audios

    @staticmethod
    def _get_duration(audio_path: str) -> float:
        """Get audio duration using ffprobe."""
        cmd = [
            "ffprobe", "-v", "error", "-show_entries",
            "format=duration", "-of",
            "default=noprint_wrappers=1:nokey=1", audio_path
        ]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, text=True)
        return float(res.stdout.strip())
