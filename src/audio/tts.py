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
                if speaker_wav and os.path.exists(speaker_wav):
                    self._tts.tts_to_file(
                        text=text,
                        speaker_wav=speaker_wav,
                        language="hi",
                        file_path=out_path,
                    )
                else:
                    self._tts.tts_to_file(
                        text=text,
                        language="hi",
                        file_path=out_path,
                    )

                # Get actual duration of synthesized audio
                duration = self._get_duration(out_path)

                segment_audios.append(SegmentAudio(
                    start=seg["start"],
                    end=seg["end"],
                    audio_path=out_path,
                    sample_rate=22050,
                ))

                logger.debug(
                    f"Segment {i}: [{seg['start']:.2f}-{seg['end']:.2f}] "
                    f"target={seg['end'] - seg['start']:.2f}s, "
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
