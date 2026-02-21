"""
compose.py
Streaming video composition using FFmpeg pipes.
No numpy full-frame buffering — writes frames directly to encoder.
"""

import subprocess
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class VideoComposer:
    """
    Streaming video composer that pipes processed frames directly
    to FFmpeg's stdin without buffering the entire video in memory.
    
    Old approach: held ALL frames in a numpy array (OOM on long videos)
    New approach: frame-by-frame pipe to FFmpeg (constant memory footprint)
    """

    def __init__(self, fps: float, width: int, height: int,
                 audio_path: Optional[str] = None,
                 output_path: str = "output.mp4"):
        self.fps = fps
        self.width = width
        self.height = height
        self.audio_path = audio_path
        self.output_path = output_path
        self._proc = None
        self._frame_count = 0

    def open(self) -> 'VideoComposer':
        """Open the FFmpeg encoder pipe."""
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            # Raw video input from stdin
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-s", f"{self.width}x{self.height}",
            "-pix_fmt", "rgb24",
            "-r", str(self.fps),
            "-i", "-",
        ]

        if self.audio_path:
            cmd += ["-i", self.audio_path]

        cmd += [
            "-c:v", "libx264",
            "-preset", "medium",
            "-crf", "18",         # High quality
            "-pix_fmt", "yuv420p",
        ]

        if self.audio_path:
            cmd += [
                "-c:a", "aac",
                "-b:a", "192k",
                "-shortest",      # Match shortest stream
            ]

        cmd.append(self.output_path)

        self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        logger.info(f"VideoComposer opened: {self.output_path} "
                     f"({self.width}x{self.height} @ {self.fps:.1f}fps)")
        return self

    def write_frame(self, frame) -> None:
        """Write a single frame to the encoder pipe."""
        import numpy as np
        assert isinstance(frame, np.ndarray), "Frame must be numpy array"
        assert frame.shape == (self.height, self.width, 3), \
            f"Frame shape {frame.shape} != ({self.height}, {self.width}, 3)"

        self._proc.stdin.write(frame.tobytes())
        self._frame_count += 1

    def close(self) -> None:
        """Close the encoder pipe and finalize the video."""
        if self._proc and self._proc.poll() is None:
            self._proc.stdin.close()
            stderr = self._proc.stderr.read()
            self._proc.wait()

            if self._proc.returncode != 0:
                logger.error(f"FFmpeg encoding error: {stderr.decode()}")
                raise RuntimeError(f"Error composing video: {self.output_path}")

            logger.info(f"VideoComposer closed: {self.output_path} "
                         f"({self._frame_count} frames)")

    def __enter__(self):
        return self.open()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    # Legacy compatibility
    def write(self) -> None:
        """
        Legacy API — not used in new pipeline.
        The new pipeline uses open()/write_frame()/close() streaming pattern.
        """
        logger.warning("Legacy write() called — use streaming API instead")
