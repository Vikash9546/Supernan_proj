"""
stream_utils.py
FFmpeg pipe-based streaming I/O for zero-copy frame processing.
Eliminates full-frame numpy buffering and disk thrashing.
"""

import subprocess
import numpy as np
import logging
from typing import Iterator, Tuple, Optional, Dict, Any

logger = logging.getLogger(__name__)


def probe_video(video_path: str) -> Dict[str, Any]:
    """Probe video for fps, width, height, duration using ffprobe."""
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate,nb_frames",
        "-show_entries", "format=duration",
        "-of", "json", video_path
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    import json
    data = json.loads(result.stdout)

    stream = data["streams"][0]
    width = int(stream["width"])
    height = int(stream["height"])

    # Parse fractional fps like "30000/1001"
    fps_str = stream["r_frame_rate"]
    if "/" in fps_str:
        num, den = fps_str.split("/")
        fps = float(num) / float(den)
    else:
        fps = float(fps_str)

    duration = float(data["format"]["duration"])
    nb_frames = int(stream.get("nb_frames", 0)) or int(duration * fps)

    return {
        "width": width,
        "height": height,
        "fps": fps,
        "duration": duration,
        "nb_frames": nb_frames,
    }


class FFmpegFrameReader:
    """
    Streams video frames one-at-a-time from ffmpeg via pipe.
    Never buffers the full video in memory.
    
    Usage:
        reader = FFmpegFrameReader("input.mp4")
        for idx, frame in reader:
            process(frame)  # frame is (H, W, 3) uint8
        reader.close()
    """

    def __init__(self, video_path: str, start_time: Optional[float] = None,
                 end_time: Optional[float] = None):
        self.video_path = video_path
        info = probe_video(video_path)
        self.width = info["width"]
        self.height = info["height"]
        self.fps = info["fps"]
        self.frame_size = self.width * self.height * 3  # RGB24

        cmd = ["ffmpeg", "-loglevel", "error"]
        if start_time is not None:
            cmd += ["-ss", str(start_time)]
        cmd += ["-i", video_path]
        if end_time is not None and start_time is not None:
            cmd += ["-t", str(end_time - start_time)]
        cmd += [
            "-f", "rawvideo",
            "-pix_fmt", "rgb24",
            "-v", "error",
            "-"
        ]

        self._proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self._frame_idx = 0
        logger.info(f"FFmpegFrameReader opened: {video_path} "
                     f"({self.width}x{self.height} @ {self.fps:.2f}fps)")

    def __iter__(self) -> Iterator[Tuple[int, np.ndarray]]:
        return self

    def __next__(self) -> Tuple[int, np.ndarray]:
        raw = self._proc.stdout.read(self.frame_size)
        if len(raw) < self.frame_size:
            self.close()
            raise StopIteration

        frame = np.frombuffer(raw, dtype=np.uint8).reshape(
            (self.height, self.width, 3)
        )
        idx = self._frame_idx
        self._frame_idx += 1
        return idx, frame

    def close(self):
        if self._proc and self._proc.poll() is None:
            self._proc.stdout.close()
            self._proc.stderr.close()
            self._proc.terminate()
            self._proc.wait()
            self._proc = None


class FFmpegFrameWriter:
    """
    Streams processed frames to ffmpeg encoder via stdin pipe.
    Accepts frames one-at-a-time, never buffers.
    
    Usage:
        writer = FFmpegFrameWriter("output.mp4", fps=30, width=1920, height=1080)
        for frame in processed_frames:
            writer.write(frame)
        writer.close()
    """

    def __init__(self, output_path: str, fps: float, width: int, height: int,
                 audio_path: Optional[str] = None):
        self.output_path = output_path
        self.width = width
        self.height = height

        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            # Video input from pipe
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-s", f"{width}x{height}",
            "-pix_fmt", "rgb24",
            "-r", str(fps),
            "-i", "-",
        ]

        if audio_path:
            cmd += ["-i", audio_path]

        cmd += [
            "-c:v", "libx264",
            "-preset", "medium",
            "-crf", "18",
            "-pix_fmt", "yuv420p",
        ]

        if audio_path:
            cmd += ["-c:a", "aac", "-b:a", "192k", "-shortest"]

        cmd.append(output_path)

        self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        self._frame_count = 0
        logger.info(f"FFmpegFrameWriter opened: {output_path} "
                     f"({width}x{height} @ {fps:.2f}fps)")

    def write(self, frame: np.ndarray) -> None:
        """Write a single (H, W, 3) uint8 frame."""
        assert frame.shape == (self.height, self.width, 3), \
            f"Frame shape {frame.shape} != expected ({self.height}, {self.width}, 3)"
        self._proc.stdin.write(frame.tobytes())
        self._frame_count += 1

    def close(self) -> None:
        if self._proc and self._proc.poll() is None:
            self._proc.stdin.close()
            self._proc.wait()
            logger.info(f"FFmpegFrameWriter closed: {self.output_path} "
                         f"({self._frame_count} frames written)")
            self._proc = None
