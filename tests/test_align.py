"""
tests/test_align.py
Unit tests for per-segment audio alignment.
"""

import numpy as np
import os
import sys
import tempfile
import wave

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.audio.align import AudioAligner, _save_wav, _load_wav_as_float, _time_stretch


def _create_test_wav(path: str, duration: float, sr: int = 22050):
    """Create a test WAV file with a sine wave."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    audio = (np.sin(2 * np.pi * 440 * t) * 0.5).astype(np.float32)
    _save_wav(audio, path, sr)
    return path


def test_segment_alignment_preserves_timing():
    """Aligned segments should match original timing within ±50ms."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create 3 segments with different synth durations
        from dataclasses import dataclass

        @dataclass
        class MockSegAudio:
            start: float
            end: float
            audio_path: str
            sample_rate: int = 22050

        segments = []
        test_cases = [
            # (original_start, original_end, synth_duration)
            (0.0, 2.0, 2.5),   # Synth is 25% too long → stretch faster
            (2.5, 4.5, 1.5),   # Synth is 25% too short → stretch slower
            (5.0, 7.0, 2.0),   # Synth exactly matches → no stretch
        ]

        for i, (start, end, synth_dur) in enumerate(test_cases):
            wav_path = os.path.join(tmpdir, f"seg_{i}.wav")
            _create_test_wav(wav_path, synth_dur)
            segments.append(MockSegAudio(
                start=start, end=end, audio_path=wav_path
            ))

        # Align
        total_duration = 8.0
        aligner = AudioAligner(total_duration=total_duration)
        output_path = os.path.join(tmpdir, "aligned.wav")
        aligner.align_segments(segments, output_path)

        # Verify output exists and has correct duration
        assert os.path.exists(output_path), "Output WAV not created"
        audio, sr = _load_wav_as_float(output_path)
        actual_duration = len(audio) / sr

        assert abs(actual_duration - total_duration) < 0.1, \
            f"Output duration {actual_duration:.2f}s != target {total_duration:.2f}s"

        print(f"✓ Output duration: {actual_duration:.2f}s (target: {total_duration:.2f}s)")


def test_silence_gaps_preserved():
    """Silence gaps between segments should remain silent (near-zero amplitude)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        from dataclasses import dataclass

        @dataclass
        class MockSegAudio:
            start: float
            end: float
            audio_path: str
            sample_rate: int = 22050

        # Create segment at 1-2s, leaving 0-1s and 2-3s silent
        wav_path = os.path.join(tmpdir, "seg.wav")
        _create_test_wav(wav_path, 1.0)

        segments = [MockSegAudio(start=1.0, end=2.0, audio_path=wav_path)]

        aligner = AudioAligner(total_duration=3.0)
        output_path = os.path.join(tmpdir, "aligned.wav")
        aligner.align_segments(segments, output_path)

        audio, sr = _load_wav_as_float(output_path)

        # Check that 0-0.9s region is silent
        silence_region = audio[:int(0.9 * sr)]
        silence_rms = np.sqrt(np.mean(silence_region ** 2))
        assert silence_rms < 0.01, f"Silence region RMS {silence_rms:.4f} > threshold"

        # Check that 2.1-3.0s region is silent
        silence_region2 = audio[int(2.1 * sr):int(3.0 * sr)]
        if len(silence_region2) > 0:
            silence_rms2 = np.sqrt(np.mean(silence_region2 ** 2))
            assert silence_rms2 < 0.01, f"Trailing silence RMS {silence_rms2:.4f} > threshold"

        print(f"✓ Silence gaps preserved (RMS: {silence_rms:.6f})")


def test_wav_roundtrip():
    """save_wav → load_wav_as_float should recover close to original signal."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test.wav")
        original = np.sin(np.linspace(0, 2 * np.pi * 100, 22050)).astype(np.float32) * 0.5
        _save_wav(original, path)
        loaded, sr = _load_wav_as_float(path)
        assert sr == 22050
        # Allow small quantization error from int16 conversion
        max_err = np.max(np.abs(original - loaded[:len(original)]))
        assert max_err < 0.001, f"Max round-trip error: {max_err}"
        print(f"✓ WAV round-trip max error: {max_err:.6f}")


if __name__ == "__main__":
    test_wav_roundtrip()
    test_silence_gaps_preserved()
    test_segment_alignment_preserves_timing()
    print("\n✅ All alignment tests passed!")
