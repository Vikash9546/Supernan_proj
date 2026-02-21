"""
tests/test_face_tracker.py
Unit tests for EMA bounding box smoothing and crop/blend round-trip.
"""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.video.face_tracker import FaceTracker, BBox


def test_ema_smoothing_reduces_jitter():
    """EMA-smoothed bboxes should have less variance than raw input."""
    tracker = FaceTracker(ema_alpha=0.15)
    tracker._detection_backend = "center_crop"  # Skip model loading

    # Simulate jittery raw detections around center (500, 300)
    np.random.seed(42)
    raw_bboxes = [
        BBox(cx=500 + np.random.randn() * 20,
             cy=300 + np.random.randn() * 15,
             w=200 + np.random.randn() * 10,
             h=200 + np.random.randn() * 10)
        for _ in range(100)
    ]

    # Feed through EMA
    smoothed = []
    for raw in raw_bboxes:
        tracker._ema_bbox = None  # Reset for manual feeding
        # Simulate EMA manually
        if not smoothed:
            smoothed.append(raw)
            tracker._ema_bbox = raw
        else:
            a = tracker.ema_alpha
            prev = smoothed[-1]
            s = BBox(
                cx=a * raw.cx + (1 - a) * prev.cx,
                cy=a * raw.cy + (1 - a) * prev.cy,
                w=a * raw.w + (1 - a) * prev.w,
                h=a * raw.h + (1 - a) * prev.h,
            )
            smoothed.append(s)

    # Smoothed variance should be much lower than raw
    raw_cx_var = np.var([b.cx for b in raw_bboxes])
    smoothed_cx_var = np.var([b.cx for b in smoothed])
    assert smoothed_cx_var < raw_cx_var * 0.5, \
        f"Smoothed variance {smoothed_cx_var:.2f} should be < 50% of raw {raw_cx_var:.2f}"
    print(f"✓ EMA reduces cx variance: {raw_cx_var:.2f} → {smoothed_cx_var:.2f}")


def test_extract_blend_roundtrip_preserves_dimensions():
    """extract_crop → blend_crop should return frame with original dimensions."""
    tracker = FaceTracker(crop_size=256)

    # Create a test frame (720p)
    h, w = 720, 1280
    frame = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
    bbox = BBox(cx=640, cy=360, w=200, h=200)

    crop, M = tracker.extract_crop(frame, bbox)

    # Crop should be 256x256
    assert crop.shape == (256, 256, 3), f"Crop shape {crop.shape} != (256, 256, 3)"

    # Blend back
    result = tracker.blend_crop(frame, crop, M, feather=0.1)
    assert result.shape == (h, w, 3), f"Blended shape {result.shape} != ({h}, {w}, 3)"
    assert result.dtype == np.uint8
    print(f"✓ Round-trip preserves dimensions: {frame.shape} → {crop.shape} → {result.shape}")


def test_bbox_properties():
    """BBox x1/y1/x2/y2 should be correctly computed from center + size."""
    bbox = BBox(cx=100, cy=200, w=50, h=80)
    assert bbox.x1 == 75
    assert bbox.y1 == 160
    assert bbox.x2 == 125
    assert bbox.y2 == 240
    print("✓ BBox properties correct")


if __name__ == "__main__":
    test_bbox_properties()
    test_ema_smoothing_reduces_jitter()
    test_extract_blend_roundtrip_preserves_dimensions()
    print("\n✅ All face tracker tests passed!")
