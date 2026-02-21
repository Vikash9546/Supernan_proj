"""
face_tracker.py
EMA-smoothed face tracking with crop extraction and alpha blending.
Prevents bounding box jitter that causes temporal flickering.
"""

import numpy as np
import logging
import cv2
from typing import Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class BBox:
    """Bounding box with center coordinates and size."""
    cx: float  # Center x
    cy: float  # Center y
    w: float   # Width
    h: float   # Height

    @property
    def x1(self) -> int:
        return int(self.cx - self.w / 2)

    @property
    def y1(self) -> int:
        return int(self.cy - self.h / 2)

    @property
    def x2(self) -> int:
        return int(self.cx + self.w / 2)

    @property
    def y2(self) -> int:
        return int(self.cy + self.h / 2)


class FaceTracker:
    """
    Detects face bounding boxes and applies EMA smoothing
    to prevent inter-frame jitter that causes temporal flickering.
    
    Uses MediaPipe Face Detection (no heavy model needed) or falls 
    back to OpenCV's DNN face detector.
    """

    def __init__(self, ema_alpha: float = 0.15, crop_size: int = 256,
                 padding_ratio: float = 0.3, keyframe_interval: int = 30):
        """
        Args:
            ema_alpha: EMA smoothing factor (0=max smooth, 1=no smooth).
                       0.15 balances responsiveness with stability.
            crop_size: Output face crop size (square).
            padding_ratio: Extra padding around face as fraction of bbox size.
            keyframe_interval: How often to run full detection vs optical flow tracking.
        """
        self.ema_alpha = ema_alpha
        self.crop_size = crop_size
        self.padding_ratio = padding_ratio
        self.keyframe_interval = keyframe_interval
        self._ema_bbox: Optional[BBox] = None
        self._detector = None
        self._detection_backend = None
        self._frame_count = 0
        self._prev_gray = None
        self._tracking_pts = None

    def load(self):
        """Load the face detection model."""
        # Try MediaPipe first (lightweight, accurate)
        try:
            from mediapipe.python.solutions import face_detection as mp_face_detection
            self._detector = mp_face_detection.FaceDetection(
                model_selection=1,  # Full-range model
                min_detection_confidence=0.5,
            )
            self._detection_backend = "mediapipe"
            logger.info("Face tracker loaded: MediaPipe")
            return self
        except ImportError:
            pass

        # Fallback to OpenCV DNN
        try:
            prototxt = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self._detector = cv2.CascadeClassifier(prototxt)
            self._detection_backend = "opencv_haar"
            logger.info("Face tracker loaded: OpenCV Haar Cascade")
            return self
        except Exception:
            pass

        logger.warning("No face detector available, will use center crop fallback")
        self._detection_backend = "center_crop"
        return self

    def detect_raw(self, frame: np.ndarray) -> Optional[BBox]:
        """Detect face in frame without smoothing. Returns raw BBox or None."""
        h, w = frame.shape[:2]

        if self._detection_backend == "mediapipe":
            frame_rgb = frame if frame.shape[2] == 3 else cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self._detector.process(frame_rgb)

            if results.detections:
                det = results.detections[0]  # Use most confident face
                bb = det.location_data.relative_bounding_box
                cx = (bb.xmin + bb.width / 2) * w
                cy = (bb.ymin + bb.height / 2) * h
                bw = bb.width * w
                bh = bb.height * h
                return BBox(cx=cx, cy=cy, w=bw, h=bh)

        elif self._detection_backend == "opencv_haar":
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            faces = self._detector.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
            if len(faces) > 0:
                x, y, fw, fh = faces[0]
                return BBox(cx=x + fw / 2, cy=y + fh / 2, w=float(fw), h=float(fh))

        return None  # No face detected

    def track(self, frame: np.ndarray) -> BBox:
        """
        Detect face and apply EMA smoothing.
        Uses optical flow tracking to reduce full detection overhead.
        Falls back to last known position or center crop if no face found.
        """
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) if frame.shape[2] == 3 else frame

        # Check if keyframe or tracking lost
        if self._ema_bbox is None or self._frame_count % self.keyframe_interval == 0:
            raw_bbox = self.detect_raw(frame)
            if raw_bbox is not None:
                # Initialize tracking points from bbox center
                self._tracking_pts = np.array([[[raw_bbox.cx, raw_bbox.cy]]], dtype=np.float32)
            else:
                self._tracking_pts = None
        else:
            raw_bbox = None
            # Apply Optical flow
            if self._prev_gray is not None and self._tracking_pts is not None:
                new_pts, status, err = cv2.calcOpticalFlowPyrLK(
                    self._prev_gray, gray, self._tracking_pts, None,
                    winSize=(15, 15), maxLevel=2,
                    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
                )
                if status[0][0] == 1:
                    # Update raw_bbox based on optical flow delta
                    dx = new_pts[0][0][0] - self._tracking_pts[0][0][0]
                    dy = new_pts[0][0][1] - self._tracking_pts[0][0][1]
                    raw_bbox = BBox(
                        cx=self._ema_bbox.cx + dx,
                        cy=self._ema_bbox.cy + dy,
                        w=self._ema_bbox.w,
                        h=self._ema_bbox.h
                    )
                    self._tracking_pts = new_pts
                else:
                    self._tracking_pts = None

        self._prev_gray = gray
        self._frame_count += 1

        if raw_bbox is None:
            if self._ema_bbox is not None:
                # Use last known position (face temporarily occluded)
                return self._ema_bbox
            else:
                # No face ever detected — center crop fallback
                size = min(w, h) * 0.4
                return BBox(cx=w / 2, cy=h / 2, w=size, h=size)

        # Apply EMA smoothing
        if self._ema_bbox is None:
            self._ema_bbox = raw_bbox
        else:
            a = self.ema_alpha
            self._ema_bbox = BBox(
                cx=a * raw_bbox.cx + (1 - a) * self._ema_bbox.cx,
                cy=a * raw_bbox.cy + (1 - a) * self._ema_bbox.cy,
                w=a * raw_bbox.w + (1 - a) * self._ema_bbox.w,
                h=a * raw_bbox.h + (1 - a) * self._ema_bbox.h,
            )

        return self._ema_bbox

    def extract_crop(self, frame: np.ndarray, bbox: BBox
                     ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract a padded, square face crop and return the affine transform matrix.
        
        Returns:
            crop: (crop_size, crop_size, 3) uint8 array
            M: 2x3 affine transform matrix for blending back
        """
        h, w = frame.shape[:2]

        # Add padding around the bbox
        pad_w = bbox.w * (1 + self.padding_ratio)
        pad_h = bbox.h * (1 + self.padding_ratio)
        size = max(pad_w, pad_h)  # Square crop

        # Source points (padded bbox corners)
        src_pts = np.float32([
            [bbox.cx - size / 2, bbox.cy - size / 2],
            [bbox.cx + size / 2, bbox.cy - size / 2],
            [bbox.cx - size / 2, bbox.cy + size / 2],
        ])

        # Destination points (crop_size square)
        cs = self.crop_size
        dst_pts = np.float32([
            [0, 0],
            [cs, 0],
            [0, cs],
        ])

        # Compute affine transform
        M = cv2.getAffineTransform(src_pts, dst_pts)
        crop = cv2.warpAffine(frame, M, (cs, cs),
                               flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REFLECT)

        return crop, M

    def blend_crop(self, frame: np.ndarray, crop: np.ndarray,
                   M: np.ndarray, feather: float = 0.1) -> np.ndarray:
        """
        Alpha-blend the processed crop back into the original 1080p frame.
        Uses Gaussian-feathered mask to avoid hard edges.
        
        Args:
            frame: Original full-resolution frame (H, W, 3)
            crop: Processed face crop (crop_size, crop_size, 3)
            M: Affine transform from extract_crop
            feather: Feather width as fraction of crop size
            
        Returns:
            Composited frame with the processed face blended in.
        """
        h, w = frame.shape[:2]
        cs = self.crop_size

        # Create soft alpha mask for the crop
        mask = np.ones((cs, cs), dtype=np.float32)
        feather_px = max(int(cs * feather), 2)

        # Apply gradient feathering on all edges
        for i in range(feather_px):
            alpha = i / feather_px
            mask[i, :] *= alpha
            mask[cs - 1 - i, :] *= alpha
            mask[:, i] *= alpha
            mask[:, cs - 1 - i] *= alpha

        # Invert the affine transform to map crop back to original frame space
        M_inv = cv2.invertAffineTransform(M)

        # Warp the crop and mask back to frame coordinates
        warped_crop = cv2.warpAffine(crop, M_inv, (w, h),
                                      flags=cv2.INTER_LINEAR,
                                      borderValue=(0, 0, 0))
        warped_mask = cv2.warpAffine(mask, M_inv, (w, h),
                                      flags=cv2.INTER_LINEAR,
                                      borderValue=0)

        # Expand mask to 3 channels
        warped_mask_3ch = warped_mask[:, :, np.newaxis]

        # Alpha composite: result = crop * alpha + background * (1 - alpha)
        result = frame.astype(np.float32)
        result = warped_crop.astype(np.float32) * warped_mask_3ch + \
                 result * (1.0 - warped_mask_3ch)

        return np.clip(result, 0, 255).astype(np.uint8)

    def reset(self):
        """Reset EMA state between scenes or shots."""
        self._ema_bbox = None
        self._frame_count = 0
        self._prev_gray = None
        self._tracking_pts = None
