"""
lip_sync.py
Crop-based lip sync engine operating on 256×256 face crops.
Never touches the 1080p background — all processing on small tensors only.
"""

import numpy as np
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)


class LipSyncEngine:
    """
    Lip-sync using Wav2Lip operating on 256×256 face crops only.
    
    Input: face crops from FaceTracker.extract_crop()
    Output: lip-synced face crops (same 256×256 size)
    
    The background pixels are NEVER processed — they remain pristine 1080p.
    This eliminates the old resolution degradation issue entirely.
    """

    def __init__(self, model_choice: str = "Wav2Lip", device: str = "cuda",
                 batch_size: int = 64):
        """
        Args:
            model_choice: "Wav2Lip" or "VideoReTalking"
            device: CUDA device string
            batch_size: Number of crops per GPU batch (64 is safe for 256×256 on 16GB)
        """
        self.device = device
        self.model_choice = model_choice
        self.batch_size = batch_size
        self._model = None

    def load(self):
        """Load the lip-sync model into VRAM."""
        logger.info(f"Loading {self.model_choice} on {self.device} (FP16)")
        # In production, load actual model weights:
        # self._model = load_wav2lip_model(self.device)
        # self._model = self._model.half()
        self._model = True  # Placeholder
        return self

    def sync_crops(self, crops: List[np.ndarray], audio_path: str,
                   fps: float) -> List[np.ndarray]:
        """
        Apply lip-sync to a list of 256×256 face crops.
        
        Args:
            crops: List of (256, 256, 3) uint8 face crop arrays
            audio_path: Path to aligned audio WAV
            fps: Video frame rate for audio-visual alignment
            
        Returns:
            List of lip-synced (256, 256, 3) uint8 crops
        """
        assert self._model is not None, "Call .load() first"

        if not crops:
            return []

        logger.info(f"Lip-syncing {len(crops)} face crops at {fps:.1f}fps")

        # Extract audio features for alignment
        audio_features = self._extract_audio_features(audio_path, len(crops), fps)

        synced_crops = []

        # Process in GPU-safe batches
        for batch_start in range(0, len(crops), self.batch_size):
            batch_end = min(batch_start + self.batch_size, len(crops))
            batch_crops = crops[batch_start:batch_end]
            batch_audio = audio_features[batch_start:batch_end] if audio_features is not None else None

            logger.debug(f"Processing batch [{batch_start}:{batch_end}] "
                          f"({len(batch_crops)} crops)")

            # --- Actual Wav2Lip inference would go here ---
            # crop_tensor = torch.from_numpy(np.stack(batch_crops)).permute(0,3,1,2).float()
            # crop_tensor = crop_tensor.half().to(self.device) / 255.0
            # audio_tensor = batch_audio.to(self.device) if batch_audio else None
            # with torch.no_grad():
            #     synced = self._model(crop_tensor, audio_tensor)
            # synced_np = (synced.permute(0,2,3,1).cpu().numpy() * 255).astype(np.uint8)
            # synced_crops.extend([s for s in synced_np])

            # Placeholder: pass-through crops (replace with actual inference)
            for crop in batch_crops:
                synced_crops.append(crop.copy())

        logger.info(f"Lip-sync complete: {len(synced_crops)} crops processed")
        return synced_crops

    def _extract_audio_features(self, audio_path: str, n_frames: int,
                                 fps: float) -> Optional[np.ndarray]:
        """
        Extract mel-spectrogram features aligned to video frame count.
        Each frame gets a mel window for audio-visual correspondence.
        """
        try:
            import librosa
            audio, sr = librosa.load(audio_path, sr=16000)
            mel = librosa.feature.melspectrogram(
                y=audio, sr=sr, n_mels=80,
                hop_length=int(sr / fps),  # Align mel windows to frame rate
                n_fft=1024,
            )
            # mel shape: (80, n_frames_approx)
            mel = mel.T  # (n_frames_approx, 80)

            # Trim or pad to match frame count
            if len(mel) > n_frames:
                mel = mel[:n_frames]
            elif len(mel) < n_frames:
                pad = np.zeros((n_frames - len(mel), mel.shape[1]))
                mel = np.vstack([mel, pad])

            return mel
        except Exception as e:
            logger.warning(f"Could not extract audio features: {e}")
            return None
