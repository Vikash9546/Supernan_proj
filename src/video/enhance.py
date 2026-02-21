"""
enhance.py
CodeFormer face enhancement on 256×256 crops ONLY.
Never processes full 1080p frames — background stays pristine.
"""

import numpy as np
import logging
from typing import List

logger = logging.getLogger(__name__)


class FaceEnhancer:
    """
    Applies CodeFormer face restoration to 256×256 lip-synced crops.
    
    Key difference from old GFPGAN approach:
    - OLD: GFPGAN on full 1080p frame → corrupted background, extreme slowness, OOM
    - NEW: CodeFormer on 256×256 crop → fast, memory-efficient, background untouched
    """

    def __init__(self, model_choice: str = "CodeFormer", device: str = "cuda",
                 fidelity_weight: float = 0.7, batch_size: int = 32):
        """
        Args:
            model_choice: "CodeFormer" or "GFPGAN" (CodeFormer strongly recommended)
            device: CUDA device string
            fidelity_weight: CodeFormer fidelity weight (0=quality, 1=fidelity).
                            0.7 preserves identity while improving lip region.
            batch_size: Number of crops per GPU batch
        """
        self.device = device
        self.model_choice = model_choice
        self.fidelity_weight = fidelity_weight
        self.batch_size = batch_size
        self._model = None

    def load(self):
        """Load the face enhancement model into VRAM."""
        logger.info(f"Loading {self.model_choice} on {self.device} "
                     f"(fidelity={self.fidelity_weight})")

        # In production, load CodeFormer:
        # from codeformer import CodeFormerModel
        # self._model = CodeFormerModel(
        #     fidelity_weight=self.fidelity_weight,
        #     device=self.device
        # )
        self._model = True  # Placeholder
        return self

    def enhance_crops(self, crops: List[np.ndarray]) -> List[np.ndarray]:
        """
        Enhance a list of 256×256 face crops.
        
        Args:
            crops: List of (256, 256, 3) uint8 arrays (lip-synced face crops)
            
        Returns:
            List of enhanced (256, 256, 3) uint8 arrays
        """
        assert self._model is not None, "Call .load() first"

        if not crops:
            return []

        logger.info(f"Enhancing {len(crops)} face crops with {self.model_choice}")
        enhanced = []

        for batch_start in range(0, len(crops), self.batch_size):
            batch_end = min(batch_start + self.batch_size, len(crops))
            batch = crops[batch_start:batch_end]

            logger.debug(f"Enhancement batch [{batch_start}:{batch_end}]")

            # --- Actual CodeFormer inference would go here ---
            # crop_tensor = torch.from_numpy(np.stack(batch)).permute(0,3,1,2).float()
            # crop_tensor = crop_tensor.to(self.device) / 255.0
            # with torch.no_grad():
            #     output = self._model(crop_tensor, w=self.fidelity_weight)
            # output_np = (output.permute(0,2,3,1).cpu().numpy() * 255).clip(0,255).astype(np.uint8)
            # enhanced.extend([o for o in output_np])

            # Placeholder: return copies (replace with actual inference)
            for crop in batch:
                enhanced.append(crop.copy())

        logger.info(f"Enhancement complete: {len(enhanced)} crops")
        return enhanced

    def enhance(self, frames: np.ndarray) -> np.ndarray:
        """
        Legacy API for backward compatibility.
        WARNING: This processes full frames — use enhance_crops() for the new pipeline.
        """
        logger.warning("enhance() called on full frames — use enhance_crops() for crop-only mode")
        return frames.copy()
