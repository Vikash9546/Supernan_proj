"""
gpu_manager.py
Stage-based model lifecycle manager.
Loads models once per stage, keeps them alive across all segments,
frees them only when the stage completes.
"""

import torch
import gc
import logging
from contextlib import contextmanager
from typing import Callable, Any, Dict

logger = logging.getLogger(__name__)


class StageContext:
    """
    Manages GPU model lifecycle for a pipeline stage.
    
    Usage:
        with StageContext("nlp") as ctx:
            whisper = ctx.load_model("whisper", lambda: WhisperModel(...))
            # whisper stays in VRAM for all segments
            for seg in segments:
                result = whisper.transcribe(seg)
        # All models freed here automatically
    """

    def __init__(self, stage_name: str, device: str = "cuda"):
        self.stage_name = stage_name
        self.device = device
        self._models: Dict[str, Any] = {}

    def __enter__(self):
        logger.info(f"[Stage:{self.stage_name}] Starting — "
                     f"VRAM before: {self._vram_str()}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.unload_all()
        logger.info(f"[Stage:{self.stage_name}] Completed — "
                     f"VRAM after cleanup: {self._vram_str()}")
        return False

    def load_model(self, name: str, factory_fn: Callable[[], Any]) -> Any:
        """
        Load a model by name. If already loaded, return cached instance.
        factory_fn is only called on first load.
        """
        if name in self._models:
            logger.debug(f"[Stage:{self.stage_name}] Model '{name}' already loaded, reusing")
            return self._models[name]

        logger.info(f"[Stage:{self.stage_name}] Loading model '{name}' — "
                     f"VRAM before: {self._vram_str()}")
        model = factory_fn()
        self._models[name] = model
        logger.info(f"[Stage:{self.stage_name}] Model '{name}' loaded — "
                     f"VRAM after: {self._vram_str()}")
        return model

    def unload_model(self, name: str) -> None:
        """Unload a specific model to free VRAM mid-stage if needed."""
        if name in self._models:
            model = self._models.pop(name)
            del model
            self._gc_flush()
            logger.info(f"[Stage:{self.stage_name}] Unloaded '{name}' — "
                         f"VRAM: {self._vram_str()}")

    def unload_all(self) -> None:
        """Delete all cached models and aggressively free VRAM."""
        names = list(self._models.keys())
        for name in names:
            del self._models[name]
        self._models.clear()
        self._gc_flush()
        logger.info(f"[Stage:{self.stage_name}] All models unloaded — "
                     f"VRAM: {self._vram_str()}")

    @staticmethod
    def _gc_flush():
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    @staticmethod
    def _vram_str() -> str:
        if not torch.cuda.is_available():
            return "N/A (CPU mode)"
        alloc = torch.cuda.memory_allocated() / (1024 ** 2)
        reserved = torch.cuda.memory_reserved() / (1024 ** 2)
        return f"Allocated {alloc:.0f}MB / Reserved {reserved:.0f}MB"


# Backwards-compatible helpers (kept for any code still referencing them)
def clear_memory(*args):
    """Explicitly deletes passed objects and flushes GPU cache."""
    for obj in args:
        del obj
    StageContext._gc_flush()


@contextmanager
def gpu_context():
    """Legacy context manager — wraps StageContext for backward compat."""
    with StageContext("legacy") as ctx:
        yield ctx
