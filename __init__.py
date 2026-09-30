"""ComfyUI-MiniMax-FusedSwiGLU: Ultra-Low VRAM Fused Stream SwiGLU for MiniMax H3."""

import logging
from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
from .fused_swiglu import HAS_TRITON

log = logging.getLogger("MiniMax-FusedSwiGLU")

if HAS_TRITON:
    log.info("ComfyUI-MiniMax-FusedSwiGLU loaded successfully with native Triton acceleration.")
else:
    log.info("ComfyUI-MiniMax-FusedSwiGLU loaded with PyTorch micro-stream fallback.")

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
