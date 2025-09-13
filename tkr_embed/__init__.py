"""
tkr_embed - GPT-OSS-20B Text Generation Server
Optimized for Apple Silicon (M1/M2/M3) with GPT-OSS-20B model
"""

__version__ = "0.1.0"
__author__ = "Tucker"

try:
    from .core.model_manager import GPTOss20bMLX
except ImportError:
    # Fallback during development
    GPTOss20bMLX = None

from .utils.memory_manager import MemoryManager

__all__ = [
    "GPTOss20bMLX",
    "MemoryManager"
]