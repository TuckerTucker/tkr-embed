"""
tkr_embed - tkr-embed | MLX Multimodal Embedding Server
Optimized for Apple Silicon (M1/M2/M3) with OpenSearch-AI/Ops-MM-embedding-v1-7B
"""

__version__ = "0.1.0"
__author__ = "Tucker"

from .core.model_manager import OpsMMEmbeddingMLX
from .utils.memory_manager import MemoryManager

__all__ = [
    "OpsMMEmbeddingMLX",
    "MemoryManager"
]