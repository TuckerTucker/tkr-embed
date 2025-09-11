#!/usr/bin/env python3
"""
Test script for MLX model loader
Tests basic functionality without loading the full 7B model
"""

import asyncio
import logging
from tkr_embed.core.model_manager import OpsMMEmbeddingMLX
from tkr_embed.utils.memory_manager import MemoryManager

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


async def test_model_manager():
    """Test the MLX model manager"""
    logger.info("🚀 Testing MLX Model Manager")
    
    # Test memory manager first
    logger.info("Testing memory manager...")
    memory_mgr = MemoryManager()
    stats = memory_mgr.get_memory_stats()
    logger.info(f"Memory stats: {stats}")
    
    # Optimize for inference
    memory_mgr.optimize_for_inference()
    
    # Create model manager (but don't load the full model yet)
    logger.info("Creating model manager...")
    model = OpsMMEmbeddingMLX(
        model_path="OpenSearch-AI/Ops-MM-embedding-v1-7B",
        quantization="auto"
    )
    
    # Test configuration detection
    logger.info(f"Detected quantization: {model.quantization}")
    logger.info(f"Model info: {model.get_model_info()}")
    
    # Test if model is ready (should be False before loading)
    logger.info(f"Model ready: {model.is_ready()}")
    
    logger.info("✅ Basic model manager tests passed!")
    
    # Note: We're not loading the actual model yet to avoid long download times
    # This will be done in the next phase
    logger.info("Note: Full model loading will be tested in the next phase")


if __name__ == "__main__":
    asyncio.run(test_model_manager())