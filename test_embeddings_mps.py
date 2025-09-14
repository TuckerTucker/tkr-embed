#!/usr/bin/env python3
"""Test embedding generation with MPS acceleration"""

import asyncio
import torch
import time
from tkr_embed.core.model_manager import OpsMMEmbeddingMLX

async def test_embeddings():
    print("Testing OpenSearch-AI/Ops-MM-embedding-v1-7B with MPS...")
    print("=" * 60)

    # Initialize model with MPS and quantization for efficiency
    model = OpsMMEmbeddingMLX(
        model_path="OpenSearch-AI/Ops-MM-embedding-v1-7B",
        quantization="q8",  # Use 8-bit quantization for faster loading
        device="gpu"  # Will use MPS if available
    )

    print("Loading model...")
    start = time.time()
    await model.load_model()
    print(f"Model loaded in {time.time() - start:.1f}s")
    print(f"Using device: {model.device}")
    print("-" * 60)

    # Test text embedding
    print("\nTesting text embedding...")
    try:
        texts = ["Hello, this is a test of the embedding model with Metal GPU acceleration."]
        start = time.time()
        embeddings = model.encode_text(texts)
        elapsed = time.time() - start

        print(f"✅ Text embedding successful!")
        print(f"   Shape: {embeddings.shape}")
        print(f"   Time: {elapsed:.3f}s")
        print(f"   Device: {model.device}")
    except Exception as e:
        print(f"❌ Text embedding failed: {e}")

    print("\nModel info:")
    info = model.get_model_info()
    for key, value in info.items():
        print(f"  {key}: {value}")

if __name__ == "__main__":
    asyncio.run(test_embeddings())