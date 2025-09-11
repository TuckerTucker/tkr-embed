"""
FastAPI server for MLX multimodal embedding service
"""

import logging
import time
import asyncio
import tempfile
import os
from contextlib import asynccontextmanager
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import mlx.core as mx
import numpy as np
from PIL import Image

from tkr_embed.core.model_manager import OpsMMEmbeddingMLX
from tkr_embed.utils.memory_manager import MemoryManager
from tkr_embed.api.models import (
    TextEmbeddingRequest, MultimodalEmbeddingRequest, EmbeddingResponse,
    HealthResponse, ModelInfoResponse, SimilarityRequest, SimilarityResponse,
    ErrorResponse
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global instances
model_instance: Optional[OpsMMEmbeddingMLX] = None
memory_manager: Optional[MemoryManager] = None
server_start_time: float = 0


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage model lifecycle during server startup/shutdown"""
    global model_instance, memory_manager, server_start_time
    
    logger.info("🚀 Starting MLX Multimodal Embedding Server")
    server_start_time = time.time()
    
    try:
        # Initialize memory manager
        logger.info("Initializing memory manager...")
        memory_manager = MemoryManager()
        memory_manager.optimize_for_inference()
        
        # Initialize model
        logger.info("Loading multimodal embedding model...")
        quantization = memory_manager.memory_profile["quantization"]
        
        model_instance = OpsMMEmbeddingMLX(
            model_path="OpenSearch-AI/Ops-MM-embedding-v1-7B",
            quantization=quantization
        )
        
        # Note: For now we'll skip the actual model loading to avoid download times
        # await model_instance.load_model()
        logger.info("✅ Server startup complete (model loading deferred)")
        
        yield
        
    except Exception as e:
        logger.error(f"Failed to initialize server: {e}")
        raise
    
    finally:
        # Cleanup
        logger.info("👋 Shutting down server...")
        if model_instance:
            del model_instance
        if memory_manager:
            memory_manager.cleanup_memory()
        logger.info("Cleanup complete")


# Create FastAPI app
app = FastAPI(
    title="MLX Multimodal Embedding Server",
    description="High-performance multimodal embedding service optimized for Apple Silicon",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal server error",
            detail=str(exc)
        ).model_dump()
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    try:
        memory_stats = memory_manager.get_memory_stats() if memory_manager else {}
        uptime = time.time() - server_start_time
        
        return HealthResponse(
            status="healthy" if model_instance else "initializing",
            model_loaded=model_instance is not None and model_instance.is_ready(),
            framework="MLX",
            device="Apple Silicon GPU",
            memory_usage_gb=memory_stats.get("process_memory_gb", 0.0),
            uptime_seconds=uptime
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/info", response_model=ModelInfoResponse)
async def model_info():
    """Get model information and capabilities"""
    try:
        if not model_instance:
            raise HTTPException(status_code=503, detail="Model not initialized")
        
        model_info_dict = model_instance.get_model_info()
        
        return ModelInfoResponse(
            model_path=model_info_dict["model_path"],
            framework="MLX",
            mlx_version=mx.__version__ if hasattr(mx, '__version__') else "0.29.0",
            quantization=model_info_dict["quantization"],
            embedding_dim=model_info_dict["embedding_dim"],
            supported_modalities=model_info_dict["supported_modalities"],
            load_time=model_info_dict.get("load_time"),
            memory_usage_gb=model_info_dict.get("memory_usage_gb")
        )
    except Exception as e:
        logger.error(f"Model info request failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/embed/text", response_model=EmbeddingResponse)
async def embed_text(request: TextEmbeddingRequest):
    """Generate embeddings for text inputs"""
    start_time = time.time()
    
    try:
        if not model_instance:
            raise HTTPException(status_code=503, detail="Model not initialized")
        
        # Use mock embeddings for now (real model loading deferred)
        logger.info(f"Generating mock embeddings for {len(request.texts)} texts")
        
        logger.info(f"Processing {len(request.texts)} text inputs")
        
        # For now, return mock embeddings since we haven't loaded the actual model
        # embeddings = model_instance.encode_text(request.texts)
        
        # Mock embeddings for testing
        embedding_dim = 1024
        mock_embeddings = []
        for text in request.texts:
            # Create deterministic mock embedding based on text hash
            import hashlib
            text_hash = int(hashlib.md5(text.encode()).hexdigest()[:8], 16)
            np.random.seed(text_hash % (2**32))
            embedding = np.random.randn(embedding_dim).astype(np.float32)
            if request.normalize:
                embedding = embedding / np.linalg.norm(embedding)
            mock_embeddings.append(embedding.tolist())
        
        processing_time = time.time() - start_time
        
        return EmbeddingResponse(
            embeddings=mock_embeddings,
            shape=[len(request.texts), embedding_dim],
            model="Ops-MM-embedding-v1-7B",
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"Text embedding failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/embed/image", response_model=EmbeddingResponse)
async def embed_image(file: UploadFile = File(...)):
    """Generate embeddings for image inputs"""
    start_time = time.time()
    
    try:
        if not model_instance:
            raise HTTPException(status_code=503, detail="Model not initialized")
        
        # Validate file type
        if not file.content_type or not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        logger.info(f"Processing image: {file.filename}")
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        try:
            # Process image
            image = Image.open(tmp_path).convert("RGB")
            image = image.resize((336, 336))  # Standard size for vision models
            
            # For now, return mock embedding
            # image_array = np.array(image).astype(np.float32) / 255.0
            # embeddings = model_instance.encode_image(mx.array(image_array))
            
            # Mock image embedding
            embedding_dim = 1024
            import hashlib
            img_hash = int(hashlib.md5(content).hexdigest()[:8], 16)
            np.random.seed(img_hash % (2**32))
            embedding = np.random.randn(embedding_dim).astype(np.float32)
            embedding = embedding / np.linalg.norm(embedding)  # Normalize
            
            processing_time = time.time() - start_time
            
            return EmbeddingResponse(
                embeddings=[embedding.tolist()],
                shape=[1, embedding_dim],
                model="Ops-MM-embedding-v1-7B",
                processing_time=processing_time
            )
            
        finally:
            # Cleanup temp file
            os.unlink(tmp_path)
            
    except Exception as e:
        logger.error(f"Image embedding failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/embed/multimodal", response_model=EmbeddingResponse)
async def embed_multimodal(
    text: Optional[str] = None,
    image: Optional[UploadFile] = File(None)
):
    """Generate unified multimodal embeddings"""
    start_time = time.time()
    
    try:
        if not model_instance:
            raise HTTPException(status_code=503, detail="Model not initialized")
        
        if not text and not image:
            raise HTTPException(status_code=400, detail="At least one of text or image must be provided")
        
        logger.info(f"Processing multimodal input - text: {text is not None}, image: {image is not None}")
        
        # Mock multimodal embedding
        embedding_dim = 1024
        seed = 42
        
        if text and image:
            # Combined text + image embedding
            content = await image.read() if image else b""
            combined_hash = int(hashlib.sha256(text.encode() + content).hexdigest()[:8], 16)
            np.random.seed(combined_hash % (2**32))
        elif text:
            # Text only
            text_hash = int(hashlib.md5(text.encode()).hexdigest()[:8], 16)
            np.random.seed(text_hash % (2**32))
        else:
            # Image only
            content = await image.read()
            img_hash = int(hashlib.md5(content).hexdigest()[:8], 16)
            np.random.seed(img_hash % (2**32))
        
        embedding = np.random.randn(embedding_dim).astype(np.float32)
        embedding = embedding / np.linalg.norm(embedding)  # Normalize
        
        processing_time = time.time() - start_time
        
        return EmbeddingResponse(
            embeddings=[embedding.tolist()],
            shape=[1, embedding_dim],
            model="Ops-MM-embedding-v1-7B",
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"Multimodal embedding failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/similarity", response_model=SimilarityResponse)
async def compute_similarity(request: SimilarityRequest):
    """Compute similarity between embeddings"""
    start_time = time.time()
    
    try:
        query_emb = np.array(request.query_embeddings, dtype=np.float32)
        candidate_emb = np.array(request.candidate_embeddings, dtype=np.float32)
        
        if request.metric == "cosine":
            # Cosine similarity
            similarities = np.dot(query_emb, candidate_emb.T)
        elif request.metric == "euclidean":
            # Euclidean distance (converted to similarity)
            distances = np.sqrt(np.sum((query_emb[:, None] - candidate_emb[None, :]) ** 2, axis=2))
            similarities = 1.0 / (1.0 + distances)  # Convert distance to similarity
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported metric: {request.metric}")
        
        processing_time = time.time() - start_time
        
        return SimilarityResponse(
            similarities=similarities.tolist(),
            shape=list(similarities.shape),
            metric=request.metric,
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"Similarity computation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    # Run server
    uvicorn.run(
        "tkr_embed.api.server:app",
        host="0.0.0.0",
        port=8000,
        log_level="info",
        reload=False
    )