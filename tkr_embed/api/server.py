"""
FastAPI server for MLX multimodal embedding service
Production-ready with authentication, rate limiting, and error handling
"""

import logging
import time
import asyncio
import tempfile
import os
from contextlib import asynccontextmanager
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, UploadFile, File, HTTPException, Request, Depends
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import mlx.core as mx
import numpy as np
from PIL import Image

# Core components
from tkr_embed.core.model_manager import OpsMMEmbeddingMLX
from tkr_embed.utils.memory_manager import MemoryManager
from tkr_embed.utils.lru_cache import CachedEmbeddingProcessor, EmbeddingCache
from tkr_embed.core.batch_processor import BatchProcessor, BatchConfig

# Production features
from tkr_embed.config import get_config
from tkr_embed.api.auth import authenticate, optional_auth, APIKey, create_api_key
from tkr_embed.api.rate_limiter import apply_rate_limit, create_rate_limit_headers, rate_limiter
from tkr_embed.api.error_handlers import setup_error_handlers, SafeModelOperation, raise_model_not_ready

# API models
from tkr_embed.api.models import (
    TextEmbeddingRequest, MultimodalEmbeddingRequest, EmbeddingResponse,
    HealthResponse, ModelInfoResponse, SimilarityRequest, SimilarityResponse,
    ErrorResponse
)

# Admin endpoints
from tkr_embed.api.admin import admin_router, setup_admin_key

# Load configuration
config = get_config()

# Configure logging based on config
logging.basicConfig(
    level=getattr(logging, config.logging.level.upper()),
    format=config.logging.format
)
logger = logging.getLogger(__name__)

# Global instances
model_instance: Optional[OpsMMEmbeddingMLX] = None
memory_manager: Optional[MemoryManager] = None
cached_processor: Optional[CachedEmbeddingProcessor] = None
batch_processor: Optional[BatchProcessor] = None
server_start_time: float = 0


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage model lifecycle and production components during server startup/shutdown"""
    global model_instance, memory_manager, cached_processor, batch_processor, server_start_time
    
    logger.info("🚀 Starting tkr-embed | MLX Multimodal Embedding Server (Production Mode)")
    server_start_time = time.time()
    
    try:
        # Initialize memory manager
        logger.info("Initializing memory manager...")
        memory_manager = MemoryManager()
        memory_manager.optimize_for_inference()
        
        # Initialize model with configuration
        logger.info("Loading multimodal embedding model...")
        model_instance = OpsMMEmbeddingMLX(
            model_path=config.model.model_path,
            quantization=config.model.quantization,
            cache_dir=config.model.cache_dir
        )
        
        # Load model if not in testing mode
        if config.environment.value != "testing":
            logger.info("Loading model (this may take a few minutes)...")
            await model_instance.load_model()
            logger.info("✅ Model loaded successfully")
        else:
            logger.info("Skipping model loading in testing mode")
        
        # Initialize production components
        if config.cache.enabled:
            logger.info("Initializing embedding cache...")
            cache = EmbeddingCache(
                max_size=config.cache.max_size,
                ttl=config.cache.ttl_seconds
            )
            cached_processor = CachedEmbeddingProcessor(model_instance, cache)
            logger.info("✅ Embedding cache initialized")
        
        # Initialize batch processor
        logger.info("Initializing batch processor...")
        batch_config = BatchConfig(
            max_batch_size=8,
            dynamic_batching=True
        )
        batch_processor = BatchProcessor(model_instance, batch_config)
        logger.info("✅ Batch processor initialized")
        
        # Start rate limiter cleanup task
        if config.rate_limit.enabled:
            logger.info("Starting rate limiter...")
            rate_limiter.start_cleanup_task()
            logger.info("✅ Rate limiter started")
        
        # Setup initial admin key if needed
        setup_admin_key()
        
        startup_time = time.time() - server_start_time
        logger.info(f"✅ Server startup complete in {startup_time:.1f}s")
        
        yield
        
    except Exception as e:
        logger.error(f"Failed to initialize server: {e}")
        raise
    
    finally:
        # Cleanup
        logger.info("👋 Shutting down server...")
        
        # Stop rate limiter
        if config.rate_limit.enabled:
            rate_limiter.stop_cleanup_task()
            
        # Cleanup model
        if model_instance:
            del model_instance
            
        # Cleanup memory manager
        if memory_manager:
            memory_manager.cleanup_memory()
            
        logger.info("Cleanup complete")


# Create FastAPI app with production configuration
app = FastAPI(
    title="tkr-embed | MLX Multimodal Embedding Server",
    description="Production-ready multimodal embedding service optimized for Apple Silicon",
    version="1.0.0",
    docs_url="/docs",  # Always enable docs for development
    redoc_url="/redoc",  # Always enable docs for development
    lifespan=lifespan,
    debug=config.debug
)

# Setup comprehensive error handling
setup_error_handlers(app, include_traceback=config.debug)

# Add CORS middleware with configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.security.cors_origins,
    allow_credentials=True,
    allow_methods=config.security.cors_methods,
    allow_headers=config.security.cors_headers,
)

# Include admin router
app.include_router(admin_router)


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
async def embed_text(
    request: TextEmbeddingRequest,
    http_request: Request,
    api_key: APIKey = Depends(authenticate if config.security.require_api_key else optional_auth)
):
    """Generate embeddings for text inputs (requires authentication)"""
    start_time = time.time()
    
    # Apply rate limiting
    if config.rate_limit.enabled:
        identifier = f"{api_key.name if api_key else 'anonymous'}:{http_request.client.host if http_request.client else 'unknown'}"
        await apply_rate_limit(http_request, identifier)
    
    try:
        # Safe model operation with error handling
        with SafeModelOperation("text_embedding", model_instance):
            if not model_instance or not model_instance.is_ready():
                raise_model_not_ready()
                
            logger.info(f"Processing {len(request.texts)} text inputs for {api_key.name if api_key else 'anonymous'}")
            
            # Use cached processor if available, otherwise direct model
            if cached_processor and config.cache.enabled:
                embeddings_array = cached_processor.encode_text(request.texts)
            else:
                embeddings_array = model_instance.encode_text(request.texts)
        
        # Convert to list format and apply normalization if requested
        embeddings_list = []
        for i in range(embeddings_array.shape[0]):
            embedding = embeddings_array[i]
            if request.normalize:
                embedding = embedding / np.linalg.norm(embedding)
            embeddings_list.append(embedding.tolist())
        
        processing_time = time.time() - start_time
        
        # Create response with rate limit headers
        response_data = EmbeddingResponse(
            embeddings=embeddings_list,
            shape=[len(request.texts), embeddings_array.shape[1]],
            model="Ops-MM-embedding-v1-7B",
            processing_time=processing_time
        )
        
        # Add rate limiting headers if enabled
        if config.rate_limit.enabled and api_key:
            rate_headers = create_rate_limit_headers(f"{api_key.name}:{http_request.client.host if http_request.client else 'unknown'}")
            return JSONResponse(
                content=response_data.model_dump(),
                headers=rate_headers
            )
        
        return response_data
        
    except Exception as e:
        logger.error(f"Text embedding failed: {e}")
        # The error will be handled by the comprehensive error handler
        raise


@app.post("/embed/image", response_model=EmbeddingResponse)
async def embed_image(file: UploadFile = File(...)):
    """Generate embeddings for image inputs"""
    start_time = time.time()
    
    try:
        if not model_instance:
            raise HTTPException(status_code=503, detail="Model not initialized")
        
        if not model_instance.is_ready():
            raise HTTPException(status_code=503, detail="Model not ready")
        
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
            # Process image with real model
            logger.info(f"Generating real image embedding for {file.filename}")
            embeddings_array = model_instance.encode_image(tmp_path)
            
            # Convert to list format and normalize if needed
            embedding = embeddings_array[0]  # Single image
            embedding = embedding / np.linalg.norm(embedding)  # Always normalize images
            
            processing_time = time.time() - start_time
            
            return EmbeddingResponse(
                embeddings=[embedding.tolist()],
                shape=[1, len(embedding)],
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
        
        if not model_instance.is_ready():
            raise HTTPException(status_code=503, detail="Model not ready")
        
        if not text and not image:
            raise HTTPException(status_code=400, detail="At least one of text or image must be provided")
        
        logger.info(f"Processing multimodal input - text: {text is not None}, image: {image is not None}")
        
        # Handle image path if provided
        tmp_path = None
        if image:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                content = await image.read()
                tmp.write(content)
                tmp_path = tmp.name
        
        try:
            # Generate real multimodal embedding
            logger.info("Generating real multimodal embedding")
            embeddings_array = model_instance.encode_multimodal(
                text=text,
                image_path=tmp_path
            )
            
            # Convert to list format and normalize
            embedding = embeddings_array[0]  # Single multimodal embedding
            embedding = embedding / np.linalg.norm(embedding)  # Normalize
            
        finally:
            # Cleanup temp file if created
            if tmp_path:
                os.unlink(tmp_path)
        
        processing_time = time.time() - start_time
        
        return EmbeddingResponse(
            embeddings=[embedding.tolist()],
            shape=[1, len(embedding)],
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