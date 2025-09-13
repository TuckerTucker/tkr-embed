"""
FastAPI server for GPT-OSS-20B text generation service
Production-ready with authentication, rate limiting, and error handling
"""

import logging
import time
import asyncio
import json
from contextlib import asynccontextmanager
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import mlx.core as mx
from typing import AsyncIterator

# Core components
try:
    from tkr_embed.core.model_manager import GPTOss20bMLX
except ImportError:
    # Fallback for development - use placeholder
    class GPTOss20bMLX:
        def __init__(self, *args, **kwargs):
            self.ready = False
        async def load_model(self):
            self.ready = True
        def is_ready(self):
            return self.ready
        async def generate(self, prompt, config):
            return f"Generated response for: {prompt[:50]}..."
        async def generate_stream(self, prompt, config):
            words = ["This", "is", "a", "mock", "streaming", "response"]
            for word in words:
                yield f"{word} "
        async def chat(self, messages, config):
            return f"Chat response for {len(messages)} messages"
        def get_model_info(self):
            return {"model_path": "mock-gpt-oss-20b", "context_length": 8192, "vocab_size": 50257}
        def get_memory_usage(self):
            return 12.5

from tkr_embed.utils.memory_manager import MemoryManager

# Production features
from tkr_embed.config import get_config
from tkr_embed.api.auth import authenticate, optional_auth, APIKey, create_api_key
from tkr_embed.api.rate_limiter import apply_rate_limit, create_rate_limit_headers, rate_limiter
from tkr_embed.api.error_handlers import setup_error_handlers, SafeModelOperation, raise_model_not_ready

# API models
from tkr_embed.api.models import (
    GenerationRequest, GenerationResponse, ChatRequest, ChatResponse,
    StreamingResponse as StreamingResponseModel, HealthResponse, ModelInfoResponse,
    ErrorResponse, ReasoningLevel
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
model_instance: Optional[GPTOss20bMLX] = None
memory_manager: Optional[MemoryManager] = None
server_start_time: float = 0
active_conversations: int = 0


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage model lifecycle and production components during server startup/shutdown"""
    global model_instance, memory_manager, server_start_time

    logger.info("🚀 Starting tkr-embed | GPT-OSS-20B Text Generation Server (Production Mode)")
    server_start_time = time.time()

    try:
        # Initialize memory manager
        logger.info("Initializing memory manager...")
        memory_manager = MemoryManager()
        memory_manager.optimize_for_generation()

        # Initialize model with configuration
        logger.info("Loading GPT-OSS-20B text generation model...")
        model_instance = GPTOss20bMLX(
            model_path=getattr(config.model, 'model_path', 'microsoft/gpt-oss-20b'),
            quantization=getattr(config.model, 'quantization', 'auto'),
            device=getattr(config.model, 'device', 'auto'),
            cache_dir=getattr(config.model, 'cache_dir', './models')
        )

        # Load model if not in testing mode
        if config.environment.value != "testing":
            logger.info("Loading model (this may take a few minutes)...")
            await model_instance.load_model()
            logger.info("✅ Model loaded successfully")
        else:
            logger.info("Skipping model loading in testing mode")

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
    title="tkr-embed | GPT-OSS-20B Text Generation Server",
    description="Production-ready text generation service optimized for Apple Silicon",
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
            status="healthy" if model_instance and model_instance.is_ready() else "initializing",
            model_loaded=model_instance is not None and model_instance.is_ready(),
            framework="MLX",
            device="Apple Silicon GPU",
            memory_usage_gb=memory_stats.get("process_memory_gb", 0.0),
            uptime_seconds=uptime,
            generation_ready=model_instance is not None and model_instance.is_ready(),
            active_conversations=active_conversations
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
            quantization=model_info_dict.get("quantization", "auto"),
            context_length=model_info_dict.get("context_length", 8192),
            vocab_size=model_info_dict.get("vocab_size", 50257),
            supported_tasks=["text_generation", "chat_completion"],
            load_time=model_info_dict.get("load_time"),
            memory_usage_gb=model_info_dict.get("memory_usage_gb"),
            reasoning_capabilities=[ReasoningLevel.LOW, ReasoningLevel.MEDIUM, ReasoningLevel.HIGH]
        )
    except Exception as e:
        logger.error(f"Model info request failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate", response_model=GenerationResponse)
async def generate_text(
    request: GenerationRequest,
    http_request: Request,
    api_key: APIKey = Depends(authenticate if config.security.require_api_key else optional_auth)
):
    """Generate text completion for a single prompt (requires authentication)"""
    start_time = time.time()

    # Apply rate limiting
    if config.rate_limit.enabled:
        identifier = f"{api_key.name if api_key else 'anonymous'}:{http_request.client.host if http_request.client else 'unknown'}"
        await apply_rate_limit(http_request, identifier)

    try:
        # Safe model operation with error handling
        with SafeModelOperation("text_generation", model_instance):
            if not model_instance or not model_instance.is_ready():
                raise_model_not_ready()

            logger.info(f"Processing text generation request for {api_key.name if api_key else 'anonymous'}")

            # Create generation config from request
            import sys
            sys.path.append('/Volumes/tkr-riffic/@tkr-projects/tkr-embed/.context-kit/_specs')
            from integration_contracts import GenerationConfig
            generation_config = GenerationConfig(
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                top_k=request.top_k,
                reasoning_level=request.reasoning_level,
                repetition_penalty=request.repetition_penalty
            )

            # Generate text
            generated_text = await model_instance.generate(request.text, generation_config)

            # Calculate token counts (mock implementation)
            prompt_tokens = len(request.text.split())
            completion_tokens = len(generated_text.split())

        processing_time = time.time() - start_time

        # Create response
        response_data = GenerationResponse(
            generated_text=generated_text,
            tokens_used=prompt_tokens + completion_tokens,
            reasoning_level=request.reasoning_level,
            processing_time=processing_time,
            finish_reason="stop",
            model="gpt-oss-20b",
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens
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
        logger.error(f"Text generation failed: {e}")
        # The error will be handled by the comprehensive error handler
        raise


@app.post("/chat", response_model=ChatResponse)
async def chat_completion(
    request: ChatRequest,
    http_request: Request,
    api_key: APIKey = Depends(authenticate if config.security.require_api_key else optional_auth)
):
    """Generate chat response for conversation (requires authentication)"""
    global active_conversations
    start_time = time.time()
    conversation_id = f"conv_{int(time.time() * 1000)}"

    # Apply rate limiting
    if config.rate_limit.enabled:
        identifier = f"{api_key.name if api_key else 'anonymous'}:{http_request.client.host if http_request.client else 'unknown'}"
        await apply_rate_limit(http_request, identifier)

    try:
        # Safe model operation with error handling
        with SafeModelOperation("chat_completion", model_instance):
            if not model_instance or not model_instance.is_ready():
                raise_model_not_ready()

            logger.info(f"Processing chat request with {len(request.messages)} messages for {api_key.name if api_key else 'anonymous'}")

            active_conversations += 1

            # Convert messages to the format expected by model
            messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]

            # Create generation config from request
            import sys
            sys.path.append('/Volumes/tkr-riffic/@tkr-projects/tkr-embed/.context-kit/_specs')
            from integration_contracts import GenerationConfig
            generation_config = GenerationConfig(
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                reasoning_level=request.reasoning_level
            )

            # Generate chat response
            response_text = await model_instance.chat(messages, generation_config)

            # Calculate token counts (mock implementation)
            total_input_tokens = sum(len(msg.content.split()) for msg in request.messages)
            completion_tokens = len(response_text.split())

        processing_time = time.time() - start_time
        active_conversations = max(0, active_conversations - 1)

        # Create response
        response_data = ChatResponse(
            response=response_text,
            conversation_id=conversation_id,
            tokens_used=total_input_tokens + completion_tokens,
            reasoning_level=request.reasoning_level,
            processing_time=processing_time,
            finish_reason="stop",
            model="gpt-oss-20b",
            prompt_tokens=total_input_tokens,
            completion_tokens=completion_tokens
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
        logger.error(f"Chat completion failed: {e}")
        active_conversations = max(0, active_conversations - 1)
        # The error will be handled by the comprehensive error handler
        raise


@app.post("/stream")
async def stream_generation(
    request: GenerationRequest,
    http_request: Request,
    api_key: APIKey = Depends(authenticate if config.security.require_api_key else optional_auth)
):
    """Stream text generation with Server-Sent Events (requires authentication)"""

    # Apply rate limiting
    if config.rate_limit.enabled:
        identifier = f"{api_key.name if api_key else 'anonymous'}:{http_request.client.host if http_request.client else 'unknown'}"
        await apply_rate_limit(http_request, identifier)

    async def generate_stream():
        try:
            # Safe model operation with error handling
            with SafeModelOperation("stream_generation", model_instance):
                if not model_instance or not model_instance.is_ready():
                    yield f"data: {json.dumps({'error': 'Model not ready'})}\n\n"
                    return

                logger.info(f"Processing streaming generation request for {api_key.name if api_key else 'anonymous'}")

                # Create generation config from request
                import sys
                sys.path.append('/Volumes/tkr-riffic/@tkr-projects/tkr-embed/.context-kit/_specs')
                from integration_contracts import GenerationConfig
                generation_config = GenerationConfig(
                    max_tokens=request.max_tokens,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    top_k=request.top_k,
                    reasoning_level=request.reasoning_level,
                    repetition_penalty=request.repetition_penalty,
                    streaming=True
                )

                tokens_generated = 0

                # Stream tokens
                async for token in model_instance.generate_stream(request.text, generation_config):
                    tokens_generated += 1

                    chunk_data = StreamingResponseModel(
                        chunk={
                            "delta": token,
                            "finish_reason": None,
                            "tokens_generated": tokens_generated
                        },
                        conversation_id=None,
                        reasoning_level=request.reasoning_level,
                        model="gpt-oss-20b"
                    )

                    yield f"data: {chunk_data.model_dump_json()}\n\n"

                # Send final chunk
                final_chunk = StreamingResponseModel(
                    chunk={
                        "delta": "",
                        "finish_reason": "stop",
                        "tokens_generated": tokens_generated
                    },
                    conversation_id=None,
                    reasoning_level=request.reasoning_level,
                    model="gpt-oss-20b"
                )

                yield f"data: {final_chunk.model_dump_json()}\n\n"
                yield "data: [DONE]\n\n"

        except Exception as e:
            logger.error(f"Streaming generation failed: {e}")
            error_chunk = {"error": str(e)}
            yield f"data: {json.dumps(error_chunk)}\n\n"

    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "*",
        }
    )


if __name__ == "__main__":
    # Run server
    uvicorn.run(
        "tkr_embed.api.server:app",
        host="0.0.0.0",
        port=8008,
        log_level="info",
        reload=False
    )