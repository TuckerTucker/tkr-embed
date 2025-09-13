"""
Pydantic models for API requests and responses
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from enum import Enum


class ReasoningLevel(str, Enum):
    """Reasoning level for text generation"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class GenerationRequest(BaseModel):
    """Request model for text generation"""
    text: str = Field(..., description="Input text prompt", min_length=1, max_length=10000)
    max_tokens: int = Field(512, description="Maximum tokens to generate", ge=1, le=4096)
    temperature: float = Field(0.7, description="Sampling temperature", ge=0.0, le=2.0)
    reasoning_level: ReasoningLevel = Field(ReasoningLevel.MEDIUM, description="Reasoning complexity level")
    top_p: Optional[float] = Field(0.9, description="Nucleus sampling probability", ge=0.0, le=1.0)
    top_k: Optional[int] = Field(50, description="Top-k sampling parameter", ge=1, le=100)
    repetition_penalty: Optional[float] = Field(1.1, description="Repetition penalty", ge=1.0, le=2.0)
    stream: bool = Field(False, description="Enable streaming response")

    class Config:
        schema_extra = {
            "example": {
                "text": "Explain the concept of machine learning in simple terms.",
                "max_tokens": 256,
                "temperature": 0.7,
                "reasoning_level": "medium",
                "stream": False
            }
        }


class GenerationResponse(BaseModel):
    """Response model for text generation"""
    generated_text: str = Field(..., description="Generated text output")
    tokens_used: int = Field(..., description="Number of tokens used in generation")
    reasoning_level: ReasoningLevel = Field(..., description="Reasoning level used")
    processing_time: float = Field(..., description="Processing time in seconds")
    finish_reason: str = Field(..., description="Reason generation finished (length, stop, error)")
    model: str = Field("gpt-oss-20b", description="Model used for generation")
    prompt_tokens: int = Field(..., description="Number of tokens in input prompt")
    completion_tokens: int = Field(..., description="Number of tokens in generated completion")

    class Config:
        schema_extra = {
            "example": {
                "generated_text": "Machine learning is a subset of artificial intelligence...",
                "tokens_used": 156,
                "reasoning_level": "medium",
                "processing_time": 2.34,
                "finish_reason": "length",
                "model": "gpt-oss-20b",
                "prompt_tokens": 12,
                "completion_tokens": 144
            }
        }


class ChatMessage(BaseModel):
    """Individual chat message"""
    role: str = Field(..., description="Message role (user, assistant, system)")
    content: str = Field(..., description="Message content", min_length=1)

    class Config:
        schema_extra = {
            "example": {
                "role": "user",
                "content": "What is the capital of France?"
            }
        }


class ChatRequest(BaseModel):
    """Request model for chat completion"""
    messages: List[ChatMessage] = Field(..., description="Conversation messages", min_items=1, max_items=50)
    system_prompt: Optional[str] = Field(None, description="Optional system prompt", max_length=2000)
    reasoning_level: ReasoningLevel = Field(ReasoningLevel.MEDIUM, description="Reasoning complexity level")
    max_tokens: int = Field(512, description="Maximum tokens to generate", ge=1, le=4096)
    temperature: float = Field(0.7, description="Sampling temperature", ge=0.0, le=2.0)
    top_p: Optional[float] = Field(0.9, description="Nucleus sampling probability", ge=0.0, le=1.0)
    stream: bool = Field(False, description="Enable streaming response")

    class Config:
        schema_extra = {
            "example": {
                "messages": [
                    {"role": "user", "content": "What is machine learning?"}
                ],
                "system_prompt": "You are a helpful AI assistant.",
                "reasoning_level": "medium",
                "max_tokens": 256,
                "temperature": 0.7,
                "stream": False
            }
        }


class ChatResponse(BaseModel):
    """Response model for chat completion"""
    response: str = Field(..., description="Generated chat response")
    conversation_id: Optional[str] = Field(None, description="Unique conversation identifier")
    tokens_used: int = Field(..., description="Total tokens used (prompt + completion)")
    reasoning_level: ReasoningLevel = Field(..., description="Reasoning level used")
    processing_time: float = Field(..., description="Processing time in seconds")
    finish_reason: str = Field(..., description="Reason generation finished")
    model: str = Field("gpt-oss-20b", description="Model used for generation")
    prompt_tokens: int = Field(..., description="Number of tokens in input")
    completion_tokens: int = Field(..., description="Number of tokens in response")

    class Config:
        schema_extra = {
            "example": {
                "response": "Machine learning is a method of data analysis...",
                "conversation_id": "conv_12345",
                "tokens_used": 178,
                "reasoning_level": "medium",
                "processing_time": 1.85,
                "finish_reason": "stop",
                "model": "gpt-oss-20b",
                "prompt_tokens": 34,
                "completion_tokens": 144
            }
        }


class StreamingChunk(BaseModel):
    """Individual chunk in streaming response"""
    delta: str = Field(..., description="Text delta for this chunk")
    finish_reason: Optional[str] = Field(None, description="Finish reason if complete")
    tokens_generated: int = Field(..., description="Total tokens generated so far")

    class Config:
        schema_extra = {
            "example": {
                "delta": "Machine learning is",
                "finish_reason": None,
                "tokens_generated": 3
            }
        }


class StreamingResponse(BaseModel):
    """Response model for streaming generation"""
    chunk: StreamingChunk = Field(..., description="Current streaming chunk")
    conversation_id: Optional[str] = Field(None, description="Conversation identifier")
    reasoning_level: ReasoningLevel = Field(..., description="Reasoning level used")
    model: str = Field("gpt-oss-20b", description="Model used for generation")

    class Config:
        schema_extra = {
            "example": {
                "chunk": {
                    "delta": "Machine learning is",
                    "finish_reason": None,
                    "tokens_generated": 3
                },
                "conversation_id": "conv_12345",
                "reasoning_level": "medium",
                "model": "gpt-oss-20b"
            }
        }


class HealthResponse(BaseModel):
    """Response model for health check"""
    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    framework: str = Field("MLX", description="ML framework")
    device: str = Field("Apple Silicon GPU", description="Compute device")
    memory_usage_gb: float = Field(..., description="Current memory usage in GB")
    uptime_seconds: float = Field(..., description="Service uptime in seconds")
    generation_ready: bool = Field(..., description="Whether generation is ready")
    active_conversations: int = Field(0, description="Number of active conversations")

    class Config:
        schema_extra = {
            "example": {
                "status": "healthy",
                "model_loaded": True,
                "framework": "MLX",
                "device": "Apple Silicon GPU",
                "memory_usage_gb": 12.5,
                "uptime_seconds": 3600,
                "generation_ready": True,
                "active_conversations": 3
            }
        }


class ModelInfoResponse(BaseModel):
    """Response model for model information"""
    model_path: str = Field(..., description="Hugging Face model path")
    framework: str = Field("MLX", description="ML framework")
    mlx_version: str = Field(..., description="MLX framework version")
    quantization: Optional[str] = Field(..., description="Quantization method used")
    context_length: int = Field(8192, description="Maximum context length in tokens")
    vocab_size: int = Field(..., description="Vocabulary size")
    supported_tasks: List[str] = Field(..., description="Supported generation tasks")
    load_time: Optional[float] = Field(None, description="Model load time in seconds")
    memory_usage_gb: Optional[float] = Field(None, description="Model memory usage in GB")
    reasoning_capabilities: List[ReasoningLevel] = Field(..., description="Available reasoning levels")

    class Config:
        schema_extra = {
            "example": {
                "model_path": "microsoft/gpt-oss-20b",
                "framework": "MLX",
                "mlx_version": "0.29.0",
                "quantization": "Q8_0",
                "context_length": 8192,
                "vocab_size": 50257,
                "supported_tasks": ["text_generation", "chat_completion"],
                "load_time": 15.2,
                "memory_usage_gb": 18.5,
                "reasoning_capabilities": ["low", "medium", "high"]
            }
        }


class ErrorResponse(BaseModel):
    """Response model for errors"""
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")
    request_id: Optional[str] = Field(None, description="Request identifier for debugging")
    error_code: Optional[str] = Field(None, description="Specific error code")

    class Config:
        schema_extra = {
            "example": {
                "error": "Invalid request parameters",
                "detail": "Temperature must be between 0.0 and 2.0",
                "request_id": "req_12345",
                "error_code": "VALIDATION_ERROR"
            }
        }