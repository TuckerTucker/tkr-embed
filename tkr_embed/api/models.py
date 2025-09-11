"""
Pydantic models for API requests and responses
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class TextEmbeddingRequest(BaseModel):
    """Request model for text embedding generation"""
    texts: List[str] = Field(..., description="List of texts to embed", min_items=1, max_items=100)
    normalize: bool = Field(True, description="Normalize embeddings to unit length")
    

class MultimodalEmbeddingRequest(BaseModel):
    """Request model for multimodal embedding generation"""
    text: Optional[str] = Field(None, description="Optional text input")
    normalize: bool = Field(True, description="Normalize embeddings to unit length")


class EmbeddingResponse(BaseModel):
    """Response model for embedding generation"""
    embeddings: List[List[float]] = Field(..., description="Generated embeddings")
    shape: List[int] = Field(..., description="Shape of embeddings array [batch_size, embedding_dim]")
    model: str = Field("Ops-MM-embedding-v1-7B", description="Model used for generation")
    processing_time: float = Field(..., description="Processing time in seconds")


class HealthResponse(BaseModel):
    """Response model for health check"""
    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    framework: str = Field("MLX", description="ML framework")
    device: str = Field("Apple Silicon GPU", description="Compute device")
    memory_usage_gb: float = Field(..., description="Current memory usage in GB")
    uptime_seconds: float = Field(..., description="Service uptime in seconds")


class ModelInfoResponse(BaseModel):
    """Response model for model information"""
    model_path: str = Field(..., description="Hugging Face model path")
    framework: str = Field("MLX", description="ML framework")
    mlx_version: str = Field(..., description="MLX framework version")
    quantization: Optional[str] = Field(..., description="Quantization method used")
    embedding_dim: int = Field(1024, description="Embedding dimensionality")
    supported_modalities: List[str] = Field(..., description="Supported input modalities")
    load_time: Optional[float] = Field(None, description="Model load time in seconds")
    memory_usage_gb: Optional[float] = Field(None, description="Model memory usage in GB")


class SimilarityRequest(BaseModel):
    """Request model for similarity computation"""
    query_embeddings: List[List[float]] = Field(..., description="Query embeddings")
    candidate_embeddings: List[List[float]] = Field(..., description="Candidate embeddings")
    metric: str = Field("cosine", description="Similarity metric (cosine, euclidean)")


class SimilarityResponse(BaseModel):
    """Response model for similarity computation"""
    similarities: List[List[float]] = Field(..., description="Similarity matrix")
    shape: List[int] = Field(..., description="Shape of similarity matrix")
    metric: str = Field(..., description="Similarity metric used")
    processing_time: float = Field(..., description="Processing time in seconds")


class ErrorResponse(BaseModel):
    """Response model for errors"""
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")
    request_id: Optional[str] = Field(None, description="Request identifier for debugging")