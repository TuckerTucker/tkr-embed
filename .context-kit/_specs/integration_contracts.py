"""
Integration Contracts for GPT-OSS-20B Implementation
Synchronization Point 1: Foundation Complete

This file defines the shared interfaces between all implementation agents
to ensure seamless integration across parallel development efforts.
"""

from typing import Optional, List, Dict, Any, AsyncIterator
from enum import Enum
from dataclasses import dataclass
import numpy as np

# =====================================
# Phase 1 Integration Contracts
# =====================================

class ReasoningLevel(str, Enum):
    """Shared reasoning level enum across all components"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

class ModelQuantization(str, Enum):
    """Quantization strategies for model loading"""
    Q4 = "q4"
    Q8 = "q8"
    MXFP4 = "mxfp4"
    NONE = "none"
    AUTO = "auto"

# =====================================
# Model Manager Interface (Agent A → All)
# =====================================

@dataclass
class GenerationConfig:
    """Configuration for text generation"""
    max_tokens: int = 4096
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    reasoning_level: ReasoningLevel = ReasoningLevel.MEDIUM
    streaming: bool = False
    repetition_penalty: float = 1.1

class IGPTOss20bModel:
    """Interface contract for GPT-OSS-20B model manager"""

    async def load_model(self) -> None:
        """Load and initialize the model with appropriate quantization"""
        ...

    async def generate(
        self,
        prompt: str,
        config: GenerationConfig
    ) -> str:
        """Generate text completion for a single prompt"""
        ...

    async def generate_stream(
        self,
        prompt: str,
        config: GenerationConfig
    ) -> AsyncIterator[str]:
        """Stream text generation token by token"""
        ...

    async def chat(
        self,
        messages: List[Dict[str, str]],
        config: GenerationConfig
    ) -> str:
        """Generate chat response for conversation"""
        ...

    def get_model_info(self) -> Dict[str, Any]:
        """Return model metadata and capabilities"""
        ...

    def is_ready(self) -> bool:
        """Check if model is loaded and ready"""
        ...

    def get_memory_usage(self) -> float:
        """Get current memory usage in GB"""
        ...

# =====================================
# API Endpoint Interface (Agent B → Testing)
# =====================================

@dataclass
class GenerationRequest:
    """Unified request format for generation endpoints"""
    prompt: Optional[str] = None
    messages: Optional[List[Dict[str, str]]] = None
    max_tokens: int = 4096
    temperature: float = 0.7
    reasoning_level: str = "medium"
    streaming: bool = False

@dataclass
class GenerationResponse:
    """Unified response format for generation endpoints"""
    text: str
    tokens_used: int
    reasoning_level: str
    processing_time: float
    model: str = "gpt-oss-20b"

class IGenerationAPI:
    """Interface contract for generation API endpoints"""

    async def generate_text(
        self,
        request: GenerationRequest
    ) -> GenerationResponse:
        """Single text generation endpoint"""
        ...

    async def chat_completion(
        self,
        request: GenerationRequest
    ) -> GenerationResponse:
        """Chat conversation endpoint"""
        ...

    async def stream_generation(
        self,
        request: GenerationRequest
    ) -> AsyncIterator[str]:
        """Streaming generation endpoint"""
        ...

# =====================================
# Infrastructure Interface (Agent C → All)
# =====================================

class IGenerationCache:
    """Interface for generation caching system"""

    async def get(
        self,
        prompt: str,
        config: GenerationConfig
    ) -> Optional[str]:
        """Retrieve cached generation if available"""
        ...

    async def set(
        self,
        prompt: str,
        config: GenerationConfig,
        result: str
    ) -> None:
        """Cache generation result"""
        ...

class IMemoryManager:
    """Interface for memory management"""

    def allocate_for_model(self, size_gb: float) -> bool:
        """Allocate memory for model loading"""
        ...

    def get_available_memory(self) -> float:
        """Get available system memory in GB"""
        ...

    def optimize_for_generation(self) -> None:
        """Optimize memory for text generation workloads"""
        ...

# =====================================
# Shared Configuration (From Phase 1)
# =====================================

@dataclass
class SystemConfig:
    """Unified system configuration from Phase 1"""
    model_path: str = "openai/gpt-oss-20b"
    quantization: ModelQuantization = ModelQuantization.AUTO
    max_concurrent_generations: int = 10
    generation_timeout: int = 300
    cache_enabled: bool = True
    streaming_chunk_size: int = 256
    memory_allocation_percentage: float = 0.85

# =====================================
# Integration Status Tracking
# =====================================

class IntegrationStatus:
    """Track integration progress across agents"""

    PHASE_1_COMPLETE = True

    # Phase 2 Status (to be updated by agents)
    model_manager_ready = False
    api_endpoints_ready = False
    infrastructure_ready = False

    # Phase 3 Status
    performance_validated = False
    cleanup_complete = False

    @classmethod
    def update_status(cls, component: str, ready: bool):
        """Update component readiness status"""
        setattr(cls, f"{component}_ready", ready)

    @classmethod
    def is_phase_2_ready(cls) -> bool:
        """Check if all Phase 2 components are ready"""
        return all([
            cls.model_manager_ready,
            cls.api_endpoints_ready,
            cls.infrastructure_ready
        ])

# =====================================
# Testing Contracts
# =====================================

class IIntegrationTest:
    """Interface for integration testing at sync points"""

    async def test_model_loading(self) -> bool:
        """Test model loads successfully"""
        ...

    async def test_generation_e2e(self) -> bool:
        """Test end-to-end generation flow"""
        ...

    async def test_api_endpoints(self) -> bool:
        """Test all API endpoints"""
        ...

    async def test_performance_targets(self) -> bool:
        """Validate performance meets targets"""
        ...

# =====================================
# Error Contracts
# =====================================

class GenerationError(Exception):
    """Base exception for generation errors"""
    pass

class ModelNotReadyError(GenerationError):
    """Model is not loaded or initialized"""
    pass

class TokenLimitExceededError(GenerationError):
    """Generation exceeds token limit"""
    pass

class ReasoningLevelError(GenerationError):
    """Invalid reasoning level specified"""
    pass

# =====================================
# Performance Targets
# =====================================

class PerformanceTargets:
    """Shared performance goals for validation"""
    MIN_TOKENS_PER_SECOND = 150
    MAX_LATENCY_MS = 100
    MAX_MEMORY_USAGE_PERCENT = 50
    MIN_CONCURRENT_REQUESTS = 100

# =====================================
# File Ownership Matrix
# =====================================

FILE_OWNERSHIP = {
    "model_manager": [
        "tkr_embed/core/model_manager.py",
        "tkr_embed/core/gpt_oss_20b.py"
    ],
    "api_endpoints": [
        "tkr_embed/api/server.py",
        "tkr_embed/api/generation_endpoints.py"
    ],
    "infrastructure": [
        "tkr_embed/utils/",
        "tkr_embed/api/error_handlers.py"
    ],
    "testing": [
        "tests/",
        "benchmarks/"
    ],
    "documentation": [
        "docs/",
        "examples/"
    ]
}

# =====================================
# Phase 2 Entry Points
# =====================================

def get_model_interface() -> IGPTOss20bModel:
    """Factory for model manager interface"""
    from tkr_embed.core.model_manager import GPTOss20bMLX
    return GPTOss20bMLX

def get_api_interface() -> IGenerationAPI:
    """Factory for API interface"""
    from tkr_embed.api.server import GenerationAPI
    return GenerationAPI

def get_infrastructure_interface() -> tuple:
    """Factory for infrastructure interfaces"""
    from tkr_embed.utils.cache import GenerationCache
    from tkr_embed.utils.memory_manager import MemoryManager
    return GenerationCache, MemoryManager