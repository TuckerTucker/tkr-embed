"""
GPT-OSS-20B Model Specification for MLX Framework
================================================================

Model: openai/gpt-oss-20b
Total Parameters: 21B (3.6B active per token)
Architecture: Mixture of Experts (MoE) Transformer
Memory Optimized: MXFP4 quantization support
Apple Silicon: MLX framework compatible

Research Summary:
- 21B total parameters with 3.6B active parameters per token
- Uses alternating dense and locally banded sparse attention
- Grouped multi-query attention with group size of 8
- Post-trained with MXFP4 quantization for 16GB memory operation
- Supports three reasoning levels: low, medium, high
- Apache 2.0 license for commercial deployment
- Requires "harmony response format" for proper functioning
"""

from typing import Dict, Any, List, Optional, Union, Literal
from dataclasses import dataclass
from enum import Enum
import mlx.core as mx
import mlx.nn as nn
from pathlib import Path


class ReasoningLevel(Enum):
    """Reasoning levels for GPT-OSS-20B model inference."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class QuantizationType(Enum):
    """Supported quantization types for MLX."""
    NONE = "none"
    Q4 = "q4"  # 4-bit quantization
    Q8 = "q8"  # 8-bit quantization
    MXFP4 = "mxfp4"  # Mixed precision FP4 (model default)


@dataclass
class ModelMemoryRequirements:
    """Memory requirements for different configurations on Apple Silicon."""

    # Base model requirements (no quantization)
    base_memory_gb: float = 84.0  # 21B * 4 bytes (FP32)

    # Quantized memory requirements
    q4_memory_gb: float = 10.5  # ~8x reduction from base
    q8_memory_gb: float = 21.0  # ~4x reduction from base
    mxfp4_memory_gb: float = 16.0  # Model's native quantization

    # Active parameters memory (3.6B active)
    active_memory_gb: float = 14.4  # 3.6B * 4 bytes
    active_q4_memory_gb: float = 1.8  # 3.6B * 0.5 bytes
    active_q8_memory_gb: float = 3.6  # 3.6B * 1 byte

    # Overhead and working memory
    context_memory_gb: float = 2.0  # Context window memory
    kv_cache_memory_gb: float = 4.0  # Key-value cache
    system_overhead_gb: float = 2.0  # System overhead

    @property
    def total_memory_requirements(self) -> Dict[str, float]:
        """Calculate total memory requirements for each quantization type."""
        overhead = self.context_memory_gb + self.kv_cache_memory_gb + self.system_overhead_gb

        return {
            "none": self.base_memory_gb + overhead,
            "q4": self.q4_memory_gb + overhead,
            "q8": self.q8_memory_gb + overhead,
            "mxfp4": self.mxfp4_memory_gb + overhead,
        }

    def is_compatible_with_memory(self, available_memory_gb: float, quantization: str = "q4") -> bool:
        """Check if model can run with available memory."""
        required = self.total_memory_requirements[quantization]
        return available_memory_gb >= required


@dataclass
class ModelConfig:
    """Configuration for GPT-OSS-20B model."""

    # Model architecture
    vocab_size: int = 50304  # Typical for GPT models
    hidden_size: int = 6144  # Estimated for 21B params
    num_layers: int = 44  # Estimated for 21B params
    num_heads: int = 48  # Estimated
    num_kv_heads: int = 6  # Grouped multi-query attention (group size 8)
    intermediate_size: int = 24576  # 4x hidden_size typical

    # MoE configuration
    num_experts: int = 8  # Typical MoE configuration
    num_experts_per_token: int = 2  # Active experts per token
    router_aux_loss_coef: float = 0.001

    # Attention configuration
    max_position_embeddings: int = 8192
    attention_dropout: float = 0.0
    hidden_dropout: float = 0.0

    # Quantization settings
    quantization: QuantizationType = QuantizationType.Q4
    use_flash_attention: bool = True

    # Apple Silicon optimizations
    use_metal_performance_shaders: bool = True
    unified_memory_optimization: bool = True


@dataclass
class GenerationConfig:
    """Configuration for text generation with reasoning levels."""

    # Generation parameters
    max_new_tokens: int = 2048
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.0

    # Reasoning configuration
    reasoning_level: ReasoningLevel = ReasoningLevel.MEDIUM
    show_chain_of_thought: bool = True
    use_harmony_format: bool = True  # Required for proper functioning

    # Performance settings
    batch_size: int = 1
    use_cache: bool = True
    pad_token_id: Optional[int] = None
    eos_token_id: Optional[int] = 50256


class GPTOSSModelInterface:
    """
    Interface specification for GPT-OSS-20B model in MLX framework.

    This interface defines the methods and configuration needed to load
    and run the GPT-OSS-20B model on Apple Silicon with MLX optimization.
    """

    def __init__(
        self,
        model_path: Union[str, Path],
        config: Optional[ModelConfig] = None,
        device: Optional[str] = None
    ):
        """
        Initialize GPT-OSS-20B model interface.

        Args:
            model_path: Path to model files or HuggingFace model ID
            config: Model configuration (uses defaults if None)
            device: Device to run on (auto-detected for Apple Silicon)
        """
        self.model_path = str(model_path)
        self.config = config or ModelConfig()
        self.device = device or "mps"  # Metal Performance Shaders
        self.model = None
        self.tokenizer = None
        self._memory_requirements = ModelMemoryRequirements()

    def load_model(self) -> bool:
        """
        Load the GPT-OSS-20B model with MLX optimization.

        Returns:
            bool: True if model loaded successfully

        Raises:
            RuntimeError: If insufficient memory or incompatible hardware
            ValueError: If model path is invalid
        """
        # Check memory requirements
        import psutil
        available_memory_gb = psutil.virtual_memory().total / (1024**3)

        if not self._memory_requirements.is_compatible_with_memory(
            available_memory_gb, self.config.quantization.value
        ):
            raise RuntimeError(
                f"Insufficient memory for {self.config.quantization.value} quantization. "
                f"Required: {self._memory_requirements.total_memory_requirements[self.config.quantization.value]:.1f}GB, "
                f"Available: {available_memory_gb:.1f}GB"
            )

        # Implementation would load model here
        # This is the interface specification
        return True

    def generate(
        self,
        prompt: str,
        generation_config: Optional[GenerationConfig] = None
    ) -> Dict[str, Any]:
        """
        Generate text with reasoning capabilities.

        Args:
            prompt: Input prompt text
            generation_config: Generation configuration

        Returns:
            Dict containing generated text and metadata
        """
        config = generation_config or GenerationConfig()

        # Format prompt with reasoning level
        formatted_prompt = self._format_prompt_with_reasoning(prompt, config)

        # Implementation would generate text here
        return {
            "generated_text": "Generated response would appear here",
            "reasoning_tokens": [] if not config.show_chain_of_thought else ["reasoning steps"],
            "reasoning_level": config.reasoning_level.value,
            "generation_stats": {
                "total_tokens": 0,
                "reasoning_tokens": 0,
                "generation_time_ms": 0,
                "tokens_per_second": 0
            }
        }

    def generate_embedding(
        self,
        text: str,
        layer: int = -1
    ) -> mx.array:
        """
        Extract embeddings from the model.

        Args:
            text: Input text
            layer: Layer to extract embeddings from (-1 for last layer)

        Returns:
            MLX array containing embeddings
        """
        # Implementation would extract embeddings here
        # Return placeholder embedding
        return mx.zeros((1, self.config.hidden_size))

    def _format_prompt_with_reasoning(
        self,
        prompt: str,
        config: GenerationConfig
    ) -> str:
        """Format prompt with reasoning level instruction."""
        reasoning_instruction = f"Reasoning: {config.reasoning_level.value}"

        if config.use_harmony_format:
            # GPT-OSS requires specific harmony response format
            formatted_prompt = f"""<|im_start|>system
{reasoning_instruction}
You are a helpful assistant. Please provide a detailed response with step-by-step reasoning.
<|im_end|>
<|im_start|>user
{prompt}
<|im_end|>
<|im_start|>assistant"""
        else:
            formatted_prompt = f"{reasoning_instruction}\n\n{prompt}"

        return formatted_prompt

    def get_memory_info(self) -> Dict[str, float]:
        """Get current memory usage information."""
        return self._memory_requirements.total_memory_requirements

    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information."""
        return {
            "model_name": "gpt-oss-20b",
            "total_parameters": "21B",
            "active_parameters": "3.6B",
            "architecture": "Mixture of Experts Transformer",
            "quantization": self.config.quantization.value,
            "memory_requirements": self.get_memory_info(),
            "reasoning_levels": [level.value for level in ReasoningLevel],
            "max_context_length": self.config.max_position_embeddings,
            "apple_silicon_optimized": True,
            "license": "Apache 2.0"
        }


class GPTOSSModelManager:
    """
    Model manager for handling multiple GPT-OSS model instances.
    Provides caching, memory management, and performance optimization.
    """

    def __init__(self, cache_size: int = 1):
        """Initialize model manager with LRU cache."""
        self._model_cache = {}
        self._cache_size = cache_size
        self._memory_monitor = ModelMemoryRequirements()

    def get_model(
        self,
        model_path: str,
        config: Optional[ModelConfig] = None
    ) -> GPTOSSModelInterface:
        """
        Get model instance with caching.

        Args:
            model_path: Path to model
            config: Model configuration

        Returns:
            GPTOSSModelInterface instance
        """
        cache_key = f"{model_path}_{hash(str(config))}"

        if cache_key not in self._model_cache:
            if len(self._model_cache) >= self._cache_size:
                # Remove oldest entry
                oldest_key = next(iter(self._model_cache))
                del self._model_cache[oldest_key]

            model = GPTOSSModelInterface(model_path, config)
            model.load_model()
            self._model_cache[cache_key] = model

        return self._model_cache[cache_key]

    def check_compatibility(self) -> Dict[str, bool]:
        """Check system compatibility for running GPT-OSS-20B."""
        import platform
        import psutil

        system_info = platform.uname()
        memory_gb = psutil.virtual_memory().total / (1024**3)

        return {
            "apple_silicon": system_info.machine in ["arm64", "aarch64"],
            "sufficient_memory_q4": memory_gb >= 18.5,  # Q4 + overhead
            "sufficient_memory_q8": memory_gb >= 29.0,  # Q8 + overhead
            "sufficient_memory_mxfp4": memory_gb >= 24.0,  # MXFP4 + overhead
            "mlx_available": self._check_mlx_availability(),
            "metal_available": self._check_metal_available()
        }

    def _check_mlx_availability(self) -> bool:
        """Check if MLX framework is available."""
        try:
            import mlx
            return True
        except ImportError:
            return False

    def _check_metal_available(self) -> bool:
        """Check if Metal Performance Shaders are available."""
        try:
            import mlx.core as mx
            # Test Metal availability
            test_array = mx.ones((2, 2))
            return test_array.device.type == mx.Device.gpu
        except:
            return False


# Example usage and testing functions
def create_recommended_config(available_memory_gb: float) -> ModelConfig:
    """Create recommended configuration based on available memory."""
    config = ModelConfig()

    if available_memory_gb >= 90:
        config.quantization = QuantizationType.NONE
    elif available_memory_gb >= 29:
        config.quantization = QuantizationType.Q8
    elif available_memory_gb >= 24:
        config.quantization = QuantizationType.MXFP4
    else:
        config.quantization = QuantizationType.Q4

    return config


def test_model_interface():
    """Test function for model interface."""
    # This would be used for testing the implementation
    manager = GPTOSSModelManager()
    compatibility = manager.check_compatibility()

    print("GPT-OSS-20B Compatibility Check:")
    for key, value in compatibility.items():
        status = "✅" if value else "❌"
        print(f"{status} {key}: {value}")

    if compatibility["apple_silicon"] and compatibility["mlx_available"]:
        print("\n✅ System is compatible with GPT-OSS-20B on MLX")

        # Recommend configuration
        import psutil
        memory_gb = psutil.virtual_memory().total / (1024**3)
        recommended_config = create_recommended_config(memory_gb)

        print(f"💾 Available memory: {memory_gb:.1f}GB")
        print(f"🔧 Recommended quantization: {recommended_config.quantization.value}")

        # Show memory requirements
        memory_reqs = ModelMemoryRequirements()
        total_reqs = memory_reqs.total_memory_requirements
        print(f"📊 Memory requirements by quantization:")
        for quant_type, memory_req in total_reqs.items():
            compatible = "✅" if memory_gb >= memory_req else "❌"
            print(f"  {compatible} {quant_type}: {memory_req:.1f}GB")
    else:
        print("\n❌ System is not compatible with GPT-OSS-20B on MLX")


if __name__ == "__main__":
    test_model_interface()