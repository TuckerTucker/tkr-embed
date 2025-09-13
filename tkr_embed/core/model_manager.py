"""
MLX Model Manager for GPT-OSS-20B
Handles model loading, quantization, and memory optimization for Apple Silicon
Supports text generation with reasoning levels and streaming capabilities
"""

import logging
from pathlib import Path
from typing import Optional, Dict, Any, List, Union, AsyncIterator
import mlx.core as mx
import mlx.nn as nn
import psutil
import time
import asyncio
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
import torch
from threading import Thread
try:
    from mlx_lm import load, generate
    from mlx_lm.sample_utils import make_sampler
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False
from ..config import get_config
try:
    import sys
    sys.path.append(str(Path(__file__).parent.parent.parent / ".context-kit" / "_specs"))
    from integration_contracts import (
        IGPTOss20bModel, GenerationConfig, ReasoningLevel,
        ModelNotReadyError, TokenLimitExceededError, GenerationError
    )
except ImportError:
    # Fallback if integration contracts not available
    from typing import Protocol
    from enum import Enum
    from dataclasses import dataclass

    class ReasoningLevel(str, Enum):
        LOW = "low"
        MEDIUM = "medium"
        HIGH = "high"

    @dataclass
    class GenerationConfig:
        max_tokens: int = 4096
        temperature: float = 0.7
        top_p: float = 0.9
        top_k: int = 50
        reasoning_level: ReasoningLevel = ReasoningLevel.MEDIUM
        streaming: bool = False
        repetition_penalty: float = 1.1

    class GenerationError(Exception):
        pass

    class ModelNotReadyError(GenerationError):
        pass

    class TokenLimitExceededError(GenerationError):
        pass

    class IGPTOss20bModel(Protocol):
        async def load_model(self) -> None: ...
        async def generate(self, prompt: str, config: GenerationConfig) -> str: ...
        async def generate_stream(self, prompt: str, config: GenerationConfig) -> AsyncIterator[str]: ...
        async def chat(self, messages: List[Dict[str, str]], config: GenerationConfig) -> str: ...
        def get_model_info(self) -> Dict[str, Any]: ...
        def is_ready(self) -> bool: ...
        def get_memory_usage(self) -> float: ...

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GPTOss20bMLX:
    """MLX implementation of GPT-OSS-20B text generation model"""
    
    def __init__(
        self,
        model_path: str = "openai/gpt-oss-20b",
        quantization: str = "auto",  # auto, q4, q8, mxfp4, or none
        device: str = "auto",
        cache_dir: str = "./models"
    ):
        """
        Initialize the MLX text generation model.

        Args:
            model_path: Hugging Face model identifier (default: openai/gpt-oss-20b)
            quantization: Quantization strategy (auto, q4, q8, mxfp4, none)
            device: Target device (auto, cpu, gpu)
            cache_dir: Local model cache directory
        """
        logger.info(f"Initializing GPTOss20bMLX with model: {model_path}")

        self.model_path = model_path
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

        # Model components
        self.model = None
        self.tokenizer = None

        # Generation state
        self.reasoning_prompts = {
            ReasoningLevel.LOW: "Answer directly and concisely.",
            ReasoningLevel.MEDIUM: "Think through this step by step and provide a detailed answer.",
            ReasoningLevel.HIGH: "Carefully analyze this problem from multiple angles, show your reasoning process, and provide a comprehensive answer."
        }

        # Get configuration
        self.config = get_config()
        
        # Performance metrics
        self.load_time = None
        self.memory_usage = None
        
        # Auto-detect quantization if needed
        if quantization == "auto":
            self.quantization = self._detect_optimal_quantization()
        else:
            self.quantization = quantization

        # Auto-detect device if needed
        if device == "auto":
            self.device = self._detect_optimal_device()
        else:
            self.device = device

        logger.info(f"Using quantization: {self.quantization}")
        logger.info(f"Using device: {self.device}")

    def _is_mlx_model(self) -> bool:
        """Check if this is an MLX quantized model"""
        mlx_indicators = [
            "MLX" in self.model_path.upper(),
            "mlx" in self.model_path.lower(),
            any(keyword in self.model_path.lower() for keyword in ["4bit", "8bit", "quantized"])
        ]
        return any(mlx_indicators)

    def _detect_optimal_quantization(self) -> str:
        """Auto-detect optimal quantization based on system memory for 21B model"""
        memory_gb = psutil.virtual_memory().total // (1024**3)

        # GPT-OSS-20B requires significant memory
        if memory_gb <= 16:
            logger.info("16GB or less detected - using Q4 quantization for 21B model")
            return "q4"
        elif memory_gb <= 32:
            logger.info("32GB detected - using Q8 quantization for optimal performance")
            return "q8"
        elif memory_gb <= 64:
            logger.info("64GB detected - using MXFP4 for best quality/performance balance")
            return "mxfp4"
        else:
            logger.info("64GB+ detected - using full precision")
            return "none"

    def _detect_optimal_device(self) -> str:
        """Auto-detect optimal device (prioritize Metal GPU on Apple Silicon)"""
        if torch.backends.mps.is_available():
            logger.info("Apple Silicon Metal GPU detected - using MPS")
            return "mps"
        elif torch.cuda.is_available():
            logger.info("CUDA GPU detected")
            return "cuda"
        else:
            logger.info("No GPU acceleration detected - using CPU")
            return "cpu"
    
    async def load_model(self) -> None:
        """Load and quantize the GPT-OSS-20B text generation model"""
        logger.info(f"Loading GPT-OSS-20B model from {self.model_path}")
        start_time = time.time()

        try:
            # Check if this is an MLX model
            if self._is_mlx_model() and MLX_AVAILABLE:
                logger.info("Detected MLX model, using MLX loading...")
                await self._load_mlx_model()
            else:
                logger.info("Using transformers loading...")
                await self._load_transformers_model()

            # Record performance metrics
            self.load_time = time.time() - start_time
            self.memory_usage = self._get_memory_usage()

            logger.info(f"Model ready! Load time: {self.load_time:.1f}s, Memory: {self.memory_usage:.1f}GB")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    async def _load_mlx_model(self) -> None:
        """Load MLX quantized model"""
        logger.info("Loading MLX model and tokenizer...")

        # Use mlx_lm to load both model and tokenizer
        self.model, self.tokenizer = load(self.model_path)

        # Add pad token if missing
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        logger.info("MLX model loaded successfully")

    async def _load_transformers_model(self) -> None:
        """Load model using transformers library"""
        # Load tokenizer
        logger.info("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            cache_dir=self.cache_dir
        )

        # Add pad token if missing
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model with appropriate precision and device
        logger.info(f"Loading GPT-OSS-20B model on {self.device}...")
        torch_dtype = self._get_torch_dtype()
        device_map = self._get_device_map()

        # Load model with quantization if specified
        if self.quantization in ["q4", "q8"]:
            logger.info(f"Loading model with {self.quantization.upper()} quantization...")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch_dtype,
                trust_remote_code=True,
                cache_dir=self.cache_dir,
                quantization_config=self._get_quantization_config(),
                device_map=device_map,
                low_cpu_mem_usage=True
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch_dtype,
                trust_remote_code=True,
                cache_dir=self.cache_dir,
                device_map=device_map,
                low_cpu_mem_usage=True
            )

        logger.info("Model loaded successfully")

        # Convert to evaluation mode
        self.model.eval()

    def _get_torch_dtype(self) -> torch.dtype:
        """Get appropriate torch dtype based on quantization"""
        if self.quantization == "none":
            return torch.float32
        else:
            return torch.float16

    def _get_device_map(self):
        """Get device mapping for model loading"""
        if self.device == "cpu":
            return "cpu"
        elif self.device in ["mps", "cuda"]:
            # Use explicit device mapping instead of "auto" to avoid gpt-oss-20b KeyError
            return {"": 0}
        else:
            return "auto"

    def _get_quantization_config(self):
        """Get quantization configuration"""
        try:
            from transformers import BitsAndBytesConfig

            if self.quantization == "q4":
                return BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
            elif self.quantization == "q8":
                return BitsAndBytesConfig(
                    load_in_8bit=True
                )
        except ImportError:
            logger.warning("BitsAndBytes not available, falling back to standard loading")
            return None

        return None
    
    def _format_prompt_with_reasoning(self, prompt: str, reasoning_level: ReasoningLevel) -> str:
        """Format prompt with reasoning level instructions"""
        reasoning_instruction = self.reasoning_prompts.get(reasoning_level, self.reasoning_prompts[ReasoningLevel.MEDIUM])
        return f"{reasoning_instruction}\n\n{prompt}"

    def _format_chat_messages(self, messages: List[Dict[str, str]]) -> str:
        """Format chat messages into a single prompt"""
        formatted_parts = []

        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")

            if role == "system":
                formatted_parts.append(f"System: {content}")
            elif role == "user":
                formatted_parts.append(f"Human: {content}")
            elif role == "assistant":
                formatted_parts.append(f"Assistant: {content}")

        # Add final assistant prompt
        formatted_parts.append("Assistant:")
        return "\n\n".join(formatted_parts)
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in GB"""
        process = psutil.Process()
        memory_bytes = process.memory_info().rss
        return memory_bytes / (1024**3)
    
    async def generate(
        self,
        prompt: str,
        config: GenerationConfig
    ) -> str:
        """
        Generate text completion for a single prompt

        Args:
            prompt: Input text prompt
            config: Generation configuration

        Returns:
            Generated text
        """
        if not self.is_ready():
            raise ModelNotReadyError("Model not loaded. Call load_model() first.")

        logger.debug(f"Generating text for prompt: {prompt[:100]}...")

        # Format prompt with reasoning level
        formatted_prompt = self._format_prompt_with_reasoning(prompt, config.reasoning_level)

        try:
            # Use MLX generation if it's an MLX model
            if self._is_mlx_model() and MLX_AVAILABLE:
                return await self._generate_mlx(formatted_prompt, config)
            else:
                return await self._generate_transformers(formatted_prompt, config)

        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise GenerationError(f"Generation failed: {e}")

    async def _generate_mlx(self, prompt: str, config: GenerationConfig) -> str:
        """Generate using MLX model"""
        logger.debug("Using MLX generation")

        # Create sampler with the generation parameters
        sampler = make_sampler(
            temp=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k
        )

        # Use mlx_lm generate function with sampler
        response = generate(
            self.model,
            self.tokenizer,
            prompt=prompt,
            max_tokens=config.max_tokens,
            sampler=sampler,
            verbose=False
        )

        return response.strip()

    async def _generate_transformers(self, formatted_prompt: str, config: GenerationConfig) -> str:
        """Generate using transformers model"""
        logger.debug("Using transformers generation")

        # Tokenize input
        inputs = self.tokenizer(
            formatted_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.model.max_sequence_length - config.max_tokens
        )

        # Move to appropriate device
        if self.device != "cpu":
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        # Check token limits
        input_length = inputs["input_ids"].shape[1]
        logger.debug(f"Input length: {input_length}, max_tokens: {config.max_tokens}")
        if input_length + config.max_tokens > self.config.model.max_sequence_length:
            raise TokenLimitExceededError(f"Total tokens ({input_length + config.max_tokens}) exceed max sequence length")

        with torch.no_grad():
            # Generate with specified parameters
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=config.max_tokens,
                temperature=config.temperature,
                top_p=config.top_p,
                top_k=config.top_k,
                repetition_penalty=config.repetition_penalty,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )

            logger.debug(f"Outputs shape: {outputs.shape}, input_length: {input_length}")

            # Decode only the generated portion
            if outputs.shape[1] <= input_length:
                logger.error(f"No new tokens generated. Output length: {outputs.shape[1]}, Input length: {input_length}")
                return "I'm sorry, I couldn't generate a response. Please try again."

            generated_tokens = outputs[0][input_length:]
            generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

            return generated_text.strip()
    
    async def generate_stream(
        self,
        prompt: str,
        config: GenerationConfig
    ) -> AsyncIterator[str]:
        """
        Stream text generation token by token

        Args:
            prompt: Input text prompt
            config: Generation configuration

        Yields:
            Generated text chunks
        """
        if not self.is_ready():
            raise ModelNotReadyError("Model not loaded. Call load_model() first.")

        if not config.streaming:
            # Non-streaming fallback
            result = await self.generate(prompt, config)
            yield result
            return

        logger.debug(f"Streaming generation for prompt: {prompt[:100]}...")

        # Format prompt with reasoning level
        formatted_prompt = self._format_prompt_with_reasoning(prompt, config.reasoning_level)

        # Tokenize input
        inputs = self.tokenizer(
            formatted_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.model.max_sequence_length - config.max_tokens
        )

        # Move to appropriate device
        if self.device != "cpu":
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        # Check token limits
        input_length = inputs["input_ids"].shape[1]
        if input_length + config.max_tokens > self.config.model.max_sequence_length:
            raise TokenLimitExceededError(f"Total tokens exceed max sequence length")

        try:
            # Create streamer
            streamer = TextIteratorStreamer(
                self.tokenizer,
                timeout=self.config.generation.generation_timeout_seconds,
                skip_prompt=True,
                skip_special_tokens=True
            )

            generation_kwargs = {
                **inputs,
                "max_new_tokens": config.max_tokens,
                "temperature": config.temperature,
                "top_p": config.top_p,
                "top_k": config.top_k,
                "repetition_penalty": config.repetition_penalty,
                "do_sample": True,
                "pad_token_id": self.tokenizer.eos_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
                "streamer": streamer
            }

            # Start generation in thread
            thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
            thread.start()

            # Stream tokens
            for new_text in streamer:
                if new_text:
                    yield new_text
                    # Small delay to prevent overwhelming the client
                    await asyncio.sleep(0.01)

            thread.join()

        except Exception as e:
            logger.error(f"Streaming generation failed: {e}")
            raise GenerationError(f"Streaming generation failed: {e}")
    
    async def chat(
        self,
        messages: List[Dict[str, str]],
        config: GenerationConfig
    ) -> str:
        """
        Generate chat response for conversation

        Args:
            messages: List of conversation messages with 'role' and 'content'
            config: Generation configuration

        Returns:
            Generated chat response
        """
        if not self.is_ready():
            raise ModelNotReadyError("Model not loaded. Call load_model() first.")

        logger.debug(f"Generating chat response for {len(messages)} messages")

        # Format chat messages into prompt
        formatted_prompt = self._format_chat_messages(messages)

        # Add reasoning level instruction
        reasoning_instruction = self.reasoning_prompts.get(config.reasoning_level, self.reasoning_prompts[ReasoningLevel.MEDIUM])
        full_prompt = f"{reasoning_instruction}\n\n{formatted_prompt}"

        # Generate response using the chat prompt
        response = await self.generate(full_prompt, config)

        return response
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information and performance metrics"""
        return {
            "model_path": self.model_path,
            "quantization": self.quantization,
            "device": self.device,
            "load_time": self.load_time,
            "memory_usage_gb": self.memory_usage,
            "model_loaded": self.model is not None,
            "parameters": "21B",  # GPT-OSS-20B parameters
            "architecture": "AutoModelForCausalLM",
            "supported_capabilities": ["text_generation", "chat", "streaming", "reasoning_levels"],
            "max_sequence_length": self.config.model.max_sequence_length,
            "reasoning_levels": list(self.reasoning_prompts.keys())
        }
    
    def is_ready(self) -> bool:
        """Check if model is loaded and ready for inference"""
        return (self.model is not None and
                self.tokenizer is not None)

    def get_memory_usage(self) -> float:
        """Get current memory usage in GB"""
        return self._get_memory_usage()
    
    def __del__(self):
        """Cleanup resources"""
        if self.model is not None:
            logger.info("Cleaning up GPT-OSS-20B model resources")
            del self.model
            if hasattr(mx, 'metal'):
                mx.metal.clear_cache()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()