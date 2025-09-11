"""
MLX Model Manager for OpenSearch-AI/Ops-MM-embedding-v1-7B
Handles model loading, quantization, and memory optimization for Apple Silicon
"""

import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load, generate
import psutil
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OpsMMEmbeddingMLX:
    """MLX implementation of Ops-MM-embedding-v1-7B multimodal embedding model"""
    
    def __init__(
        self,
        model_path: str = "OpenSearch-AI/Ops-MM-embedding-v1-7B",
        quantization: str = "auto",  # auto, q4, q8, or none
        device: str = "gpu",
        cache_dir: str = "./models"
    ):
        """
        Initialize the MLX multimodal embedding model.
        
        Args:
            model_path: Hugging Face model identifier
            quantization: Quantization strategy (auto, q4, q8, none)
            device: Target device (gpu for Metal)
            cache_dir: Local model cache directory
        """
        logger.info(f"Initializing OpsMMEmbeddingMLX with model: {model_path}")
        
        self.model_path = model_path
        self.device = device
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # Model components
        self.model = None
        self.tokenizer = None
        self.processor = None
        
        # Performance metrics
        self.load_time = None
        self.memory_usage = None
        
        # Auto-detect quantization if needed
        if quantization == "auto":
            self.quantization = self._detect_optimal_quantization()
        else:
            self.quantization = quantization
            
        logger.info(f"Using quantization: {self.quantization}")
        
    def _detect_optimal_quantization(self) -> str:
        """Auto-detect optimal quantization based on system memory"""
        memory_gb = psutil.virtual_memory().total // (1024**3)
        
        if memory_gb <= 16:
            logger.info("16GB or less detected - using Q4_0 quantization")
            return "q4"
        elif memory_gb <= 32:
            logger.info("32GB or less detected - using Q8_0 quantization") 
            return "q8"
        else:
            logger.info("64GB+ detected - using full precision")
            return "none"
    
    async def load_model(self) -> None:
        """Load and quantize the multimodal embedding model"""
        logger.info(f"Loading model from {self.model_path}")
        start_time = time.time()
        
        try:
            # Load base model and tokenizer
            logger.info("Loading base model and tokenizer...")
            self.model, self.tokenizer = load(
                self.model_path,
                tokenizer_config={"trust_remote_code": True}
            )
            
            logger.info("Model loaded successfully")
            
            # Apply quantization if specified
            if self.quantization == "q4":
                logger.info("Applying 4-bit quantization...")
                self._quantize_4bit()
            elif self.quantization == "q8":
                logger.info("Applying 8-bit quantization...")
                self._quantize_8bit()
            else:
                logger.info("Using full precision (no quantization)")
            
            # Record performance metrics
            self.load_time = time.time() - start_time
            self.memory_usage = self._get_memory_usage()
            
            logger.info(f"Model ready! Load time: {self.load_time:.1f}s, Memory: {self.memory_usage:.1f}GB")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def _quantize_4bit(self) -> None:
        """Apply 4-bit quantization for 16GB systems"""
        from mlx.nn import quantize
        
        logger.info("Quantizing model to 4-bit precision...")
        self.model = quantize(
            self.model, 
            bits=4, 
            group_size=64
        )
        logger.info("4-bit quantization complete")
    
    def _quantize_8bit(self) -> None:
        """Apply 8-bit quantization for 32GB systems"""
        from mlx.nn import quantize
        
        logger.info("Quantizing model to 8-bit precision...")
        self.model = quantize(
            self.model,
            bits=8,
            group_size=128
        )
        logger.info("8-bit quantization complete")
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in GB"""
        process = psutil.Process()
        memory_bytes = process.memory_info().rss
        return memory_bytes / (1024**3)
    
    def encode_text(self, texts: List[str]) -> mx.array:
        """
        Encode text inputs to embeddings
        
        Args:
            texts: List of text strings to encode
            
        Returns:
            MLX array of embeddings
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        logger.debug(f"Encoding {len(texts)} text inputs")
        
        # Tokenize inputs
        inputs = self.tokenizer(texts, return_tensors="mlx", padding=True, truncation=True)
        
        # Generate embeddings
        with mx.no_grad():
            outputs = self.model(**inputs)
            # Extract embeddings (typically from last_hidden_state)
            embeddings = outputs.last_hidden_state.mean(axis=1)  # Mean pooling
        
        return embeddings
    
    def encode_image(self, images: mx.array) -> mx.array:
        """
        Encode image inputs to embeddings
        
        Args:
            images: MLX array of preprocessed images
            
        Returns:
            MLX array of image embeddings
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        logger.debug(f"Encoding {images.shape[0]} image inputs")
        
        with mx.no_grad():
            # Process images through vision encoder
            image_embeddings = self.model.encode_image(images)
        
        return image_embeddings
    
    def encode_multimodal(self, text: Optional[str] = None, image: Optional[mx.array] = None) -> mx.array:
        """
        Encode multimodal inputs (text + image) to unified embedding
        
        Args:
            text: Optional text input
            image: Optional image input as MLX array
            
        Returns:
            MLX array of multimodal embedding
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        if text is None and image is None:
            raise ValueError("At least one of text or image must be provided")
        
        logger.debug("Encoding multimodal input")
        
        with mx.no_grad():
            # Encode text if provided
            text_embedding = None
            if text is not None:
                text_inputs = self.tokenizer([text], return_tensors="mlx")
                text_outputs = self.model(**text_inputs)
                text_embedding = text_outputs.last_hidden_state.mean(axis=1)
            
            # Encode image if provided
            image_embedding = None
            if image is not None:
                image_embedding = self.model.encode_image(mx.expand_dims(image, axis=0))
            
            # Combine embeddings if both provided
            if text_embedding is not None and image_embedding is not None:
                # Simple concatenation - the model may have more sophisticated fusion
                combined_embedding = mx.concatenate([text_embedding, image_embedding], axis=-1)
                return combined_embedding
            elif text_embedding is not None:
                return text_embedding
            else:
                return image_embedding
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information and performance metrics"""
        return {
            "model_path": self.model_path,
            "quantization": self.quantization,
            "device": self.device,
            "load_time": self.load_time,
            "memory_usage_gb": self.memory_usage,
            "model_loaded": self.model is not None,
            "embedding_dim": 1024,  # Ops-MM default dimension
            "supported_modalities": ["text", "image", "multimodal"]
        }
    
    def is_ready(self) -> bool:
        """Check if model is loaded and ready for inference"""
        return self.model is not None and self.tokenizer is not None
    
    def __del__(self):
        """Cleanup resources"""
        if self.model is not None:
            logger.info("Cleaning up MLX model resources")
            del self.model
            mx.metal.clear_cache()