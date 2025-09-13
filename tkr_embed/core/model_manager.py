"""
MLX Model Manager for OpenSearch-AI/Ops-MM-embedding-v1-7B
Handles model loading, quantization, and memory optimization for Apple Silicon
OpenSearch-AI Ops-MM-embedding-v1-7B multimodal embedding model with MLX backend
"""

import logging
from pathlib import Path
from typing import Optional, Dict, Any, List, Union
import mlx.core as mx
import mlx.nn as nn
import psutil
import time
import numpy as np
from transformers import AutoTokenizer, AutoModel, AutoImageProcessor
import torch

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
        self.image_processor = None
        
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
        logger.info(f"Loading OpenSearch-AI Ops-MM-embedding model from {self.model_path}")
        start_time = time.time()
        
        try:
            # Load tokenizer and processor
            logger.info("Loading tokenizer and processor...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path, 
                trust_remote_code=True
            )
            self.image_processor = AutoImageProcessor.from_pretrained(
                self.model_path,
                trust_remote_code=True
            )
            
            # Load embedding model with appropriate precision
            logger.info("Loading Ops-MM-embedding model...")
            torch_dtype = torch.float16 if self.quantization != "none" else torch.float32

            # Load the embedding model
            self.model = AutoModel.from_pretrained(
                self.model_path,
                torch_dtype=torch_dtype,
                trust_remote_code=True,
                device_map="auto"
            )
            
            logger.info("Model loaded successfully")
            
            # Convert to evaluation mode
            self.model.eval()
            
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
        """Apply 4-bit quantization for 16GB systems using BitsAndBytes"""
        try:
            from transformers import BitsAndBytesConfig
            import torch
            
            logger.info("Converting to 4-bit precision with BitsAndBytes...")
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            
            # Reload model with quantization
            self.model = AutoModel.from_pretrained(
                self.model_path,
                quantization_config=quantization_config,
                trust_remote_code=True,
                device_map="auto"
            )
            logger.info("4-bit quantization complete")
        except ImportError:
            logger.warning("BitsAndBytes not available, using float16 instead")
            self.model = self.model.half()
    
    def _quantize_8bit(self) -> None:
        """Apply 8-bit quantization for 32GB systems using BitsAndBytes"""
        try:
            from transformers import BitsAndBytesConfig
            
            logger.info("Converting to 8-bit precision with BitsAndBytes...")
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True
            )
            
            # Reload model with quantization
            self.model = AutoModel.from_pretrained(
                self.model_path,
                quantization_config=quantization_config,
                trust_remote_code=True,
                device_map="auto"
            )
            logger.info("8-bit quantization complete")
        except ImportError:
            logger.warning("BitsAndBytes not available, using float16 instead")
            self.model = self.model.half()
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in GB"""
        process = psutil.Process()
        memory_bytes = process.memory_info().rss
        return memory_bytes / (1024**3)
    
    def encode_text(self, texts: List[str]) -> np.ndarray:
        """
        Encode text inputs to embeddings using Ops-MM-embedding model

        Args:
            texts: List of text strings to encode

        Returns:
            NumPy array of embeddings
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        logger.debug(f"Encoding {len(texts)} text inputs")

        embeddings_list = []

        with torch.no_grad():
            for text in texts:
                # Tokenize the text
                inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)

                # Get embeddings from the model (no hidden_states needed for embedding models)
                outputs = self.model(**inputs)

                # For embedding models, use the pooler_output or last_hidden_state
                if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                    embedding = outputs.pooler_output
                else:
                    # Use mean pooling over sequence length
                    embedding = outputs.last_hidden_state.mean(dim=1)  # Shape: (1, hidden_size)

                embeddings_list.append(embedding.cpu().numpy())

        # Stack all embeddings
        embeddings = np.vstack(embeddings_list)
        return embeddings
    
    def encode_image(self, images: Union[str, List[str]]) -> np.ndarray:
        """
        Encode image inputs to embeddings using Ops-MM-embedding vision encoder

        Args:
            images: Image file paths (string or list of strings)

        Returns:
            NumPy array of image embeddings
        """
        from PIL import Image

        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Ensure images is a list
        if isinstance(images, str):
            images = [images]

        logger.debug(f"Encoding {len(images)} image inputs")

        embeddings_list = []

        with torch.no_grad():
            for image_path in images:
                # Load and process the image
                image = Image.open(image_path).convert('RGB')

                # Process the image with the image processor
                inputs = self.image_processor(image, return_tensors="pt")

                # Get embeddings from the model
                outputs = self.model(**inputs)

                # For embedding models, use pooler_output or mean pooling
                if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                    embedding = outputs.pooler_output
                else:
                    # Use mean pooling over spatial dimensions
                    embedding = outputs.last_hidden_state.mean(dim=1)

                embeddings_list.append(embedding.cpu().numpy())

        # Stack all embeddings
        embeddings = np.vstack(embeddings_list)
        return embeddings
    
    def encode_multimodal(self, text: Optional[str] = None, image_path: Optional[str] = None) -> np.ndarray:
        """
        Encode multimodal inputs (text + image) to unified embedding using Ops-MM-embedding model

        Args:
            text: Optional text input
            image_path: Optional image file path

        Returns:
            NumPy array of multimodal embedding
        """
        from PIL import Image

        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        if text is None and image_path is None:
            raise ValueError("At least one of text or image_path must be provided")

        logger.debug("Encoding multimodal input")

        with torch.no_grad():
            inputs = {}

            # Process text input
            if text is not None:
                text_inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
                inputs.update(text_inputs)

            # Process image input
            if image_path is not None:
                image = Image.open(image_path).convert('RGB')
                image_inputs = self.image_processor(image, return_tensors="pt")
                inputs.update(image_inputs)

            # Get multimodal embeddings from the model
            outputs = self.model(**inputs)

            # For embedding models, use pooler_output or mean pooling
            if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                embedding = outputs.pooler_output
            else:
                # Use mean pooling over sequence length for multimodal embedding
                embedding = outputs.last_hidden_state.mean(dim=1)

            return embedding.cpu().numpy()
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information and performance metrics"""
        return {
            "model_path": self.model_path,
            "quantization": self.quantization,
            "device": self.device,
            "load_time": self.load_time,
            "memory_usage_gb": self.memory_usage,
            "model_loaded": self.model is not None,
            "embedding_dim": 768,  # Standard embedding dimension for Ops-MM-embedding
            "architecture": "AutoModel",
            "supported_modalities": ["text", "image", "multimodal"]
        }
    
    def is_ready(self) -> bool:
        """Check if model is loaded and ready for inference"""
        return (self.model is not None and
                self.tokenizer is not None and
                self.image_processor is not None)
    
    def __del__(self):
        """Cleanup resources"""
        if self.model is not None:
            logger.info("Cleaning up MLX model resources")
            del self.model
            mx.metal.clear_cache()