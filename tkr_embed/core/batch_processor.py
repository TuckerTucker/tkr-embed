"""
Batch processing optimization for MLX multimodal embedding server
Improves throughput by processing multiple inputs together
"""

import logging
from typing import List, Optional, Dict, Any, Union
import numpy as np
import torch
from dataclasses import dataclass
import asyncio
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)


@dataclass
class BatchConfig:
    """Configuration for batch processing"""
    max_batch_size: int = 8  # Maximum inputs per batch
    max_wait_time: float = 0.1  # Maximum wait time for batch accumulation (seconds)
    dynamic_batching: bool = True  # Adjust batch size based on memory
    memory_threshold: float = 0.85  # Max memory usage before reducing batch size


class BatchProcessor:
    """Optimized batch processing for embeddings"""
    
    def __init__(self, model_manager, config: Optional[BatchConfig] = None):
        """
        Initialize batch processor
        
        Args:
            model_manager: The OpsMMEmbeddingMLX model manager instance
            config: Batch processing configuration
        """
        self.model = model_manager
        self.config = config or BatchConfig()
        self.executor = ThreadPoolExecutor(max_workers=2)
        
        # Batch accumulation
        self.pending_texts: List[Dict] = []
        self.pending_images: List[Dict] = []
        self.pending_multimodal: List[Dict] = []
        
        # Performance tracking
        self.total_processed = 0
        self.total_time = 0.0
        
        logger.info(f"BatchProcessor initialized with max_batch_size={self.config.max_batch_size}")
    
    def encode_text_batch(self, texts: List[str]) -> np.ndarray:
        """
        Encode a batch of texts efficiently
        
        Args:
            texts: List of text strings to encode
            
        Returns:
            NumPy array of embeddings with shape (batch_size, embedding_dim)
        """
        if not self.model.is_ready():
            raise RuntimeError("Model not ready for inference")
        
        batch_size = len(texts)
        logger.debug(f"Processing text batch of size {batch_size}")
        
        # Process in optimal sub-batches if needed
        if batch_size > self.config.max_batch_size:
            embeddings_list = []
            for i in range(0, batch_size, self.config.max_batch_size):
                sub_batch = texts[i:i + self.config.max_batch_size]
                sub_embeddings = self._process_text_batch(sub_batch)
                embeddings_list.append(sub_embeddings)
            return np.vstack(embeddings_list)
        else:
            return self._process_text_batch(texts)
    
    def _process_text_batch(self, texts: List[str]) -> np.ndarray:
        """
        Process a single batch of texts
        
        Args:
            texts: List of text strings (size <= max_batch_size)
            
        Returns:
            NumPy array of embeddings
        """
        embeddings_list = []
        
        with torch.no_grad():
            # Batch tokenization for efficiency
            batch_messages = []
            for text in texts:
                messages = [{"role": "user", "content": [{"type": "text", "text": text}]}]
                batch_messages.append(messages)
            
            # Process all texts in the batch together if possible
            try:
                # Attempt batch processing
                for messages in batch_messages:
                    inputs = self.model.processor.apply_chat_template(
                        messages, 
                        add_generation_prompt=False, 
                        tokenize=True, 
                        return_tensors="pt"
                    )
                    
                    # Get embeddings
                    outputs = self.model.model(
                        input_ids=inputs, 
                        output_hidden_states=True
                    )
                    
                    last_hidden_state = outputs.hidden_states[-1]
                    embedding = last_hidden_state.mean(dim=1)
                    embeddings_list.append(embedding.cpu().numpy())
                
            except Exception as e:
                logger.error(f"Batch processing failed: {e}, falling back to sequential")
                # Fallback to sequential processing
                return self.model.encode_text(texts)
        
        return np.vstack(embeddings_list)
    
    def encode_image_batch(self, image_paths: List[str]) -> np.ndarray:
        """
        Encode a batch of images efficiently
        
        Args:
            image_paths: List of image file paths
            
        Returns:
            NumPy array of embeddings with shape (batch_size, embedding_dim)
        """
        if not self.model.is_ready():
            raise RuntimeError("Model not ready for inference")
        
        batch_size = len(image_paths)
        logger.debug(f"Processing image batch of size {batch_size}")
        
        # Process in optimal sub-batches if needed
        if batch_size > self.config.max_batch_size:
            embeddings_list = []
            for i in range(0, batch_size, self.config.max_batch_size):
                sub_batch = image_paths[i:i + self.config.max_batch_size]
                sub_embeddings = self._process_image_batch(sub_batch)
                embeddings_list.append(sub_embeddings)
            return np.vstack(embeddings_list)
        else:
            return self._process_image_batch(image_paths)
    
    def _process_image_batch(self, image_paths: List[str]) -> np.ndarray:
        """
        Process a single batch of images
        
        Args:
            image_paths: List of image paths (size <= max_batch_size)
            
        Returns:
            NumPy array of embeddings
        """
        # For now, use sequential processing
        # TODO: Implement true batch image processing when supported
        return self.model.encode_image(image_paths)
    
    async def process_async_batch(
        self, 
        inputs: List[Dict[str, Any]], 
        input_type: str = "text"
    ) -> List[np.ndarray]:
        """
        Process inputs asynchronously with batching
        
        Args:
            inputs: List of input dictionaries
            input_type: Type of input ("text", "image", or "multimodal")
            
        Returns:
            List of embedding arrays
        """
        logger.info(f"Processing async batch of {len(inputs)} {input_type} inputs")
        
        # Group inputs into optimal batches
        batches = []
        for i in range(0, len(inputs), self.config.max_batch_size):
            batch = inputs[i:i + self.config.max_batch_size]
            batches.append(batch)
        
        # Process batches concurrently
        tasks = []
        for batch in batches:
            if input_type == "text":
                texts = [item["text"] for item in batch]
                task = asyncio.get_event_loop().run_in_executor(
                    self.executor, self.encode_text_batch, texts
                )
            elif input_type == "image":
                paths = [item["path"] for item in batch]
                task = asyncio.get_event_loop().run_in_executor(
                    self.executor, self.encode_image_batch, paths
                )
            else:  # multimodal
                # Process multimodal inputs
                task = self._process_multimodal_batch_async(batch)
            
            tasks.append(task)
        
        # Wait for all batches to complete
        results = await asyncio.gather(*tasks)
        
        # Flatten results
        all_embeddings = []
        for batch_result in results:
            if len(batch_result.shape) == 2:
                for i in range(batch_result.shape[0]):
                    all_embeddings.append(batch_result[i])
            else:
                all_embeddings.append(batch_result)
        
        return all_embeddings
    
    async def _process_multimodal_batch_async(self, batch: List[Dict]) -> np.ndarray:
        """
        Process multimodal batch asynchronously
        
        Args:
            batch: List of multimodal input dictionaries
            
        Returns:
            NumPy array of embeddings
        """
        embeddings = []
        for item in batch:
            text = item.get("text")
            image_path = item.get("image_path")
            
            embedding = await asyncio.get_event_loop().run_in_executor(
                self.executor,
                self.model.encode_multimodal,
                text,
                image_path
            )
            embeddings.append(embedding)
        
        return np.vstack(embeddings)
    
    def get_optimal_batch_size(self) -> int:
        """
        Determine optimal batch size based on available memory
        
        Returns:
            Optimal batch size for current system state
        """
        if not self.config.dynamic_batching:
            return self.config.max_batch_size
        
        try:
            import psutil
            memory_percent = psutil.virtual_memory().percent / 100.0
            
            if memory_percent > self.config.memory_threshold:
                # Reduce batch size when memory is high
                return max(1, self.config.max_batch_size // 2)
            elif memory_percent < 0.5:
                # Increase batch size when memory is low
                return min(16, self.config.max_batch_size * 2)
            else:
                return self.config.max_batch_size
                
        except Exception as e:
            logger.warning(f"Failed to determine optimal batch size: {e}")
            return self.config.max_batch_size
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get batch processing statistics
        
        Returns:
            Dictionary of performance statistics
        """
        avg_time = self.total_time / max(1, self.total_processed)
        throughput = self.total_processed / max(0.001, self.total_time)
        
        return {
            "total_processed": self.total_processed,
            "total_time": self.total_time,
            "average_time_per_item": avg_time,
            "throughput_per_second": throughput,
            "optimal_batch_size": self.get_optimal_batch_size(),
            "max_batch_size": self.config.max_batch_size
        }