"""
LRU Cache implementation for embedding results
Significantly improves performance for frequently requested embeddings
"""

import hashlib
import json
import time
import logging
from typing import Dict, Any, Optional, Tuple, List
from collections import OrderedDict
import numpy as np
import threading

logger = logging.getLogger(__name__)


class EmbeddingCache:
    """
    Thread-safe LRU cache for embedding results
    """
    
    def __init__(self, max_size: int = 1000, ttl: float = 3600.0):
        """
        Initialize LRU cache
        
        Args:
            max_size: Maximum number of cached entries
            ttl: Time-to-live for cache entries in seconds (default 1 hour)
        """
        self.max_size = max_size
        self.ttl = ttl
        self.cache: OrderedDict[str, Tuple[np.ndarray, float]] = OrderedDict()
        self.lock = threading.RLock()
        
        # Statistics
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        
        logger.info(f"EmbeddingCache initialized with max_size={max_size}, ttl={ttl}s")
    
    def _generate_key(self, input_type: str, content: Any) -> str:
        """
        Generate cache key from input
        
        Args:
            input_type: Type of input (text, image, multimodal)
            content: Input content
            
        Returns:
            Hash-based cache key
        """
        if input_type == "text":
            # For text, use the text itself
            key_content = content if isinstance(content, str) else json.dumps(content)
        elif input_type == "image":
            # For images, use file path or content hash
            key_content = content if isinstance(content, str) else str(content)
        elif input_type == "multimodal":
            # For multimodal, combine text and image info
            text = content.get("text", "")
            image = content.get("image", "")
            key_content = f"{text}|{image}"
        else:
            key_content = str(content)
        
        # Generate SHA256 hash
        hash_obj = hashlib.sha256(f"{input_type}:{key_content}".encode())
        return hash_obj.hexdigest()
    
    def get(self, input_type: str, content: Any) -> Optional[np.ndarray]:
        """
        Get embedding from cache
        
        Args:
            input_type: Type of input
            content: Input content
            
        Returns:
            Cached embedding array or None if not found/expired
        """
        key = self._generate_key(input_type, content)
        
        with self.lock:
            if key not in self.cache:
                self.misses += 1
                return None
            
            # Check if entry is expired
            embedding, timestamp = self.cache[key]
            if time.time() - timestamp > self.ttl:
                # Remove expired entry
                del self.cache[key]
                self.misses += 1
                logger.debug(f"Cache entry expired for key {key[:8]}...")
                return None
            
            # Move to end (most recently used)
            self.cache.move_to_end(key)
            self.hits += 1
            
            logger.debug(f"Cache hit for key {key[:8]}...")
            return embedding.copy()  # Return copy to prevent modification
    
    def put(self, input_type: str, content: Any, embedding: np.ndarray) -> None:
        """
        Store embedding in cache
        
        Args:
            input_type: Type of input
            content: Input content
            embedding: Embedding array to cache
        """
        key = self._generate_key(input_type, content)
        
        with self.lock:
            # Remove key if it exists (to reinsert at end)
            if key in self.cache:
                del self.cache[key]
            
            # Add new entry
            self.cache[key] = (embedding.copy(), time.time())
            
            # Evict oldest if over capacity
            while len(self.cache) > self.max_size:
                evicted_key = next(iter(self.cache))
                del self.cache[evicted_key]
                self.evictions += 1
                logger.debug(f"Evicted cache entry {evicted_key[:8]}...")
            
            logger.debug(f"Cached embedding for key {key[:8]}...")
    
    def get_batch(self, input_type: str, contents: List[Any]) -> Tuple[List[Optional[np.ndarray]], List[int]]:
        """
        Get multiple embeddings from cache
        
        Args:
            input_type: Type of input
            contents: List of input contents
            
        Returns:
            Tuple of (cached embeddings list, indices of cache misses)
        """
        results = []
        miss_indices = []
        
        for i, content in enumerate(contents):
            cached = self.get(input_type, content)
            results.append(cached)
            if cached is None:
                miss_indices.append(i)
        
        return results, miss_indices
    
    def put_batch(self, input_type: str, contents: List[Any], embeddings: np.ndarray) -> None:
        """
        Store multiple embeddings in cache
        
        Args:
            input_type: Type of input
            contents: List of input contents
            embeddings: Array of embeddings with shape (batch_size, embedding_dim)
        """
        for i, content in enumerate(contents):
            self.put(input_type, content, embeddings[i])
    
    def clear(self) -> None:
        """Clear all cache entries"""
        with self.lock:
            self.cache.clear()
            logger.info("Cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics
        
        Returns:
            Dictionary of cache statistics
        """
        with self.lock:
            total_requests = self.hits + self.misses
            hit_rate = self.hits / max(1, total_requests)
            
            return {
                "size": len(self.cache),
                "max_size": self.max_size,
                "hits": self.hits,
                "misses": self.misses,
                "evictions": self.evictions,
                "hit_rate": hit_rate,
                "total_requests": total_requests,
                "ttl": self.ttl
            }
    
    def __repr__(self) -> str:
        stats = self.get_stats()
        return (f"EmbeddingCache(size={stats['size']}/{stats['max_size']}, "
                f"hit_rate={stats['hit_rate']:.2%}, "
                f"hits={stats['hits']}, misses={stats['misses']})")


class CachedEmbeddingProcessor:
    """
    Wrapper that adds caching to any embedding model
    """
    
    def __init__(self, model_manager, cache: Optional[EmbeddingCache] = None):
        """
        Initialize cached processor
        
        Args:
            model_manager: The underlying model manager (e.g., OpsMMEmbeddingMLX)
            cache: Optional pre-configured cache, otherwise creates default
        """
        self.model = model_manager
        self.cache = cache or EmbeddingCache(max_size=1000, ttl=3600)
        
        logger.info("CachedEmbeddingProcessor initialized")
    
    def encode_text(self, texts: List[str], use_cache: bool = True) -> np.ndarray:
        """
        Encode texts with caching
        
        Args:
            texts: List of text strings
            use_cache: Whether to use cache (default True)
            
        Returns:
            Array of embeddings
        """
        if not use_cache:
            return self.model.encode_text(texts)
        
        # Check cache for all texts
        cached_results, miss_indices = self.cache.get_batch("text", texts)
        
        # If all cached, return immediately
        if not miss_indices:
            logger.info(f"All {len(texts)} texts found in cache")
            return np.vstack(cached_results)
        
        # Process cache misses
        texts_to_process = [texts[i] for i in miss_indices]
        logger.info(f"Processing {len(texts_to_process)} texts (cache misses)")
        
        new_embeddings = self.model.encode_text(texts_to_process)
        
        # Store new embeddings in cache
        self.cache.put_batch("text", texts_to_process, new_embeddings)
        
        # Combine cached and new results
        result = []
        new_idx = 0
        for i, cached in enumerate(cached_results):
            if cached is not None:
                result.append(cached)
            else:
                result.append(new_embeddings[new_idx])
                new_idx += 1
        
        return np.vstack(result)
    
    def encode_image(self, images: List[str], use_cache: bool = True) -> np.ndarray:
        """
        Encode images with caching
        
        Args:
            images: List of image paths
            use_cache: Whether to use cache
            
        Returns:
            Array of embeddings
        """
        if not use_cache:
            return self.model.encode_image(images)
        
        # Similar caching logic for images
        cached_results, miss_indices = self.cache.get_batch("image", images)
        
        if not miss_indices:
            logger.info(f"All {len(images)} images found in cache")
            return np.vstack(cached_results)
        
        images_to_process = [images[i] for i in miss_indices]
        logger.info(f"Processing {len(images_to_process)} images (cache misses)")
        
        new_embeddings = self.model.encode_image(images_to_process)
        self.cache.put_batch("image", images_to_process, new_embeddings)
        
        # Combine results
        result = []
        new_idx = 0
        for i, cached in enumerate(cached_results):
            if cached is not None:
                result.append(cached)
            else:
                result.append(new_embeddings[new_idx])
                new_idx += 1
        
        return np.vstack(result)
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return self.cache.get_stats()
    
    def clear_cache(self) -> None:
        """Clear the cache"""
        self.cache.clear()