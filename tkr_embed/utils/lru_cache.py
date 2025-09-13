"""
LRU Cache implementation for text generation results
Significantly improves performance for frequently requested generations
Updated for GPT-OSS-20B text generation workloads
"""

import hashlib
import json
import time
import logging
from typing import Dict, Any, Optional, Tuple, List
from collections import OrderedDict
import threading
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class GenerationCacheEntry:
    """Cache entry for generation results"""
    text: str
    tokens_used: int
    reasoning_level: str
    timestamp: float
    generation_time: float


class GenerationCache:
    """
    Thread-safe LRU cache for text generation results
    Optimized for variable-length text outputs with reasoning levels
    """

    def __init__(self, max_size: int = 200, ttl: float = 1800.0):
        """
        Initialize generation cache

        Args:
            max_size: Maximum number of cached entries (reduced for larger responses)
            ttl: Time-to-live for cache entries in seconds (default 30 minutes)
        """
        self.max_size = max_size
        self.ttl = ttl
        self.cache: OrderedDict[str, GenerationCacheEntry] = OrderedDict()
        self.lock = threading.RLock()

        # Statistics
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.total_tokens_cached = 0
        self.total_tokens_served = 0

        logger.info(f"GenerationCache initialized with max_size={max_size}, ttl={ttl}s")
    
    def _generate_key(self, prompt: str, config: Dict[str, Any]) -> str:
        """
        Generate cache key from prompt and generation parameters

        Args:
            prompt: Input prompt text
            config: Generation configuration including reasoning level

        Returns:
            Hash-based cache key
        """
        # Include generation parameters in cache key
        key_components = {
            "prompt": prompt,
            "max_tokens": config.get("max_tokens", 4096),
            "temperature": config.get("temperature", 0.7),
            "top_p": config.get("top_p", 0.9),
            "top_k": config.get("top_k", 50),
            "reasoning_level": config.get("reasoning_level", "medium"),
            "repetition_penalty": config.get("repetition_penalty", 1.1)
        }

        # Create deterministic string representation
        key_string = json.dumps(key_components, sort_keys=True)

        # Generate SHA256 hash
        hash_obj = hashlib.sha256(key_string.encode())
        return hash_obj.hexdigest()

    def _generate_chat_key(self, messages: List[Dict[str, str]], config: Dict[str, Any]) -> str:
        """
        Generate cache key for chat conversations

        Args:
            messages: List of conversation messages
            config: Generation configuration

        Returns:
            Hash-based cache key
        """
        # Include conversation context and generation parameters
        key_components = {
            "messages": messages,
            "max_tokens": config.get("max_tokens", 4096),
            "temperature": config.get("temperature", 0.7),
            "top_p": config.get("top_p", 0.9),
            "top_k": config.get("top_k", 50),
            "reasoning_level": config.get("reasoning_level", "medium"),
            "repetition_penalty": config.get("repetition_penalty", 1.1)
        }

        # Create deterministic string representation
        key_string = json.dumps(key_components, sort_keys=True)

        # Generate SHA256 hash
        hash_obj = hashlib.sha256(key_string.encode())
        return hash_obj.hexdigest()
    
    def get(self, prompt: str, config: Dict[str, Any]) -> Optional[str]:
        """
        Get generation result from cache

        Args:
            prompt: Input prompt text
            config: Generation configuration

        Returns:
            Cached generation text or None if not found/expired
        """
        key = self._generate_key(prompt, config)

        with self.lock:
            if key not in self.cache:
                self.misses += 1
                return None

            # Check if entry is expired
            entry = self.cache[key]
            if time.time() - entry.timestamp > self.ttl:
                # Remove expired entry
                del self.cache[key]
                self.misses += 1
                logger.debug(f"Generation cache entry expired for key {key[:8]}...")
                return None

            # Move to end (most recently used)
            self.cache.move_to_end(key)
            self.hits += 1
            self.total_tokens_served += entry.tokens_used

            logger.debug(f"Generation cache hit for key {key[:8]}... (saved {entry.tokens_used} tokens)")
            return entry.text

    def get_chat(self, messages: List[Dict[str, str]], config: Dict[str, Any]) -> Optional[str]:
        """
        Get chat generation result from cache

        Args:
            messages: Conversation messages
            config: Generation configuration

        Returns:
            Cached generation text or None if not found/expired
        """
        key = self._generate_chat_key(messages, config)

        with self.lock:
            if key not in self.cache:
                self.misses += 1
                return None

            # Check if entry is expired
            entry = self.cache[key]
            if time.time() - entry.timestamp > self.ttl:
                # Remove expired entry
                del self.cache[key]
                self.misses += 1
                logger.debug(f"Chat cache entry expired for key {key[:8]}...")
                return None

            # Move to end (most recently used)
            self.cache.move_to_end(key)
            self.hits += 1
            self.total_tokens_served += entry.tokens_used

            logger.debug(f"Chat cache hit for key {key[:8]}... (saved {entry.tokens_used} tokens)")
            return entry.text
    
    def put(self, prompt: str, config: Dict[str, Any], result: str, tokens_used: int, generation_time: float) -> None:
        """
        Store generation result in cache

        Args:
            prompt: Input prompt text
            config: Generation configuration
            result: Generated text result
            tokens_used: Number of tokens used in generation
            generation_time: Time taken for generation
        """
        key = self._generate_key(prompt, config)

        with self.lock:
            # Remove key if it exists (to reinsert at end)
            if key in self.cache:
                old_entry = self.cache[key]
                self.total_tokens_cached -= old_entry.tokens_used
                del self.cache[key]

            # Create cache entry
            entry = GenerationCacheEntry(
                text=result,
                tokens_used=tokens_used,
                reasoning_level=config.get("reasoning_level", "medium"),
                timestamp=time.time(),
                generation_time=generation_time
            )

            # Add new entry
            self.cache[key] = entry
            self.total_tokens_cached += tokens_used

            # Evict oldest if over capacity
            while len(self.cache) > self.max_size:
                evicted_key = next(iter(self.cache))
                evicted_entry = self.cache[evicted_key]
                self.total_tokens_cached -= evicted_entry.tokens_used
                del self.cache[evicted_key]
                self.evictions += 1
                logger.debug(f"Evicted generation cache entry {evicted_key[:8]}... ({evicted_entry.tokens_used} tokens)")

            logger.debug(f"Cached generation for key {key[:8]}... ({tokens_used} tokens)")

    def put_chat(self, messages: List[Dict[str, str]], config: Dict[str, Any], result: str, tokens_used: int, generation_time: float) -> None:
        """
        Store chat generation result in cache

        Args:
            messages: Conversation messages
            config: Generation configuration
            result: Generated text result
            tokens_used: Number of tokens used in generation
            generation_time: Time taken for generation
        """
        key = self._generate_chat_key(messages, config)

        with self.lock:
            # Remove key if it exists (to reinsert at end)
            if key in self.cache:
                old_entry = self.cache[key]
                self.total_tokens_cached -= old_entry.tokens_used
                del self.cache[key]

            # Create cache entry
            entry = GenerationCacheEntry(
                text=result,
                tokens_used=tokens_used,
                reasoning_level=config.get("reasoning_level", "medium"),
                timestamp=time.time(),
                generation_time=generation_time
            )

            # Add new entry
            self.cache[key] = entry
            self.total_tokens_cached += tokens_used

            # Evict oldest if over capacity
            while len(self.cache) > self.max_size:
                evicted_key = next(iter(self.cache))
                evicted_entry = self.cache[evicted_key]
                self.total_tokens_cached -= evicted_entry.tokens_used
                del self.cache[evicted_key]
                self.evictions += 1
                logger.debug(f"Evicted chat cache entry {evicted_key[:8]}... ({evicted_entry.tokens_used} tokens)")

            logger.debug(f"Cached chat generation for key {key[:8]}... ({tokens_used} tokens)")
    
    def clear(self) -> None:
        """Clear all cache entries"""
        with self.lock:
            self.total_tokens_cached = 0
            self.total_tokens_served = 0
            self.cache.clear()
            logger.info("Generation cache cleared")

    def get_stats(self) -> Dict[str, Any]:
        """
        Get generation cache statistics

        Returns:
            Dictionary of cache statistics including token metrics
        """
        with self.lock:
            total_requests = self.hits + self.misses
            hit_rate = self.hits / max(1, total_requests)

            # Calculate average tokens per entry
            avg_tokens_cached = self.total_tokens_cached / max(1, len(self.cache))
            token_savings = self.total_tokens_served

            return {
                "size": len(self.cache),
                "max_size": self.max_size,
                "hits": self.hits,
                "misses": self.misses,
                "evictions": self.evictions,
                "hit_rate": hit_rate,
                "total_requests": total_requests,
                "ttl": self.ttl,
                # Generation-specific metrics
                "total_tokens_cached": self.total_tokens_cached,
                "total_tokens_served": self.total_tokens_served,
                "avg_tokens_per_entry": avg_tokens_cached,
                "token_savings": token_savings,
                "cache_efficiency": token_savings / max(1, self.total_tokens_cached + token_savings)
            }

    def get_stats_by_reasoning_level(self) -> Dict[str, Any]:
        """Get cache statistics broken down by reasoning level"""
        with self.lock:
            level_stats = {"low": 0, "medium": 0, "high": 0}
            level_tokens = {"low": 0, "medium": 0, "high": 0}

            for entry in self.cache.values():
                level = entry.reasoning_level
                if level in level_stats:
                    level_stats[level] += 1
                    level_tokens[level] += entry.tokens_used

            return {
                "entries_by_level": level_stats,
                "tokens_by_level": level_tokens,
                "avg_tokens_by_level": {
                    level: tokens / max(1, count)
                    for level, (tokens, count) in zip(level_tokens.keys(), zip(level_tokens.values(), level_stats.values()))
                }
            }

    def __repr__(self) -> str:
        stats = self.get_stats()
        return (f"GenerationCache(size={stats['size']}/{stats['max_size']}, "
                f"hit_rate={stats['hit_rate']:.2%}, "
                f"tokens_cached={stats['total_tokens_cached']}, "
                f"token_savings={stats['total_tokens_served']})")


class CachedGenerationProcessor:
    """
    Wrapper that adds caching to text generation model
    Optimized for GPT-OSS-20B generation workloads
    """

    def __init__(self, model_manager, cache: Optional[GenerationCache] = None):
        """
        Initialize cached generation processor

        Args:
            model_manager: The underlying generation model manager
            cache: Optional pre-configured cache, otherwise creates default
        """
        self.model = model_manager
        self.cache = cache or GenerationCache(max_size=200, ttl=1800)

        logger.info("CachedGenerationProcessor initialized")

    async def generate(self, prompt: str, config: Dict[str, Any], use_cache: bool = True) -> str:
        """
        Generate text with caching

        Args:
            prompt: Input prompt text
            config: Generation configuration
            use_cache: Whether to use cache (default True)

        Returns:
            Generated text
        """
        if not use_cache:
            return await self.model.generate(prompt, config)

        # Check cache first
        cached_result = self.cache.get(prompt, config)
        if cached_result is not None:
            logger.info(f"Generation cache hit for prompt: {prompt[:50]}...")
            return cached_result

        # Generate new result
        logger.info(f"Generation cache miss, generating for prompt: {prompt[:50]}...")
        start_time = time.time()
        result = await self.model.generate(prompt, config)
        generation_time = time.time() - start_time

        # Estimate token count (rough approximation)
        tokens_used = len(result.split()) * 1.3  # Rough token estimate

        # Cache the result
        self.cache.put(prompt, config, result, int(tokens_used), generation_time)

        return result

    async def chat(self, messages: List[Dict[str, str]], config: Dict[str, Any], use_cache: bool = True) -> str:
        """
        Generate chat response with caching

        Args:
            messages: Conversation messages
            config: Generation configuration
            use_cache: Whether to use cache

        Returns:
            Generated response text
        """
        if not use_cache:
            return await self.model.chat(messages, config)

        # Check cache first
        cached_result = self.cache.get_chat(messages, config)
        if cached_result is not None:
            logger.info("Chat generation cache hit")
            return cached_result

        # Generate new result
        logger.info("Chat generation cache miss, generating response...")
        start_time = time.time()
        result = await self.model.chat(messages, config)
        generation_time = time.time() - start_time

        # Estimate token count
        tokens_used = len(result.split()) * 1.3

        # Cache the result
        self.cache.put_chat(messages, config, result, int(tokens_used), generation_time)

        return result

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return self.cache.get_stats()

    def get_cache_stats_by_reasoning_level(self) -> Dict[str, Any]:
        """Get cache statistics by reasoning level"""
        return self.cache.get_stats_by_reasoning_level()

    def clear_cache(self) -> None:
        """Clear the cache"""
        self.cache.clear()

    def optimize_cache_for_reasoning_level(self, reasoning_level: str) -> None:
        """Optimize cache settings for specific reasoning level"""
        with self.cache.lock:
            if reasoning_level == "high":
                # High reasoning generates longer responses, reduce cache size
                self.cache.max_size = min(100, self.cache.max_size)
                self.cache.ttl = 900  # 15 minutes for high reasoning
            elif reasoning_level == "low":
                # Low reasoning generates shorter responses, can cache more
                self.cache.max_size = min(500, self.cache.max_size * 2)
                self.cache.ttl = 3600  # 1 hour for low reasoning
            # Medium reasoning uses default settings

        logger.info(f"Optimized cache for {reasoning_level} reasoning level: "
                   f"max_size={self.cache.max_size}, ttl={self.cache.ttl}s")


# Legacy support - keep the old class name for backward compatibility
EmbeddingCache = GenerationCache
CachedEmbeddingProcessor = CachedGenerationProcessor