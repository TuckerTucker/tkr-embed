"""
Batch processing optimization for text generation workloads
Adapted for GPT-OSS-20B variable-length text generation
Handles reasoning levels and optimizes for generation throughput
"""

import logging
from typing import List, Optional, Dict, Any, Union, AsyncIterator
import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import time

logger = logging.getLogger(__name__)


@dataclass
class GenerationBatchConfig:
    """Configuration for generation batch processing"""
    max_batch_size: int = 4  # Maximum generations per batch (smaller for generation)
    max_wait_time: float = 0.05  # Maximum wait time for batch accumulation (faster for generation)
    dynamic_batching: bool = True  # Adjust batch size based on memory and reasoning level
    memory_threshold: float = 0.80  # Lower threshold for generation workloads
    max_tokens_per_batch: int = 8192  # Maximum total tokens per batch
    reasoning_level_weights: Dict[str, float] = None  # Batch size multipliers by reasoning level

    def __post_init__(self):
        if self.reasoning_level_weights is None:
            self.reasoning_level_weights = {
                "low": 1.5,      # Can batch more low reasoning requests
                "medium": 1.0,   # Standard batching
                "high": 0.5      # Reduce batch size for high reasoning
            }


class GenerationBatchProcessor:
    """Optimized batch processing for text generation workloads"""

    def __init__(self, model_manager, config: Optional[GenerationBatchConfig] = None):
        """
        Initialize generation batch processor

        Args:
            model_manager: The GPT-OSS-20B model manager instance
            config: Generation batch processing configuration
        """
        self.model = model_manager
        self.config = config or GenerationBatchConfig()
        self.executor = ThreadPoolExecutor(max_workers=4)  # More workers for generation

        # Batch accumulation by reasoning level
        self.pending_generations: Dict[str, List[Dict]] = {
            "low": [],
            "medium": [],
            "high": []
        }

        # Performance tracking
        self.total_processed = 0
        self.total_time = 0.0
        self.total_tokens_generated = 0
        self.generations_by_level = {"low": 0, "medium": 0, "high": 0}

        logger.info(f"GenerationBatchProcessor initialized with max_batch_size={self.config.max_batch_size}")
    
    async def generate_batch(self, requests: List[Dict[str, Any]]) -> List[str]:
        """
        Generate text for a batch of requests efficiently

        Args:
            requests: List of generation request dictionaries with prompt, config

        Returns:
            List of generated text strings
        """
        if not self.model.is_ready():
            raise RuntimeError("Model not ready for generation")

        batch_size = len(requests)
        logger.debug(f"Processing generation batch of size {batch_size}")

        # Group by reasoning level for optimal batching
        batches_by_level = self._group_by_reasoning_level(requests)

        # Process each reasoning level batch
        all_results = []
        for level, level_requests in batches_by_level.items():
            if not level_requests:
                continue

            level_results = await self._process_generation_batch(level_requests, level)
            all_results.extend(level_results)

        return all_results

    def _group_by_reasoning_level(self, requests: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Group requests by reasoning level for optimal batching"""
        groups = {"low": [], "medium": [], "high": []}

        for request in requests:
            reasoning_level = request.get("config", {}).get("reasoning_level", "medium")
            if reasoning_level in groups:
                groups[reasoning_level].append(request)

        return groups
    
    async def _process_generation_batch(self, requests: List[Dict[str, Any]], reasoning_level: str) -> List[str]:
        """
        Process a single batch of generation requests

        Args:
            requests: List of generation requests (size <= optimal batch size for level)
            reasoning_level: The reasoning level for this batch

        Returns:
            List of generated text strings
        """
        start_time = time.time()
        optimal_batch_size = self.get_optimal_batch_size_for_level(reasoning_level)

        # Split into sub-batches if needed
        if len(requests) > optimal_batch_size:
            results = []
            for i in range(0, len(requests), optimal_batch_size):
                sub_batch = requests[i:i + optimal_batch_size]
                sub_results = await self._process_single_generation_batch(sub_batch, reasoning_level)
                results.extend(sub_results)
            return results
        else:
            return await self._process_single_generation_batch(requests, reasoning_level)

    async def _process_single_generation_batch(self, requests: List[Dict[str, Any]], reasoning_level: str) -> List[str]:
        """
        Process a single generation batch at the atomic level

        Args:
            requests: List of generation requests
            reasoning_level: The reasoning level for this batch

        Returns:
            List of generated text strings
        """
        results = []

        # For now, process sequentially as generation is inherently sequential
        # TODO: Implement true parallel generation when model supports it
        for request in requests:
            try:
                prompt = request.get("prompt", "")
                config = request.get("config", {})

                # Generate response
                result = await self.model.generate(prompt, config)
                results.append(result)

                # Track metrics
                self.generations_by_level[reasoning_level] += 1
                self.total_tokens_generated += len(result.split()) * 1.3  # Estimate tokens

            except Exception as e:
                logger.error(f"Generation failed for prompt: {request.get('prompt', '')[:50]}... Error: {e}")
                results.append("")  # Empty result for failed generation

        return results
    
    async def chat_batch(self, requests: List[Dict[str, Any]]) -> List[str]:
        """
        Generate chat responses for a batch of conversation requests

        Args:
            requests: List of chat request dictionaries with messages, config

        Returns:
            List of generated response strings
        """
        if not self.model.is_ready():
            raise RuntimeError("Model not ready for chat generation")

        batch_size = len(requests)
        logger.debug(f"Processing chat batch of size {batch_size}")

        # Group by reasoning level for optimal batching
        batches_by_level = self._group_chat_by_reasoning_level(requests)

        # Process each reasoning level batch
        all_results = []
        for level, level_requests in batches_by_level.items():
            if not level_requests:
                continue

            level_results = await self._process_chat_batch(level_requests, level)
            all_results.extend(level_results)

        return all_results

    def _group_chat_by_reasoning_level(self, requests: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Group chat requests by reasoning level for optimal batching"""
        groups = {"low": [], "medium": [], "high": []}

        for request in requests:
            reasoning_level = request.get("config", {}).get("reasoning_level", "medium")
            if reasoning_level in groups:
                groups[reasoning_level].append(request)

        return groups

    async def _process_chat_batch(self, requests: List[Dict[str, Any]], reasoning_level: str) -> List[str]:
        """Process a batch of chat generation requests"""
        results = []

        for request in requests:
            try:
                messages = request.get("messages", [])
                config = request.get("config", {})

                # Generate chat response
                result = await self.model.chat(messages, config)
                results.append(result)

                # Track metrics
                self.generations_by_level[reasoning_level] += 1
                self.total_tokens_generated += len(result.split()) * 1.3

            except Exception as e:
                logger.error(f"Chat generation failed for conversation. Error: {e}")
                results.append("")  # Empty result for failed generation

        return results
    
    async def stream_generation_batch(self, requests: List[Dict[str, Any]]) -> AsyncIterator[Dict[str, Any]]:
        """
        Process streaming generation requests in batches

        Args:
            requests: List of streaming generation request dictionaries

        Yields:
            Dictionary with request_id and generated chunk
        """
        if not self.model.is_ready():
            raise RuntimeError("Model not ready for streaming generation")

        logger.info(f"Processing streaming batch of {len(requests)} requests")

        # For streaming, process requests concurrently but independently
        tasks = []
        for i, request in enumerate(requests):
            task = self._process_streaming_request(request, i)
            tasks.append(task)

        # Yield results as they become available
        async for result in self._yield_streaming_results(tasks):
            yield result

    async def _process_streaming_request(self, request: Dict[str, Any], request_id: int) -> AsyncIterator[Dict[str, Any]]:
        """Process a single streaming generation request"""
        try:
            prompt = request.get("prompt", "")
            config = request.get("config", {})
            config["streaming"] = True

            async for chunk in self.model.generate_stream(prompt, config):
                yield {"request_id": request_id, "chunk": chunk}

        except Exception as e:
            logger.error(f"Streaming generation failed for request {request_id}: {e}")
            yield {"request_id": request_id, "chunk": "", "error": str(e)}

    async def _yield_streaming_results(self, tasks: List[AsyncIterator]) -> AsyncIterator[Dict[str, Any]]:
        """Yield results from multiple streaming tasks as they become available"""
        # This is a simplified implementation
        # In practice, you'd want more sophisticated multiplexing
        for task in tasks:
            async for result in task:
                yield result
    
    def get_optimal_batch_size_for_level(self, reasoning_level: str = "medium") -> int:
        """
        Determine optimal batch size based on reasoning level and memory

        Args:
            reasoning_level: The reasoning level ("low", "medium", "high")

        Returns:
            Optimal batch size for current system state and reasoning level
        """
        if not self.config.dynamic_batching:
            base_size = self.config.max_batch_size
        else:
            base_size = self._get_memory_adjusted_batch_size()

        # Apply reasoning level weight
        level_weight = self.config.reasoning_level_weights.get(reasoning_level, 1.0)
        optimal_size = max(1, int(base_size * level_weight))

        logger.debug(f"Optimal batch size for {reasoning_level} reasoning: {optimal_size}")
        return optimal_size

    def _get_memory_adjusted_batch_size(self) -> int:
        """Get memory-adjusted base batch size"""
        try:
            import psutil
            memory_percent = psutil.virtual_memory().percent / 100.0

            if memory_percent > self.config.memory_threshold:
                # Reduce batch size when memory is high (more aggressive for generation)
                return max(1, self.config.max_batch_size // 3)
            elif memory_percent < 0.6:
                # Slightly increase batch size when memory is available
                return min(self.config.max_batch_size * 1.5, self.config.max_batch_size + 2)
            else:
                return self.config.max_batch_size

        except Exception as e:
            logger.warning(f"Failed to determine memory-adjusted batch size: {e}")
            return self.config.max_batch_size

    def estimate_batch_token_usage(self, requests: List[Dict[str, Any]]) -> int:
        """Estimate total token usage for a batch of requests"""
        total_tokens = 0
        for request in requests:
            # Estimate input tokens
            prompt = request.get("prompt", "")
            input_tokens = len(prompt.split()) * 1.3

            # Estimate output tokens based on max_tokens config
            config = request.get("config", {})
            max_tokens = config.get("max_tokens", 4096)
            reasoning_level = config.get("reasoning_level", "medium")

            # Adjust expected output length by reasoning level
            reasoning_multiplier = {"low": 0.5, "medium": 1.0, "high": 1.5}.get(reasoning_level, 1.0)
            estimated_output = max_tokens * 0.7 * reasoning_multiplier  # Assume 70% of max_tokens on average

            total_tokens += int(input_tokens + estimated_output)

        return total_tokens

    def should_split_batch(self, requests: List[Dict[str, Any]]) -> bool:
        """Determine if a batch should be split based on token limits"""
        estimated_tokens = self.estimate_batch_token_usage(requests)
        return estimated_tokens > self.config.max_tokens_per_batch
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get generation batch processing statistics

        Returns:
            Dictionary of performance statistics including token metrics
        """
        avg_time = self.total_time / max(1, self.total_processed)
        throughput = self.total_processed / max(0.001, self.total_time)
        token_throughput = self.total_tokens_generated / max(0.001, self.total_time)

        return {
            "total_processed": self.total_processed,
            "total_time": self.total_time,
            "average_time_per_generation": avg_time,
            "generations_per_second": throughput,
            "tokens_per_second": token_throughput,
            "total_tokens_generated": self.total_tokens_generated,
            "avg_tokens_per_generation": self.total_tokens_generated / max(1, self.total_processed),
            "generations_by_level": self.generations_by_level.copy(),
            "optimal_batch_sizes": {
                level: self.get_optimal_batch_size_for_level(level)
                for level in ["low", "medium", "high"]
            },
            "max_batch_size": self.config.max_batch_size,
            "max_tokens_per_batch": self.config.max_tokens_per_batch
        }

    def reset_stats(self) -> None:
        """Reset performance statistics"""
        self.total_processed = 0
        self.total_time = 0.0
        self.total_tokens_generated = 0
        self.generations_by_level = {"low": 0, "medium": 0, "high": 0}
        logger.info("Generation batch processor stats reset")


# Legacy support - keep the old class name for backward compatibility
BatchProcessor = GenerationBatchProcessor
BatchConfig = GenerationBatchConfig