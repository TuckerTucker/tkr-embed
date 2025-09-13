"""
Memory management utilities for Apple Silicon optimization
Updated for 21B model text generation workloads
"""

import logging
from typing import Dict, Any, Optional
import mlx.core as mx
import psutil

logger = logging.getLogger(__name__)


class MemoryManager:
    """Manage memory allocation and optimization for MLX on Apple Silicon with 21B model support"""

    def __init__(self):
        """Initialize memory manager with system detection"""
        self.total_memory = psutil.virtual_memory().total
        self.memory_profile = self.detect_profile()
        self.generation_mode = True  # Flag for generation-optimized settings
        logger.info(f"Memory profile detected: {self.memory_profile}")

    def detect_profile(self) -> Dict[str, Any]:
        """Auto-detect optimal settings for 21B model generation based on system memory"""
        gb = self.total_memory // (1024**3)
        logger.info(f"System memory: {gb}GB (optimized for 21B model generation)")

        if gb <= 16:
            # Minimal support for 21B model - heavy quantization required
            return {
                "tier": "16GB",
                "quantization": "q4",  # Heavy quantization for 21B model
                "batch_size": 1,  # Single request only
                "cache_size": 100,  # Minimal cache for generation results
                "max_tokens": 2048,  # Reduced token limit
                "metal_memory_limit": 0.85,  # Higher allocation for 21B model
                "kv_cache_size": 0.1,  # Small KV cache
                "model_memory_gb": 12,  # Expected model memory for Q4
            }
        elif gb <= 32:
            # Standard configuration for 21B model
            return {
                "tier": "32GB",
                "quantization": "q8",  # Balanced quantization
                "batch_size": 2,  # Small batch processing
                "cache_size": 200,  # Moderate cache
                "max_tokens": 4096,  # Standard token limit
                "metal_memory_limit": 0.80,  # Balanced allocation
                "kv_cache_size": 0.15,  # Moderate KV cache
                "model_memory_gb": 18,  # Expected model memory for Q8
            }
        elif gb <= 64:
            # Optimal configuration for 21B model
            return {
                "tier": "64GB",
                "quantization": "mxfp4",  # High-quality quantization
                "batch_size": 4,  # Moderate batch processing
                "cache_size": 500,  # Large cache
                "max_tokens": 8192,  # Extended token limit
                "metal_memory_limit": 0.75,  # Conservative for stability
                "kv_cache_size": 0.2,  # Large KV cache
                "model_memory_gb": 25,  # Expected model memory for MXFP4
            }
        else:
            # High-end configuration for 21B model
            return {
                "tier": "64GB+",
                "quantization": "none",  # Full precision if memory allows
                "batch_size": 8,  # Larger batch processing
                "cache_size": 1000,  # Maximum cache
                "max_tokens": 16384,  # Maximum token limit
                "metal_memory_limit": 0.70,  # Conservative for large model
                "kv_cache_size": 0.25,  # Maximum KV cache
                "model_memory_gb": 40,  # Expected model memory for full precision
            }
    
    def optimize_for_inference(self) -> None:
        """Configure MLX for optimal inference performance"""
        logger.warning("optimize_for_inference is deprecated, use optimize_for_generation instead")
        self.optimize_for_generation()

    def optimize_for_generation(self) -> None:
        """Configure MLX for optimal text generation performance with 21B model"""
        logger.info("Optimizing MLX for 21B model text generation workload")

        try:
            # Set memory limits based on profile (convert percentage to bytes)
            memory_limit_pct = self.memory_profile["metal_memory_limit"]
            memory_limit_bytes = int(self.total_memory * memory_limit_pct)

            # Use the current MLX API
            if hasattr(mx, 'set_memory_limit'):
                mx.set_memory_limit(memory_limit_bytes)
                logger.info(f"Set memory limit to {memory_limit_pct*100}% ({memory_limit_bytes // (1024**3)}GB)")
            elif hasattr(mx.metal, 'set_memory_limit'):
                mx.metal.set_memory_limit(memory_limit_bytes)
                logger.info(f"Set Metal memory limit to {memory_limit_pct*100}% ({memory_limit_bytes // (1024**3)}GB)")

        except Exception as e:
            logger.warning(f"Could not set memory limit: {e}")

        # Configure threading optimized for generation workloads
        import os
        cpu_count = os.cpu_count() or 8
        try:
            # Use more threads for generation due to sequential processing
            optimal_threads = min(cpu_count, 12) if self.generation_mode else min(cpu_count, 8)
            if hasattr(mx, 'set_num_threads'):
                mx.set_num_threads(optimal_threads)
                logger.info(f"Set thread count to {optimal_threads} (generation optimized)")
        except Exception as e:
            logger.warning(f"Could not set thread count: {e}")

        logger.info("MLX generation optimization complete")

    def allocate_for_model(self, size_gb: float) -> bool:
        """Allocate memory for 21B model loading"""
        available_gb = self.get_available_memory()
        expected_model_gb = self.memory_profile["model_memory_gb"]

        logger.info(f"Allocating {size_gb}GB for model (expected: {expected_model_gb}GB, available: {available_gb}GB)")

        if size_gb > available_gb:
            logger.error(f"Insufficient memory: need {size_gb}GB, have {available_gb}GB")
            return False

        if size_gb > expected_model_gb * 1.2:  # 20% tolerance
            logger.warning(f"Model size {size_gb}GB exceeds expected {expected_model_gb}GB for {self.memory_profile['tier']}")

        return True
    
    def get_available_memory(self) -> float:
        """Get available system memory in GB"""
        vm = psutil.virtual_memory()
        return vm.available / (1024**3)

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get current memory usage statistics for generation workloads"""
        vm = psutil.virtual_memory()
        process = psutil.Process()

        stats = {
            "system_total_gb": vm.total / (1024**3),
            "system_available_gb": vm.available / (1024**3),
            "system_used_percent": vm.percent,
            "process_memory_gb": process.memory_info().rss / (1024**3),
            "profile_tier": self.memory_profile["tier"],
            "recommended_quantization": self.memory_profile["quantization"],
            "metal_memory_limit": self.memory_profile["metal_memory_limit"],
            # Generation-specific metrics
            "generation_mode": self.generation_mode,
            "max_tokens": self.memory_profile["max_tokens"],
            "expected_model_memory_gb": self.memory_profile["model_memory_gb"],
            "kv_cache_size": self.memory_profile["kv_cache_size"],
            "optimal_batch_size": self.memory_profile["batch_size"],
        }

        # Calculate memory pressure specific to generation
        model_memory_used = self.memory_profile["model_memory_gb"]
        available_for_kv = vm.available / (1024**3) - model_memory_used
        kv_cache_capacity = available_for_kv * self.memory_profile["kv_cache_size"]

        stats.update({
            "available_for_kv_cache_gb": available_for_kv,
            "kv_cache_capacity_gb": kv_cache_capacity,
            "generation_memory_pressure": vm.percent > 80  # Different threshold for generation
        })

        return stats
    
    def check_memory_pressure(self) -> bool:
        """Check if system is under memory pressure for generation workloads"""
        vm = psutil.virtual_memory()
        # Lower threshold for generation due to KV cache growth
        return vm.percent > 80  # Consider 80%+ as high pressure for generation

    def suggest_batch_size(self, default_batch_size: int) -> int:
        """Suggest optimal batch size for generation based on memory pressure"""
        if self.check_memory_pressure():
            # Be more aggressive in reducing batch size for generation
            suggested = max(1, default_batch_size // 2)
            logger.warning(f"Generation memory pressure detected, reducing batch size to {suggested}")
            return suggested

        return self.memory_profile["batch_size"]

    def suggest_max_tokens(self, requested_tokens: int) -> int:
        """Suggest safe max tokens based on available memory"""
        max_tokens = self.memory_profile["max_tokens"]

        if self.check_memory_pressure():
            # Reduce token limit under pressure
            safe_tokens = min(requested_tokens, max_tokens // 2)
            logger.warning(f"Memory pressure detected, limiting tokens to {safe_tokens}")
            return safe_tokens

        return min(requested_tokens, max_tokens)

    def monitor_kv_cache_usage(self) -> Dict[str, Any]:
        """Monitor KV cache usage for generation workloads"""
        stats = self.get_memory_stats()

        return {
            "kv_cache_capacity_gb": stats["kv_cache_capacity_gb"],
            "available_for_kv_gb": stats["available_for_kv_cache_gb"],
            "kv_cache_utilization": min(1.0, stats["kv_cache_capacity_gb"] / max(0.1, stats["available_for_kv_cache_gb"])),
            "recommended_max_tokens": self.memory_profile["max_tokens"],
            "memory_pressure": stats["generation_memory_pressure"]
        }

    def cleanup_memory(self) -> None:
        """Force cleanup of MLX memory caches including KV cache"""
        logger.info("Cleaning up MLX memory caches (including KV cache)")
        try:
            if hasattr(mx.metal, 'clear_cache'):
                mx.metal.clear_cache()
            elif hasattr(mx, 'metal') and hasattr(mx.metal, 'clear_cache'):
                mx.metal.clear_cache()
            logger.info("KV cache and model cache cleared")
        except Exception as e:
            logger.warning(f"Could not clear cache: {e}")

    def get_optimal_config(self) -> Dict[str, Any]:
        """Get optimal configuration for 21B model generation"""
        return {
            "quantization": self.memory_profile["quantization"],
            "batch_size": self.memory_profile["batch_size"],
            "max_tokens": self.memory_profile["max_tokens"],
            "cache_size": self.memory_profile["cache_size"],
            "metal_memory_limit": self.memory_profile["metal_memory_limit"],
            "kv_cache_size": self.memory_profile["kv_cache_size"],
            "model_memory_gb": self.memory_profile["model_memory_gb"],
            "generation_mode": self.generation_mode
        }