"""
Memory management utilities for Apple Silicon optimization
"""

import logging
from typing import Dict, Any
import mlx.core as mx
import psutil

logger = logging.getLogger(__name__)


class MemoryManager:
    """Manage memory allocation and optimization for MLX on Apple Silicon"""
    
    def __init__(self):
        """Initialize memory manager with system detection"""
        self.total_memory = psutil.virtual_memory().total
        self.memory_profile = self.detect_profile()
        logger.info(f"Memory profile detected: {self.memory_profile}")
    
    def detect_profile(self) -> Dict[str, Any]:
        """Auto-detect optimal settings based on system memory"""
        gb = self.total_memory // (1024**3)
        logger.info(f"System memory: {gb}GB")
        
        if gb <= 16:
            return {
                "tier": "16GB",
                "quantization": "q4",
                "batch_size": 16,
                "cache_size": 500,
                "max_context": 2048,
                "metal_memory_limit": 0.70,  # 70% for tighter systems
            }
        elif gb <= 32:
            return {
                "tier": "32GB", 
                "quantization": "q8",
                "batch_size": 32,
                "cache_size": 1000,
                "max_context": 4096,
                "metal_memory_limit": 0.75,  # 75% for medium systems
            }
        else:
            return {
                "tier": "64GB+",
                "quantization": "none",
                "batch_size": 64,
                "cache_size": 2000,
                "max_context": 8192,
                "metal_memory_limit": 0.80,  # 80% for large systems
            }
    
    def optimize_for_inference(self) -> None:
        """Configure MLX for optimal inference performance"""
        logger.info("Optimizing MLX for inference workload")
        
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
        
        # Configure threading for Apple Silicon cores
        import os
        cpu_count = os.cpu_count() or 8
        try:
            if hasattr(mx, 'set_num_threads'):
                mx.set_num_threads(min(cpu_count, 8))
                logger.info(f"Set thread count to {min(cpu_count, 8)}")
        except Exception as e:
            logger.warning(f"Could not set thread count: {e}")
            
        logger.info("MLX optimization complete")
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get current memory usage statistics"""
        vm = psutil.virtual_memory()
        process = psutil.Process()
        
        return {
            "system_total_gb": vm.total / (1024**3),
            "system_available_gb": vm.available / (1024**3),
            "system_used_percent": vm.percent,
            "process_memory_gb": process.memory_info().rss / (1024**3),
            "profile_tier": self.memory_profile["tier"],
            "recommended_quantization": self.memory_profile["quantization"],
            "metal_memory_limit": self.memory_profile["metal_memory_limit"]
        }
    
    def check_memory_pressure(self) -> bool:
        """Check if system is under memory pressure"""
        vm = psutil.virtual_memory()
        return vm.percent > 85  # Consider 85%+ as high pressure
    
    def suggest_batch_size(self, default_batch_size: int) -> int:
        """Suggest optimal batch size based on memory pressure"""
        if self.check_memory_pressure():
            # Reduce batch size under pressure
            suggested = max(1, default_batch_size // 2)
            logger.warning(f"Memory pressure detected, reducing batch size to {suggested}")
            return suggested
        
        return self.memory_profile["batch_size"]
    
    def cleanup_memory(self) -> None:
        """Force cleanup of MLX memory caches"""
        logger.info("Cleaning up MLX memory caches")
        try:
            if hasattr(mx.metal, 'clear_cache'):
                mx.metal.clear_cache()
            elif hasattr(mx, 'metal') and hasattr(mx.metal, 'clear_cache'):
                mx.metal.clear_cache()
        except Exception as e:
            logger.warning(f"Could not clear cache: {e}")
        
    def get_optimal_config(self) -> Dict[str, Any]:
        """Get optimal configuration for current system"""
        return {
            "quantization": self.memory_profile["quantization"],
            "batch_size": self.memory_profile["batch_size"], 
            "max_context_length": self.memory_profile["max_context"],
            "cache_size": self.memory_profile["cache_size"],
            "metal_memory_limit": self.memory_profile["metal_memory_limit"]
        }