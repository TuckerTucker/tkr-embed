"""
Rate limiting middleware for MLX embedding server
Prevents abuse and manages API usage quotas
"""

import time
import asyncio
from typing import Dict, Any, Optional, Tuple
from collections import defaultdict, deque
from datetime import datetime, timedelta
import logging
from fastapi import HTTPException, Request
import threading
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class RateLimit:
    """Rate limit configuration"""
    requests_per_minute: int = 60
    requests_per_hour: int = 1000
    requests_per_day: int = 10000
    burst_size: int = 10  # Allow burst of requests


@dataclass
class Usage:
    """Usage tracking for rate limiting"""
    requests: int = 0
    first_request: float = 0
    last_request: float = 0
    
    def reset_if_expired(self, window_seconds: int) -> bool:
        """Reset usage if window has expired"""
        now = time.time()
        if now - self.first_request > window_seconds:
            self.requests = 0
            self.first_request = now
            return True
        return False


class TokenBucket:
    """Token bucket algorithm for rate limiting"""
    
    def __init__(self, capacity: int, refill_rate: float):
        """
        Initialize token bucket
        
        Args:
            capacity: Maximum number of tokens
            refill_rate: Tokens added per second
        """
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.tokens = capacity
        self.last_refill = time.time()
        self.lock = threading.Lock()
    
    def consume(self, tokens: int = 1) -> bool:
        """
        Try to consume tokens from bucket
        
        Args:
            tokens: Number of tokens to consume
            
        Returns:
            True if tokens were consumed, False otherwise
        """
        with self.lock:
            now = time.time()
            
            # Refill tokens
            time_passed = now - self.last_refill
            tokens_to_add = time_passed * self.refill_rate
            self.tokens = min(self.capacity, self.tokens + tokens_to_add)
            self.last_refill = now
            
            # Try to consume tokens
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            
            return False
    
    def get_wait_time(self, tokens: int = 1) -> float:
        """Get time to wait before tokens are available"""
        with self.lock:
            if self.tokens >= tokens:
                return 0.0
            
            needed_tokens = tokens - self.tokens
            return needed_tokens / self.refill_rate


class SlidingWindowCounter:
    """Sliding window counter for rate limiting"""
    
    def __init__(self, window_size: int, limit: int):
        """
        Initialize sliding window counter
        
        Args:
            window_size: Window size in seconds
            limit: Maximum requests in window
        """
        self.window_size = window_size
        self.limit = limit
        self.requests = deque()
        self.lock = threading.Lock()
    
    def is_allowed(self) -> bool:
        """Check if request is allowed"""
        with self.lock:
            now = time.time()
            
            # Remove old requests outside window
            while self.requests and self.requests[0] <= now - self.window_size:
                self.requests.popleft()
            
            # Check if limit exceeded
            if len(self.requests) >= self.limit:
                return False
            
            # Add current request
            self.requests.append(now)
            return True
    
    def get_reset_time(self) -> float:
        """Get time when rate limit resets"""
        with self.lock:
            if not self.requests:
                return 0.0
            return self.requests[0] + self.window_size


class RateLimiter:
    """Comprehensive rate limiter with multiple algorithms"""
    
    def __init__(self):
        # Storage for different rate limiting strategies
        self.token_buckets: Dict[str, TokenBucket] = {}
        self.sliding_windows: Dict[str, SlidingWindowCounter] = {}
        self.usage_counters: Dict[str, Dict[str, Usage]] = defaultdict(dict)
        
        # Default rate limits
        self.default_limits = RateLimit(
            requests_per_minute=60,
            requests_per_hour=1000,
            requests_per_day=10000,
            burst_size=10
        )
        
        # Custom limits per API key or IP
        self.custom_limits: Dict[str, RateLimit] = {}
        
        # Cleanup task
        self._cleanup_task = None
        
        logger.info("RateLimiter initialized")
    
    def set_custom_limit(self, identifier: str, rate_limit: RateLimit):
        """Set custom rate limit for specific API key or IP"""
        self.custom_limits[identifier] = rate_limit
        logger.info(f"Set custom rate limit for {identifier}")
    
    def get_rate_limit(self, identifier: str) -> RateLimit:
        """Get rate limit configuration for identifier"""
        return self.custom_limits.get(identifier, self.default_limits)
    
    def _get_token_bucket_key(self, identifier: str, window: str) -> str:
        """Generate key for token bucket storage"""
        return f"bucket:{identifier}:{window}"
    
    def _get_sliding_window_key(self, identifier: str, window: str) -> str:
        """Generate key for sliding window storage"""
        return f"window:{identifier}:{window}"
    
    def check_rate_limit(self, identifier: str, endpoint: str = "default") -> Tuple[bool, Dict[str, Any]]:
        """
        Check if request is allowed under rate limits
        
        Args:
            identifier: API key, IP address, or other identifier
            endpoint: API endpoint for per-endpoint limits
            
        Returns:
            Tuple of (is_allowed, rate_limit_info)
        """
        rate_limit = self.get_rate_limit(identifier)
        now = time.time()
        
        # Check token bucket (burst protection)
        bucket_key = self._get_token_bucket_key(identifier, "burst")
        if bucket_key not in self.token_buckets:
            self.token_buckets[bucket_key] = TokenBucket(
                capacity=rate_limit.burst_size,
                refill_rate=rate_limit.requests_per_minute / 60.0
            )
        
        bucket = self.token_buckets[bucket_key]
        if not bucket.consume(1):
            wait_time = bucket.get_wait_time(1)
            return False, {
                "error": "Rate limit exceeded (burst)",
                "retry_after": wait_time,
                "limit_type": "burst"
            }
        
        # Check sliding window limits
        windows = [
            ("minute", 60, rate_limit.requests_per_minute),
            ("hour", 3600, rate_limit.requests_per_hour),
            ("day", 86400, rate_limit.requests_per_day)
        ]
        
        for window_name, window_size, limit in windows:
            window_key = self._get_sliding_window_key(identifier, window_name)
            if window_key not in self.sliding_windows:
                self.sliding_windows[window_key] = SlidingWindowCounter(window_size, limit)
            
            window = self.sliding_windows[window_key]
            if not window.is_allowed():
                reset_time = window.get_reset_time()
                return False, {
                    "error": f"Rate limit exceeded ({window_name})",
                    "retry_after": reset_time - now,
                    "limit_type": window_name,
                    "limit": limit,
                    "reset_time": reset_time
                }
        
        # Request allowed
        return True, {
            "allowed": True,
            "limits": {
                "burst": rate_limit.burst_size,
                "per_minute": rate_limit.requests_per_minute,
                "per_hour": rate_limit.requests_per_hour,
                "per_day": rate_limit.requests_per_day
            }
        }
    
    def get_usage_stats(self, identifier: str) -> Dict[str, Any]:
        """Get usage statistics for identifier"""
        stats = {"identifier": identifier}
        
        # Get current usage from sliding windows
        for window_name in ["minute", "hour", "day"]:
            window_key = self._get_sliding_window_key(identifier, window_name)
            if window_key in self.sliding_windows:
                window = self.sliding_windows[window_key]
                with window.lock:
                    current_usage = len(window.requests)
                    stats[f"requests_in_{window_name}"] = current_usage
            else:
                stats[f"requests_in_{window_name}"] = 0
        
        # Get token bucket status
        bucket_key = self._get_token_bucket_key(identifier, "burst")
        if bucket_key in self.token_buckets:
            bucket = self.token_buckets[bucket_key]
            with bucket.lock:
                stats["burst_tokens_available"] = int(bucket.tokens)
                stats["burst_capacity"] = bucket.capacity
        
        return stats
    
    async def cleanup_expired_entries(self):
        """Cleanup expired rate limiting entries"""
        while True:
            try:
                now = time.time()
                
                # Cleanup sliding windows (remove old entries)
                windows_to_remove = []
                for key, window in self.sliding_windows.items():
                    with window.lock:
                        # Remove old requests
                        while window.requests and window.requests[0] <= now - window.window_size:
                            window.requests.popleft()
                        
                        # Mark for removal if empty for too long
                        if not window.requests:
                            windows_to_remove.append(key)
                
                # Remove empty windows
                for key in windows_to_remove:
                    if key in self.sliding_windows:
                        del self.sliding_windows[key]
                
                logger.debug(f"Cleaned up {len(windows_to_remove)} expired rate limit entries")
                
                # Sleep for 5 minutes before next cleanup
                await asyncio.sleep(300)
                
            except Exception as e:
                logger.error(f"Error in rate limiter cleanup: {e}")
                await asyncio.sleep(60)  # Shorter sleep on error
    
    def start_cleanup_task(self):
        """Start the background cleanup task"""
        if self._cleanup_task is None or self._cleanup_task.done():
            self._cleanup_task = asyncio.create_task(self.cleanup_expired_entries())
            logger.info("Started rate limiter cleanup task")
    
    def stop_cleanup_task(self):
        """Stop the background cleanup task"""
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()
            logger.info("Stopped rate limiter cleanup task")


# Global rate limiter instance
rate_limiter = RateLimiter()


class RateLimitExceeded(HTTPException):
    """Rate limit exceeded exception"""
    
    def __init__(self, detail: str, retry_after: float, headers: Dict[str, str] = None):
        super().__init__(status_code=429, detail=detail, headers=headers or {})
        self.retry_after = retry_after


async def apply_rate_limit(request: Request, identifier: str = None) -> Dict[str, Any]:
    """
    Apply rate limiting to request
    
    Args:
        request: FastAPI request object
        identifier: Optional custom identifier (defaults to IP + API key)
    
    Returns:
        Rate limit information
        
    Raises:
        RateLimitExceeded: If rate limit is exceeded
    """
    if identifier is None:
        # Use combination of IP and API key if available
        client_ip = request.client.host if request.client else "unknown"
        api_key = request.headers.get("X-API-Key", "")
        identifier = f"{client_ip}:{api_key}" if api_key else client_ip
    
    # Get endpoint path for per-endpoint limiting
    endpoint = request.url.path
    
    # Check rate limits
    allowed, rate_info = rate_limiter.check_rate_limit(identifier, endpoint)
    
    if not allowed:
        retry_after = rate_info.get("retry_after", 60)
        headers = {
            "Retry-After": str(int(retry_after)),
            "X-RateLimit-Limit": str(rate_info.get("limit", "unknown")),
            "X-RateLimit-Remaining": "0",
            "X-RateLimit-Reset": str(int(time.time() + retry_after))
        }
        
        logger.warning(f"Rate limit exceeded for {identifier}: {rate_info['error']}")
        raise RateLimitExceeded(
            detail=rate_info["error"],
            retry_after=retry_after,
            headers=headers
        )
    
    return rate_info


def create_rate_limit_headers(identifier: str) -> Dict[str, str]:
    """Create rate limit headers for response"""
    stats = rate_limiter.get_usage_stats(identifier)
    rate_limit = rate_limiter.get_rate_limit(identifier)
    
    return {
        "X-RateLimit-Limit-Minute": str(rate_limit.requests_per_minute),
        "X-RateLimit-Limit-Hour": str(rate_limit.requests_per_hour),
        "X-RateLimit-Limit-Day": str(rate_limit.requests_per_day),
        "X-RateLimit-Remaining-Minute": str(max(0, rate_limit.requests_per_minute - stats.get("requests_in_minute", 0))),
        "X-RateLimit-Remaining-Hour": str(max(0, rate_limit.requests_per_hour - stats.get("requests_in_hour", 0))),
        "X-RateLimit-Remaining-Day": str(max(0, rate_limit.requests_per_day - stats.get("requests_in_day", 0))),
    }