"""Rate limiting utilities for multi-model API calls."""

import asyncio
import time
import logging
from typing import Dict, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque

logger = logging.getLogger(__name__)


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting."""
    requests_per_minute: int = 60
    burst_size: int = 10  # Allow short bursts
    cooldown_seconds: float = 1.0  # Min time between requests


@dataclass 
class ModelRateLimitStats:
    """Statistics for a specific model's rate limiting."""
    total_requests: int = 0
    total_wait_time: float = 0.0
    rate_limit_hits: int = 0
    last_request_time: Optional[datetime] = None
    request_timestamps: deque = field(default_factory=lambda: deque(maxlen=100))


class ModelRateLimiter:
    """Rate limiter that tracks limits per model type."""
    
    def __init__(self):
        """Initialize the rate limiter."""
        self._locks: Dict[str, asyncio.Lock] = {}
        self._request_times: Dict[str, deque] = defaultdict(lambda: deque())
        self._stats: Dict[str, ModelRateLimitStats] = defaultdict(ModelRateLimitStats)
        self._configs: Dict[str, RateLimitConfig] = {}
        
        # Default configurations per model
        self._default_configs = {
            "openai/gpt-4": RateLimitConfig(requests_per_minute=60, burst_size=5),
            "openai/gpt-3.5-turbo": RateLimitConfig(requests_per_minute=90, burst_size=10),
            "anthropic/claude-3-sonnet-20240229": RateLimitConfig(requests_per_minute=60, burst_size=5),
            "google/gemini-pro": RateLimitConfig(requests_per_minute=60, burst_size=8),
            "google/gemini-2.5-flash": RateLimitConfig(requests_per_minute=120, burst_size=15),
        }
    
    def get_lock(self, model_type: str) -> asyncio.Lock:
        """Get or create a lock for the model type."""
        if model_type not in self._locks:
            self._locks[model_type] = asyncio.Lock()
        return self._locks[model_type]
    
    def get_config(self, model_type: str) -> RateLimitConfig:
        """Get rate limit configuration for a model."""
        if model_type not in self._configs:
            # Use default config or create a generic one
            self._configs[model_type] = self._default_configs.get(
                model_type, 
                RateLimitConfig()  # Generic default
            )
        return self._configs[model_type]
    
    async def acquire(self, model_type: str) -> None:
        """Acquire permission to make a request for the specified model.
        
        Args:
            model_type: The model type to rate limit
        """
        lock = self.get_lock(model_type)
        config = self.get_config(model_type)
        
        async with lock:
            await self._wait_if_needed(model_type, config)
            self._record_request(model_type)
    
    async def _wait_if_needed(self, model_type: str, config: RateLimitConfig) -> None:
        """Wait if rate limit would be exceeded."""
        now = time.time()
        request_times = self._request_times[model_type]
        stats = self._stats[model_type]
        
        # Remove requests older than 1 minute
        cutoff_time = now - 60.0
        while request_times and request_times[0] < cutoff_time:
            request_times.popleft()
        
        # Check if we're at the rate limit
        if len(request_times) >= config.requests_per_minute:
            # Calculate how long to wait
            oldest_request = request_times[0]
            wait_time = 60.0 - (now - oldest_request)
            
            if wait_time > 0:
                stats.rate_limit_hits += 1
                stats.total_wait_time += wait_time
                logger.info(f"Rate limit reached for {model_type}, waiting {wait_time:.2f}s")
                await asyncio.sleep(wait_time)
        
        # Enforce minimum time between requests
        if request_times and config.cooldown_seconds > 0:
            time_since_last = now - request_times[-1]
            if time_since_last < config.cooldown_seconds:
                wait_time = config.cooldown_seconds - time_since_last
                stats.total_wait_time += wait_time
                await asyncio.sleep(wait_time)
    
    def _record_request(self, model_type: str) -> None:
        """Record that a request was made."""
        now = time.time()
        self._request_times[model_type].append(now)
        
        stats = self._stats[model_type]
        stats.total_requests += 1
        stats.last_request_time = datetime.now()
        stats.request_timestamps.append(now)
    
    def update_from_headers(self, model_type: str, headers: Dict[str, str]) -> None:
        """Update rate limits based on API response headers.
        
        Args:
            model_type: The model type
            headers: Response headers potentially containing rate limit info
        """
        # Look for standard rate limit headers
        limit_headers = {
            "x-ratelimit-limit": "requests_per_minute",
            "x-ratelimit-remaining": "remaining",
            "x-ratelimit-reset": "reset_time"
        }
        
        rate_info = {}
        for header, key in limit_headers.items():
            if header in headers:
                try:
                    if key == "reset_time":
                        rate_info[key] = datetime.fromtimestamp(int(headers[header]))
                    else:
                        rate_info[key] = int(headers[header])
                except (ValueError, TypeError):
                    logger.debug(f"Could not parse rate limit header {header}: {headers[header]}")
        
        # Update config if we got limit info
        if "requests_per_minute" in rate_info:
            config = self.get_config(model_type)
            new_limit = rate_info["requests_per_minute"]
            if new_limit != config.requests_per_minute:
                logger.info(f"Updating rate limit for {model_type}: {config.requests_per_minute} -> {new_limit}")
                config.requests_per_minute = new_limit
    
    def get_stats(self, model_type: Optional[str] = None) -> Dict[str, ModelRateLimitStats]:
        """Get rate limiting statistics.
        
        Args:
            model_type: Specific model to get stats for, or None for all models
            
        Returns:
            Dictionary of model type to stats
        """
        if model_type:
            return {model_type: self._stats[model_type]}
        return dict(self._stats)
    
    def get_current_rate(self, model_type: str) -> Tuple[int, int]:
        """Get current request rate for a model.
        
        Args:
            model_type: The model type
            
        Returns:
            Tuple of (requests_in_last_minute, limit_per_minute)
        """
        now = time.time()
        cutoff = now - 60.0
        request_times = self._request_times[model_type]
        
        # Count requests in last minute
        recent_count = sum(1 for t in request_times if t >= cutoff)
        limit = self.get_config(model_type).requests_per_minute
        
        return recent_count, limit
    
    def reset_stats(self) -> None:
        """Reset all statistics (but keep configurations)."""
        self._stats.clear()
        self._request_times.clear()
        logger.info("Rate limiter statistics reset")


class PriorityQueue:
    """Priority queue for mixed-model experiments to optimize throughput."""
    
    def __init__(self, rate_limiter: ModelRateLimiter):
        """Initialize with a rate limiter instance."""
        self.rate_limiter = rate_limiter
        self._queues: Dict[str, deque] = defaultdict(deque)
        self._processing = False
    
    async def enqueue(self, model_type: str, task) -> None:
        """Add a task to the queue for a specific model.
        
        Args:
            model_type: The model type
            task: Coroutine to execute (will be awaited)
        """
        self._queues[model_type].append(task)
        
        # Start processing if not already running
        if not self._processing:
            asyncio.create_task(self._process_queues())
    
    async def _process_queues(self) -> None:
        """Process queued tasks while respecting rate limits."""
        self._processing = True
        
        try:
            while any(self._queues.values()):
                # Find models with available capacity
                available_models = []
                
                for model_type, queue in self._queues.items():
                    if not queue:
                        continue
                    
                    current, limit = self.rate_limiter.get_current_rate(model_type)
                    if current < limit:
                        available_models.append((model_type, limit - current))
                
                if not available_models:
                    # All models at capacity, wait a bit
                    await asyncio.sleep(0.5)
                    continue
                
                # Sort by available capacity (highest first)
                available_models.sort(key=lambda x: x[1], reverse=True)
                
                # Process one task from the model with most capacity
                model_type, _ = available_models[0]
                task = self._queues[model_type].popleft()
                
                # Acquire rate limit and run task
                await self.rate_limiter.acquire(model_type)
                # Check if it's a coroutine or already a task
                if asyncio.iscoroutine(task):
                    await task
                else:
                    await task
                
        finally:
            self._processing = False