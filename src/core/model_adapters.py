"""Model adapter framework for multi-model support."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import asyncio
import logging
from datetime import datetime

from src.core.models import ModelConfig, ModelType

logger = logging.getLogger(__name__)


class ModelAdapter(ABC):
    """Abstract base class for model adapters."""
    
    def __init__(self, model_config: ModelConfig):
        """Initialize adapter with model configuration.
        
        Args:
            model_config: Configuration for the specific model
        """
        self.model_config = model_config
        self.last_request_time: Optional[datetime] = None
    
    @abstractmethod
    def get_request_params(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Get model-specific request parameters.
        
        Args:
            prompt: The prompt to send to the model
            **kwargs: Additional parameters
            
        Returns:
            Dictionary of parameters formatted for the specific model
        """
        pass
    
    @abstractmethod
    def parse_response(self, response: Dict[str, Any]) -> str:
        """Parse model-specific response format.
        
        Args:
            response: Raw response from the API
            
        Returns:
            Extracted text content from the response
        """
        pass
    
    async def enforce_rate_limit(self, rate_limiter=None):
        """Enforce rate limiting based on model configuration.
        
        Args:
            rate_limiter: Optional ModelRateLimiter instance for centralized rate limiting
        """
        # If a centralized rate limiter is provided, use it
        if rate_limiter:
            await rate_limiter.acquire(self.model_config.model_type)
        else:
            # Fall back to simple per-adapter rate limiting
            if self.last_request_time is not None:
                elapsed = (datetime.now() - self.last_request_time).total_seconds()
                min_interval = 60.0 / self.model_config.rate_limit
                
                if elapsed < min_interval:
                    sleep_time = min_interval - elapsed
                    logger.debug(f"Rate limiting: sleeping for {sleep_time:.2f}s")
                    await asyncio.sleep(sleep_time)
            
            self.last_request_time = datetime.now()
    
    def get_headers(self) -> Dict[str, str]:
        """Get headers for API request.
        
        Returns:
            Headers including authorization
        """
        return {
            "Authorization": f"Bearer {self.get_api_key()}",
            "Content-Type": "application/json"
        }
    
    def get_api_key(self) -> str:
        """Get API key from environment.
        
        Returns:
            API key string
            
        Raises:
            ValueError: If API key not found in environment
        """
        import os
        api_key = os.environ.get(self.model_config.api_key_env)
        if not api_key:
            raise ValueError(f"API key not found in environment: {self.model_config.api_key_env}")
        return api_key


class UnifiedOpenRouterAdapter(ModelAdapter):
    """Adapter that handles all models via OpenRouter's unified interface."""
    
    # Model-specific parameter mappings
    PARAM_MAPPINGS = {
        "openai/gpt-4": {
            "max_tokens": "max_tokens",
            "temperature": "temperature"
        },
        "openai/gpt-3.5-turbo": {
            "max_tokens": "max_tokens",
            "temperature": "temperature"
        },
        "anthropic/claude-3-sonnet-20240229": {
            "max_tokens": "max_tokens",
            "temperature": "temperature"
        },
        "google/gemini-pro": {
            "max_tokens": "max_tokens",
            "temperature": "temperature"
        },
        "google/gemini-2.5-flash": {
            "max_tokens": "max_tokens",
            "temperature": "temperature"
        }
    }
    
    def get_request_params(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Get request parameters for OpenRouter API.
        
        Args:
            prompt: The prompt to send to the model
            **kwargs: Additional parameters (can override defaults)
            
        Returns:
            Dictionary of parameters for OpenRouter API
        """
        # Get parameter mapping for this model
        param_map = self.PARAM_MAPPINGS.get(
            self.model_config.model_type,
            self.PARAM_MAPPINGS["openai/gpt-4"]  # Default mapping
        )
        
        # Build base parameters
        params = {
            "model": self.model_config.model_type,
            "messages": [{"role": "user", "content": prompt}],
        }
        
        # Map model-specific parameters
        if param_map.get("max_tokens"):
            params[param_map["max_tokens"]] = kwargs.get("max_tokens", self.model_config.max_tokens)
        
        if param_map.get("temperature"):
            params[param_map["temperature"]] = kwargs.get("temperature", self.model_config.temperature)
        
        # Add any custom parameters from model config
        for key, value in self.model_config.custom_params.items():
            if key not in params:
                params[key] = value
        
        # Allow kwargs to override any parameter
        for key, value in kwargs.items():
            if key not in ["prompt"]:  # Don't include prompt in params
                params[key] = value
        
        return params
    
    def parse_response(self, response: Dict[str, Any]) -> str:
        """Parse OpenRouter unified response format.
        
        Args:
            response: Raw response from OpenRouter API
            
        Returns:
            Extracted text content from the response
            
        Raises:
            ValueError: If response format is unexpected
        """
        try:
            # OpenRouter unified format
            if "choices" in response and len(response["choices"]) > 0:
                message = response["choices"][0].get("message", {})
                return message.get("content", "")
            else:
                raise ValueError(f"Unexpected response format: {response}")
        except (KeyError, IndexError) as e:
            raise ValueError(f"Failed to parse response: {e}") from e


class ModelAdapterFactory:
    """Factory for creating model adapters."""
    
    _adapters_cache: Dict[str, ModelAdapter] = {}
    
    @classmethod
    def get_adapter(cls, model_config: ModelConfig) -> ModelAdapter:
        """Get adapter for the specified model configuration.
        
        Args:
            model_config: Configuration for the model
            
        Returns:
            Appropriate adapter instance
        """
        # Use cached adapter if available (per model type)
        cache_key = f"{model_config.model_type}_{model_config.api_key_env}"
        
        if cache_key not in cls._adapters_cache:
            # For now, all models use UnifiedOpenRouterAdapter
            # In future, could add model-specific adapters if needed
            cls._adapters_cache[cache_key] = UnifiedOpenRouterAdapter(model_config)
            logger.info(f"Created new adapter for {model_config.model_type}")
        
        return cls._adapters_cache[cache_key]
    
    @classmethod
    def clear_cache(cls):
        """Clear the adapter cache (useful for testing)."""
        cls._adapters_cache.clear()


class FallbackHandler:
    """Handles fallback logic when models fail."""
    
    DEFAULT_MODEL_CONFIG = ModelConfig(
        model_type=ModelType.GEMINI_25_FLASH.value,
        api_key_env="OPENROUTER_API_KEY"
    )
    
    @classmethod
    async def handle_model_failure(
        cls,
        error: Exception,
        model_config: ModelConfig,
        context: Dict[str, Any]
    ) -> Optional[ModelConfig]:
        """Handle model failure with fallback logic.
        
        Args:
            error: The exception that occurred
            model_config: The model configuration that failed
            context: Additional context about the failure
            
        Returns:
            Fallback model configuration, or None if no fallback available
        """
        agent_id = context.get("agent_id", "unknown")
        round_num = context.get("round", "unknown")
        
        logger.warning(
            f"Model failure for agent {agent_id} in round {round_num}: "
            f"Model={model_config.model_type}, Error={str(error)}"
        )
        
        # If already using default model, no fallback available
        if model_config.model_type == cls.DEFAULT_MODEL_CONFIG.model_type:
            logger.error("Default model failed, no fallback available")
            return None
        
        # Fall back to default model
        logger.info(
            f"Falling back to default model {cls.DEFAULT_MODEL_CONFIG.model_type} "
            f"for agent {agent_id}"
        )
        
        return cls.DEFAULT_MODEL_CONFIG