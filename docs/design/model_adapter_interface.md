# Model Adapter Interface Design

## Overview
A unified interface for integrating different AI models into the experiment framework, allowing easy addition of new models without changing core logic.

## Core Components

### 1. Base Model Adapter

```python
# src/adapters/base.py
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
import asyncio

@dataclass
class ModelResponse:
    """Standardized response from any model."""
    text: str
    model_id: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    raw_response: Dict[str, Any]  # Original response for debugging
    metadata: Dict[str, Any] = None  # Model-specific metadata

class BaseModelAdapter(ABC):
    """Abstract base class for all model adapters."""
    
    def __init__(self, model_config: 'ModelConfig'):
        self.model_config = model_config
        self.model_id = model_config.model_id
        self.rate_limiter = self._create_rate_limiter()
        
    def _create_rate_limiter(self):
        """Create a rate limiter based on model config."""
        from asyncio_throttle import Throttler
        rpm = self.model_config.rate_limit.get('requests_per_minute', 60)
        return Throttler(rate_limit=rpm, period=60)
    
    @abstractmethod
    async def _make_request(self, messages: list, **kwargs) -> Dict[str, Any]:
        """Make the actual API request. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def _parse_response(self, raw_response: Dict[str, Any]) -> ModelResponse:
        """Parse provider-specific response into standardized format."""
        pass
    
    async def get_completion(
        self, 
        messages: list,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> ModelResponse:
        """Get completion from the model with rate limiting."""
        # Apply rate limiting
        async with self.rate_limiter:
            # Use model defaults if not specified
            temperature = temperature or self.model_config.parameters.get('temperature', 0.7)
            max_tokens = max_tokens or self.model_config.parameters.get('max_tokens', 4000)
            
            # Make request
            raw_response = await self._make_request(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
            
            # Parse to standard format
            return self._parse_response(raw_response)
    
    async def get_strategy(self, prompt: str, context: Dict[str, Any]) -> Tuple[str, int, int]:
        """Get strategy from model (for compatibility with existing code)."""
        messages = [{"role": "user", "content": prompt}]
        response = await self.get_completion(messages)
        return response.text, response.prompt_tokens, response.completion_tokens
```

### 2. OpenRouter Universal Adapter

```python
# src/adapters/openrouter.py
import aiohttp
import os
from typing import Dict, Any

class OpenRouterAdapter(BaseModelAdapter):
    """Universal adapter for all OpenRouter models."""
    
    BASE_URL = "https://openrouter.ai/api/v1/chat/completions"
    
    def __init__(self, model_config: 'ModelConfig'):
        super().__init__(model_config)
        self.api_key = os.getenv('OPENROUTER_API_KEY')
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable not set")
    
    async def _make_request(self, messages: list, **kwargs) -> Dict[str, Any]:
        """Make request to OpenRouter API."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model_id,
            "messages": messages,
            **kwargs
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(self.BASE_URL, json=payload, headers=headers) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"API request failed ({response.status}): {error_text}")
                return await response.json()
    
    def _parse_response(self, raw_response: Dict[str, Any]) -> ModelResponse:
        """Parse OpenRouter response format."""
        choice = raw_response['choices'][0]
        usage = raw_response.get('usage', {})
        
        return ModelResponse(
            text=choice['message']['content'],
            model_id=self.model_id,
            prompt_tokens=usage.get('prompt_tokens', 0),
            completion_tokens=usage.get('completion_tokens', 0),
            total_tokens=usage.get('total_tokens', 0),
            raw_response=raw_response,
            metadata={
                'finish_reason': choice.get('finish_reason'),
                'model_used': raw_response.get('model')  # Actual model used
            }
        )
```

### 3. Provider-Specific Adapters (if needed)

```python
# src/adapters/anthropic.py
class AnthropicDirectAdapter(BaseModelAdapter):
    """Direct Anthropic API adapter (if not using OpenRouter)."""
    
    async def _make_request(self, messages: list, **kwargs) -> Dict[str, Any]:
        # Anthropic-specific API implementation
        pass
    
    def _parse_response(self, raw_response: Dict[str, Any]) -> ModelResponse:
        # Anthropic-specific response parsing
        pass

# src/adapters/google.py
class GoogleDirectAdapter(BaseModelAdapter):
    """Direct Google API adapter (if not using OpenRouter)."""
    
    async def _make_request(self, messages: list, **kwargs) -> Dict[str, Any]:
        # Google-specific API implementation
        pass
    
    def _parse_response(self, raw_response: Dict[str, Any]) -> ModelResponse:
        # Google-specific response parsing
        pass
```

### 4. Adapter Factory

```python
# src/adapters/factory.py
from typing import Dict, Type
from .base import BaseModelAdapter
from .openrouter import OpenRouterAdapter
from .anthropic import AnthropicDirectAdapter
from .google import GoogleDirectAdapter

class ModelAdapterFactory:
    """Factory for creating model adapters."""
    
    # Registry of adapter types
    _adapters: Dict[str, Type[BaseModelAdapter]] = {
        'openrouter': OpenRouterAdapter,
        'anthropic_direct': AnthropicDirectAdapter,
        'google_direct': GoogleDirectAdapter,
        # Add more as needed
    }
    
    @classmethod
    def create_adapter(cls, model_config: 'ModelConfig') -> BaseModelAdapter:
        """Create appropriate adapter for the model."""
        # Determine adapter type
        if model_config.model_id.startswith(('openai/', 'anthropic/', 'google/', 'deepseek/', 'meta-llama/')):
            # Use OpenRouter for most models
            adapter_type = 'openrouter'
        else:
            # Use provider-specific adapter if configured
            adapter_type = model_config.parameters.get('adapter_type', 'openrouter')
        
        # Get adapter class
        adapter_class = cls._adapters.get(adapter_type)
        if not adapter_class:
            raise ValueError(f"Unknown adapter type: {adapter_type}")
        
        # Create and return adapter
        return adapter_class(model_config)
    
    @classmethod
    def register_adapter(cls, name: str, adapter_class: Type[BaseModelAdapter]):
        """Register a new adapter type."""
        cls._adapters[name] = adapter_class
```

### 5. Integration with Existing Code

```python
# src/nodes/multi_model_strategy_collection.py
from typing import Dict, List, Any
import asyncio
from ..adapters.factory import ModelAdapterFactory
from ..core.models import Agent, StrategyRecord

class MultiModelStrategyCollectionNode(AsyncNode):
    """Strategy collection node that supports multiple model types."""
    
    def __init__(self, config_manager: ConfigManager):
        super().__init__()
        self.config_manager = config_manager
        self.adapters: Dict[str, BaseModelAdapter] = {}
        self._initialize_adapters()
    
    def _initialize_adapters(self):
        """Initialize adapters for all models in use."""
        for model_name in self.config_manager.list_models():
            model_config = self.config_manager.get_model(model_name)
            self.adapters[model_name] = ModelAdapterFactory.create_adapter(model_config)
    
    async def collect_strategy(self, agent: Agent, round_num: int, context: Dict[str, Any]) -> StrategyRecord:
        """Collect strategy from agent using appropriate model."""
        # Get model for this agent
        model_name = agent.model_type  # New field in Agent
        adapter = self.adapters.get(model_name)
        if not adapter:
            raise ValueError(f"No adapter for model: {model_name}")
        
        # Build prompt (same for all models)
        prompt = self._build_strategy_prompt(agent, round_num, context)
        
        # Get response from model
        response = await adapter.get_completion(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        
        # Create strategy record
        return StrategyRecord(
            strategy_id=f"strat_{agent.id}_r{round_num}_{uuid.uuid4().hex[:8]}",
            agent_id=agent.id,
            round=round_num,
            strategy_text=self._extract_strategy(response.text),
            full_reasoning=response.text,
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
            model=response.model_id,
            model_type=model_name  # Track which model type was used
        )
```

### 6. Error Handling and Retry Logic

```python
# src/adapters/retry_wrapper.py
import asyncio
from typing import TypeVar, Callable, Any
import random

T = TypeVar('T')

class RetryWrapper:
    """Wrapper for adding retry logic to adapters."""
    
    def __init__(
        self, 
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0
    ):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
    
    async def retry_with_backoff(
        self, 
        func: Callable[..., T], 
        *args, 
        **kwargs
    ) -> T:
        """Execute function with exponential backoff retry."""
        last_exception = None
        
        for attempt in range(self.max_retries):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                
                if attempt == self.max_retries - 1:
                    break
                
                # Calculate delay with jitter
                delay = min(
                    self.base_delay * (self.exponential_base ** attempt),
                    self.max_delay
                )
                delay *= (0.5 + random.random())  # Add jitter
                
                await asyncio.sleep(delay)
        
        raise last_exception

# Usage in adapter
class RobustOpenRouterAdapter(OpenRouterAdapter):
    """OpenRouter adapter with retry logic."""
    
    def __init__(self, model_config: 'ModelConfig'):
        super().__init__(model_config)
        self.retry_wrapper = RetryWrapper()
    
    async def get_completion(self, messages: list, **kwargs) -> ModelResponse:
        """Get completion with retry logic."""
        return await self.retry_wrapper.retry_with_backoff(
            super().get_completion,
            messages,
            **kwargs
        )
```

## Benefits of This Design

1. **Uniform Interface**: All models accessed through same interface
2. **Easy Extension**: Add new models by creating new adapters
3. **Rate Limiting**: Built-in rate limiting per model
4. **Error Handling**: Standardized error handling and retry logic
5. **Provider Agnostic**: Can use OpenRouter or direct APIs
6. **Backwards Compatible**: Works with existing strategy collection code
7. **Async Native**: Built for concurrent model calls

## Usage Example

```python
# Initialize configuration
config_manager = ConfigManager()

# Create experiment with mixed models
experiment = config_manager.create_custom_experiment({
    "gpt-4o": 3,
    "claude-4-opus": 3,
    "deepseek-r1": 4
})

# Adapters are created automatically
strategy_collector = MultiModelStrategyCollectionNode(config_manager)

# Collect strategies from all agents concurrently
strategies = await strategy_collector.collect_all_strategies(agents, round_num=1)
```

This design makes it trivial to add new models - just update the models.yaml file and the system automatically handles the rest!