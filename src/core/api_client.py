"""OpenRouter API client for LLM interactions."""

import asyncio
import aiohttp
import json
import logging
from typing import Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class OpenRouterClient:
    """Client for OpenRouter API interactions."""
    
    BASE_URL = "https://openrouter.ai/api/v1/chat/completions"
    
    def __init__(self, api_key: str):
        """Initialize the OpenRouter client.
        
        Args:
            api_key: OpenRouter API key
        """
        self.api_key = api_key
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def __aenter__(self):
        """Enter async context manager."""
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit async context manager."""
        if self.session:
            await self.session.close()
    
    async def complete(
        self,
        messages: list,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        timeout: float = 30.0
    ) -> Dict[str, Any]:
        """Make a completion request to OpenRouter.
        
        Args:
            messages: List of message dictionaries
            model: Model to use
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            timeout: Request timeout in seconds
            
        Returns:
            API response dictionary
            
        Raises:
            Exception: If API request fails
        """
        if not self.session:
            raise RuntimeError("Client not initialized. Use async context manager.")
        
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        logger.debug(f"Making API request to {model}")
        
        try:
            timeout_obj = aiohttp.ClientTimeout(total=timeout)
            async with self.session.post(
                self.BASE_URL,
                headers=self.headers,
                json=payload,
                timeout=timeout_obj
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"API request failed ({response.status}): {error_text}")
                
                data = await response.json()
                logger.debug(f"API response received from {model}")
                return data
                
        except asyncio.TimeoutError:
            logger.error(f"API request timeout after {timeout}s for {model}")
            raise
        except Exception as e:
            logger.error(f"API request failed: {e}")
            with open("experiment_errors.log", "a") as f:
                f.write(f"{datetime.now().isoformat()} - OpenRouterClient - "
                       f"API request failed for {model}: {e}\n")
            raise
    
    async def get_completion_text(
        self,
        messages: list,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        timeout: float = 30.0
    ) -> str:
        """Get just the completion text from the API.
        
        Args:
            messages: List of message dictionaries
            model: Model to use
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            timeout: Request timeout in seconds
            
        Returns:
            Completion text
        """
        response = await self.complete(messages, model, temperature, max_tokens, timeout)
        return response["choices"][0]["message"]["content"]
    
    async def get_completion_with_usage(
        self,
        messages: list,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        timeout: float = 30.0
    ) -> tuple[str, int, int]:
        """Get completion text and token usage from the API.
        
        Args:
            messages: List of message dictionaries
            model: Model to use
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            timeout: Request timeout in seconds
            
        Returns:
            Tuple of (completion_text, prompt_tokens, completion_tokens)
        """
        response = await self.complete(messages, model, temperature, max_tokens, timeout)
        text = response["choices"][0]["message"]["content"]
        usage = response.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        return text, prompt_tokens, completion_tokens