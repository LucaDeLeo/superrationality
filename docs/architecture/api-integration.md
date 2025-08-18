# API Integration

## OpenRouter Integration

```python
class OpenRouterClient:
    BASE_URL = "https://openrouter.ai/api/v1/chat/completions"
    
    async def complete(self, messages: list, model: str, **kwargs):
        """Make completion request to OpenRouter"""
        payload = {
            "model": model,
            "messages": messages,
            "temperature": kwargs.get("temperature", 0.7),
            "max_tokens": kwargs.get("max_tokens", 1000)
        }
        
        async with self.session.post(
            self.BASE_URL,
            headers=self.headers,
            json=payload
        ) as response:
            return await response.json()
```

## Rate Limiting Strategy

- Simple token bucket algorithm
- 60 requests per minute limit
- Exponential backoff on 429 errors
- Parallel requests for strategy collection
- Sequential requests for game decisions
