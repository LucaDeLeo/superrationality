# API Specification

## OpenRouter API Integration

Since this experiment consumes external LLM APIs rather than exposing its own API, this section documents the OpenRouter API integration patterns.

### API Client Configuration
```python
class OpenRouterClient:
    BASE_URL = "https://openrouter.ai/api/v1"
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.session = aiohttp.ClientSession()
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/acausal-experiment",
            "X-Title": "Acausal Cooperation Experiment"
        }
```

### Strategy Collection Request
```python