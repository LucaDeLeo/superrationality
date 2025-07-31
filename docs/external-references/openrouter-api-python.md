# OpenRouter API Documentation - Python Integration

## Overview

OpenRouter provides a unified API for hundreds of AI models and supports integration using OpenAI SDK, direct API calls, or third-party frameworks. OpenRouter normalizes the schema across models and providers to comply with the OpenAI Chat API, so you only need to learn one.

## Installation

Since OpenRouter is compatible with the OpenAI API, you can use the official OpenAI Python SDK:

```bash
pip install openai
```

## Basic Setup

### Using OpenAI SDK

```python
import os
from openai import OpenAI

client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)
```

### With Optional Headers

```python
client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    default_headers={
        "HTTP-Referer": "https://your-app.com/",  # Optional, for your app's URL
        "X-Title": "Your App Name"  # Optional, shows in provider dashboards
    }
)
```

## Basic Usage

### Simple Chat Completion

```python
response = client.chat.completions.create(
    model="google/gemini-2.0-flash-exp:free",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"}
    ]
)

print(response.choices[0].message.content)
```

### Streaming Responses

OpenRouter supports streaming for all models with Server-Sent Events (SSE):

```python
stream = client.chat.completions.create(
    model="google/gemini-2.0-flash-exp:free",
    messages=[{"role": "user", "content": "Count to 5 slowly"}],
    stream=True
)

for chunk in stream:
    if chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end="")
```

## Model Selection

### Model Format
Models must include the organization prefix:
- `openai/gpt-4`
- `google/gemini-2.0-flash-exp:free`
- `anthropic/claude-3-opus-20240229`

### Get Available Models

```python
import requests

headers = {
    "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}"
}

response = requests.get(
    "https://openrouter.ai/api/v1/models",
    headers=headers
)

models = response.json()
```

## Advanced Features

### Structured Outputs with Instructor

For type-safe, validated responses:

```python
from pydantic import BaseModel
import instructor
from openai import OpenAI

# Initialize client with instructor
client = instructor.from_openai(
    OpenAI(
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1"
    )
)

# Define your model
class User(BaseModel):
    name: str
    age: int
    email: str

# Create structured output
user = client.chat.completions.create(
    model="google/gemini-2.0-flash-exp:free",
    messages=[{"role": "user", "content": "Extract: Jason is 25 years old, email: jason@example.com"}],
    response_model=User
)

print(user)  # User(name='Jason', age=25, email='jason@example.com')
```

### Using Async Client

```python
import asyncio
from openai import AsyncOpenAI

async_client = AsyncOpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)

async def get_response():
    response = await async_client.chat.completions.create(
        model="google/gemini-2.0-flash-exp:free",
        messages=[{"role": "user", "content": "Hello!"}]
    )
    return response.choices[0].message.content

# Run async function
result = asyncio.run(get_response())
```

## Error Handling

```python
try:
    response = client.chat.completions.create(
        model="google/gemini-2.0-flash-exp:free",
        messages=[{"role": "user", "content": "Hello"}]
    )
except Exception as e:
    print(f"Error: {e}")
    # Handle rate limits, API errors, etc.
```

## Cost Management

### Setting Cost Limits

```python
response = client.chat.completions.create(
    model="google/gemini-2.0-flash-exp:free",
    messages=[{"role": "user", "content": "Hello"}],
    extra_body={
        "transforms": ["middle-out"],
        "route": "fallback",
        "max_cost": 0.01  # Maximum cost in USD
    }
)
```

## Authentication Best Practices

1. **Never hardcode API keys** - Use environment variables
2. **Protect your API keys** - Never commit them to public repositories
3. **Use different keys for different environments** - Development vs Production

```python
# Good practice
api_key = os.getenv("OPENROUTER_API_KEY")
if not api_key:
    raise ValueError("OPENROUTER_API_KEY environment variable is required")

# Bad practice - Never do this!
# api_key = "sk-or-v1-xxxxx"  # NEVER hardcode keys
```

## Common Parameters

- `model`: The model to use (required unless default is set)
- `messages`: Array of message objects
- `temperature`: Controls randomness (0-2)
- `max_tokens`: Maximum tokens to generate
- `stream`: Enable streaming responses
- `top_p`: Nucleus sampling parameter
- `frequency_penalty`: Penalize repeated tokens
- `presence_penalty`: Penalize tokens based on presence

## Example: Complete Integration

```python
import os
from openai import OpenAI
from typing import List, Dict

class OpenRouterClient:
    def __init__(self):
        self.client = OpenAI(
            api_key=os.getenv("OPENROUTER_API_KEY"),
            base_url="https://openrouter.ai/api/v1"
        )
    
    def chat(self, messages: List[Dict[str, str]], model: str = "google/gemini-2.0-flash-exp:free"):
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.7,
                max_tokens=1000
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error: {str(e)}"
    
    def stream_chat(self, messages: List[Dict[str, str]], model: str = "google/gemini-2.0-flash-exp:free"):
        stream = self.client.chat.completions.create(
            model=model,
            messages=messages,
            stream=True
        )
        
        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

# Usage
client = OpenRouterClient()
response = client.chat([
    {"role": "user", "content": "What is OpenRouter?"}
])
print(response)
```

## Resources

- **API Reference**: https://openrouter.ai/docs/api-reference/overview
- **Quickstart Guide**: https://openrouter.ai/docs/quickstart
- **OpenAI SDK Integration**: https://openrouter.ai/docs/community/open-ai-sdk
- **GitHub Examples**: https://github.com/OpenRouterTeam/openrouter-examples
- **Available Models**: https://openrouter.ai/models