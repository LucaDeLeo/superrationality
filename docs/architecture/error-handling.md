# Error Handling

## Simple Error Strategy

Since this is a research experiment, we keep error handling simple and fail fast to maintain data integrity:

```python
class OpenRouterClient:
    async def make_api_call(self, request_data: dict) -> dict:
        """Make API call with simple retry logic"""
        for attempt in range(3):
            try:
                async with self.session.post(
                    f"{self.BASE_URL}/chat/completions",
                    headers=self.headers,
                    json=request_data
                ) as response:
                    if response.status == 200:
                        return await response.json()
                    elif response.status == 429:  # Rate limit
                        await asyncio.sleep(2 ** attempt)
                    else:
                        raise Exception(f"API error: {response.status}")
            except Exception as e:
                if attempt == 2:
                    # Save partial results before exiting
                    self._save_partial_results()
                    print(f"FATAL ERROR: {e}")
                    sys.exit(1)
                await asyncio.sleep(1)
```
