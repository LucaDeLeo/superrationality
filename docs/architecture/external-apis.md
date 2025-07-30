# External APIs

## OpenRouter API

- **Purpose:** Provides unified access to multiple LLM models (Gemini 2.5 Flash and GPT-4.1 Nano)
- **Documentation:** https://openrouter.ai/docs
- **Base URL(s):** https://openrouter.ai/api/v1
- **Authentication:** Bearer token (API key in Authorization header)
- **Rate Limits:** Varies by model - typically 60 requests/minute for free tier

**Key Endpoints Used:**
- `POST /chat/completions` - Generate LLM completions for strategies and decisions

**Integration Notes:** 
- All API calls go through centralized OpenRouterClient to handle authentication and retries
- Experiment halts immediately on API failures to maintain data integrity
- Cost tracking implemented to monitor API usage (~$5 per complete experiment)

No other external APIs are required for this experiment.
