# Available Models for Epic 6: Multi-Model Experiments

## Current Configuration (from codebase)
- **Main Model**: `google/gemini-2.5-flash`
- **Sub Model**: `openai/GPT-4.1-nano` (used for subagent decisions)

## OpenRouter Available Models (2025)

### Major Model Families

#### OpenAI Models
- **GPT-4o** - Dynamic model that updates in real-time
- **GPT-4o-mini** - 60% cheaper than GPT-3.5 Turbo, advanced small model
- **o1** - New reasoning model for complex tasks
- **o1-mini** - Fast, cost-effective reasoning model for programming/math
- **GPT-4** - Original GPT-4
- **GPT-3.5-turbo** - Fast, cost-effective

#### Anthropic Claude Models (Including Claude 4!)
- **Claude 4 Opus** - Most powerful model, best coding (72.5% SWE-bench), $15/$75 per M tokens
- **Claude 4 Sonnet** - Balanced performance (72.7% SWE-bench), $3/$15 per M tokens
- **Claude 3.5 Sonnet** - Surpasses Claude 3 Opus with faster speeds
- **Claude 3.5 Haiku** - Fastest next-generation model
- **Claude 3 Opus** - Handles highly complex tasks
- **Claude 3 Haiku** - Near-instantaneous responses
- *Note: Claude 4 has hybrid modes - instant responses and extended thinking*

#### Google Gemini Models
- **Gemini 2.5 Pro** - Enhanced with Deep Think mode, significant leap in reasoning
- **Gemini 2.0 Flash** - 1M token context, native tool usage, multimodal generation
- **Gemini 1.5 Pro** - Efficient multimodal processing
- **Gemini 1.5 Flash** - Optimized multimodal processing

#### DeepSeek Models (Major 2025 Release!)
- **DeepSeek R1** - 671B params, performance on par with OpenAI o1, MIT licensed
- **DeepSeek V3** - Pre-trained on 15T tokens, rivals leading competitors
- **DeepSeek-R1T-Chimera** - NEW: Merges R1 and V3, 685B params, 40% more efficient
- **DeepSeek V2.5** - Integrates general and coding capabilities
- **DeepSeek Chat** - Multiple variants including free versions
- *Note: DeepSeek models offer competitive performance at dramatically lower costs*

#### Meta Llama Models
- **Llama 3.1** - Various sizes (8B, 70B, 405B) Instruct variants
- **Llama 3** - Previous generation
- **Llama 2** - Original open-source series

#### Mistral Models
- **Mistral Small 3.1** - 24B params, multimodal (Text + Image → Text)
- **Mistral Large** - Most capable
- **Mistral Medium** - Balanced performance
- **Mistral 7B** - Popular open-source

#### OpenRouter In-House Models
- **OpenRouter Optimus Alpha** - General-purpose assistant, low-latency API
- **OpenRouter Quasar Alpha** - Specialized for reasoning and knowledge

#### Other Notable Models
- **Amazon Nova** (Lite, Micro, Pro) - AWS models
- **Cohere Command** - Enterprise-focused

## Recommended Models for Epic 6 Experiments

### Primary Test Models (Different Architectures)
1. **OpenAI**: `openai/gpt-4o` or `openai/o1` (reasoning model)
2. **Anthropic**: `anthropic/claude-4-opus` or `anthropic/claude-4-sonnet`
3. **Google**: `google/gemini-2.5-pro` or `google/gemini-2.0-flash`
4. **DeepSeek**: `deepseek/deepseek-r1` or `deepseek/deepseek-v3`

### Secondary Test Models (Additional Diversity)
5. **Meta**: `meta-llama/llama-3.1-70b-instruct`
6. **Mistral**: `mistralai/mistral-small-3.1` (multimodal)
7. **OpenRouter**: `openrouter/optimus-alpha` (in-house model)

### Budget-Friendly Alternatives
- **OpenAI**: `openai/gpt-4o-mini` or `openai/gpt-3.5-turbo`
- **Anthropic**: `anthropic/claude-3-haiku` or `anthropic/claude-3.5-haiku`
- **Google**: `google/gemini-1.5-flash`
- **DeepSeek**: `deepseek/deepseek-chat` (free tier available)

## Key 2025 Model Updates

### Claude 4 (Released May 2025)
- Only 2 models: Opus 4 and Sonnet 4 (no Haiku 4)
- **Hybrid architecture**: Instant responses + extended thinking modes
- Can use tools (like web search) during extended thinking
- Leading performance on coding benchmarks

### DeepSeek Revolution (January 2025)
- Disrupted the AI industry with open-source models matching Western performance
- **DeepSeek R1**: 671B params, MIT licensed, comparable to OpenAI o1
- **DeepSeek-R1T-Chimera**: NEW hybrid combining R1 + V3, 40% more efficient
- Dramatically lower costs than competitors

### Other Notable Updates
- **Gemini 2.5 Pro**: Now with Deep Think mode for enhanced reasoning
- **GPT-4o**: Dynamic model that auto-updates to latest version
- **OpenRouter In-House**: Optimus Alpha and Quasar Alpha models

## Model Selection Criteria for Experiments

### Key Considerations:
1. **Architecture Diversity**: Different tokenizers, training approaches
2. **Capability Parity**: Similar performance levels for fair comparison
3. **API Availability**: Consistent access through OpenRouter
4. **Cost Efficiency**: Balance between quality and experiment budget
5. **Context Window**: Sufficient for strategy explanations (min 4K tokens)
6. **Special Features**: Consider hybrid modes (Claude 4), reasoning modes (o1), multimodal (Mistral)

### Suggested Experiment Configurations:

#### Homogeneous Groups (Baseline)
- 10x GPT-4
- 10x Claude-3-Sonnet
- 10x Gemini-Pro

#### Balanced Mixed (50/50)
- 5x GPT-4 + 5x Claude-3-Sonnet
- 5x GPT-4 + 5x Gemini-Pro
- 5x Claude-3-Sonnet + 5x Gemini-Pro

#### Diverse Mix (3-3-4)
- 3x GPT-4 + 3x Claude-3-Sonnet + 4x Gemini-Pro

#### Cost-Optimized Mix
- 3x GPT-4 + 3x Claude-3-Haiku + 4x Gemini-Flash

## Implementation Notes

### Model ID Format
OpenRouter uses the format: `provider/model-name`
- OpenAI: `openai/gpt-4`
- Anthropic: `anthropic/claude-3-sonnet`
- Google: `google/gemini-pro`
- Meta: `meta-llama/llama-3.1-70b`

### API Considerations
- **Rate Limits**: Vary by model (typically 20-60 requests/minute)
- **Context Windows**: Range from 4K to 200K tokens
- **Pricing**: Varies significantly (check OpenRouter pricing page)
- **Availability**: Some models may have limited availability

### Testing Priority
1. Start with major providers (OpenAI, Anthropic, Google)
2. Ensure API keys are configured for each provider
3. Test with small groups first (3-5 agents)
4. Scale up to full 10-agent experiments

## Next Steps for Epic 6
1. Configure API adapters for each model family
2. Implement model-specific parameter tuning
3. Create test scenarios with different model combinations
4. Develop analysis framework for cross-model cooperation
5. Set up statistical testing for significance analysis