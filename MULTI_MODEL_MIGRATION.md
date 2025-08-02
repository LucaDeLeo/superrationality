# Multi-Model Configuration Migration Guide

⚠️ **WARNING: This is an experimental feature. Use at your own risk.** ⚠️

This guide helps you migrate from single-model experiments to multi-model experiments that test acausal cooperation across different AI architectures.

## Table of Contents
1. [Overview](#overview)
2. [Backward Compatibility](#backward-compatibility)
3. [Enabling Multi-Model Support](#enabling-multi-model-support)
4. [Configuration Examples](#configuration-examples)
5. [Gradual Migration Strategy](#gradual-migration-strategy)
6. [Troubleshooting](#troubleshooting)
7. [Rollback Procedures](#rollback-procedures)

## Overview

The multi-model feature allows you to run experiments where different agents use different AI models (GPT-4, Claude, Gemini, etc.). This helps test whether acausal cooperation patterns are consistent across model architectures.

### Key Features
- **Feature Flag Protection**: All functionality is behind `ENABLE_MULTI_MODEL` flag
- **Zero Breaking Changes**: Existing experiments work unchanged
- **Graceful Fallback**: Failed models automatically fall back to default
- **Per-Model Configuration**: Customize parameters for each model type

## Backward Compatibility

**No action required for existing users!** Your experiments will continue to work exactly as before.

The system maintains full backward compatibility:
- Default behavior unchanged when `ENABLE_MULTI_MODEL: false` (default)
- Existing YAML configs load without modification
- All current API calls work identically
- Test suites pass without changes

## Enabling Multi-Model Support

### Step 1: Update Your Configuration

Add the following to your YAML configuration:

```yaml
# Enable the experimental multi-model feature
ENABLE_MULTI_MODEL: true

# Define available models
model_configs:
  openai/gpt-4:
    model_type: openai/gpt-4
    api_key_env: OPENROUTER_API_KEY  # Uses default key
    max_tokens: 1500
    temperature: 0.7
    rate_limit: 60

# Define scenarios (model distributions)
scenarios:
  - name: My First Multi-Model Experiment
    model_distribution:
      openai/gpt-4: 10  # All agents use GPT-4
```

### Step 2: Run Your Experiment

No code changes needed! Just run as usual:

```bash
python run_experiment.py --config your_config.yaml
```

Or specify a scenario:

```bash
python run_experiment.py --config your_config.yaml --scenario "My First Multi-Model Experiment"
```

## Configuration Examples

### Example 1: Homogeneous Group (All Same Model)

See `configs/examples/multi_model/homogeneous_gpt4.yaml`

This is the safest way to start - all agents use the same model, just not the default one.

### Example 2: Mixed Models (50/50 Split)

See `configs/examples/multi_model/mixed_5_5.yaml`

Split agents between two models to compare their cooperation patterns.

### Example 3: Diverse Models (Multiple Types)

See `configs/examples/multi_model/diverse_3_3_4.yaml`

Test with three or more different model types for comprehensive analysis.

## Gradual Migration Strategy

We recommend a phased approach:

### Phase 1: Test with Feature Disabled
1. Add `ENABLE_MULTI_MODEL: false` to your config
2. Run your normal experiments
3. Verify everything works as before

### Phase 2: Single Agent Different Model
1. Set `ENABLE_MULTI_MODEL: true`
2. Configure 9 agents with default model, 1 with different model
3. Run experiment and check logs

### Phase 3: Homogeneous Non-Default
1. Configure all agents to use a non-default model
2. Compare results with your baseline
3. Check `experiment_errors.log` for issues

### Phase 4: Mixed Models
1. Start with 50/50 split between two models
2. Monitor for behavioral differences
3. Check model assignment logs

### Phase 5: Full Diversity
1. Use 3+ different models
2. Analyze cross-model cooperation patterns
3. Generate comparative reports

## Troubleshooting

### Common Issues

**Issue**: "Model not found" errors
- **Solution**: Check model identifier matches OpenRouter's format exactly
- **Example**: Use `anthropic/claude-3-sonnet-20240229` not `claude-3-sonnet`

**Issue**: Rate limiting errors
- **Solution**: Adjust `rate_limit` in model config
- **Default**: 60 requests/minute

**Issue**: Experiment fails to start
- **Solution**: Check that model distribution sums to NUM_AGENTS
- **Example**: With NUM_AGENTS=10, distributions must total 10

### Debugging

1. **Check Logs**:
   - `experiment_errors.log` - Model-specific failures
   - `results/{id}/model_assignments.json` - Which agent got which model

2. **Verify API Keys**:
   ```bash
   echo $OPENROUTER_API_KEY  # Should not be empty
   ```

3. **Test Individual Models**:
   ```python
   # Quick test script
   from src.core.api_client import OpenRouterClient
   import asyncio
   
   async def test():
       async with OpenRouterClient(api_key) as client:
           response = await client.complete(
               messages=[{"role": "user", "content": "Hi"}],
               model="openai/gpt-4"
           )
           print(response)
   
   asyncio.run(test())
   ```

## Rollback Procedures

If you encounter issues, you can disable multi-model support instantly:

### Quick Disable
1. Set `ENABLE_MULTI_MODEL: false` in your config
2. Remove or comment out `model_configs` and `scenarios` sections
3. Run experiment normally

### Full Rollback
If needed, you can completely remove multi-model code:

1. **Restore Original Files**:
   ```bash
   git checkout src/core/models.py
   git checkout src/core/config.py
   git checkout src/flows/experiment.py
   git checkout src/core/api_client.py
   git checkout src/nodes/strategy_collection.py
   rm src/core/model_adapters.py
   rm test_multi_model.py
   ```

2. **Remove Config Examples**:
   ```bash
   rm -rf configs/examples/multi_model/
   ```

3. **Restore .env.example**:
   ```bash
   git checkout .env.example
   ```

## Important Notes

1. **Experimental Status**: This feature is experimental and may change
2. **Performance**: Different models have different response times
3. **Costs**: Some models are more expensive than others
4. **Behavior**: Models may exhibit different cooperation patterns
5. **Logging**: Always check logs for model-specific issues

## Support

For issues or questions:
1. Check `experiment_errors.log` for detailed error messages
2. Review example configurations in `configs/examples/multi_model/`
3. Ensure your OpenRouter API key has access to desired models
4. Verify model identifiers match OpenRouter's documentation

Remember: When in doubt, disable the feature and run with default settings!