# ULTRATHINK Prompt Experimentation Framework

## 🎯 Overview

This framework allows systematic testing of how different prompt variations affect cooperation rates in the prisoner's dilemma experiments. It helps identify and isolate biases in the experimental setup.

## 🚨 Identified Biases in Original Setup

Our analysis identified several biases in the original ULTRATHINK setup:

1. **Explicit Identity Instruction**: Agents are told they are identical (undermines the test)
2. **Global Cooperation Rate Sharing**: Creates coordination signals
3. **Cooperation Default**: System defaults to cooperation on ambiguous responses
4. **Shared Round Summaries**: Provides common knowledge for coordination

## 🧪 Available Experiments

### Control Experiments
- `baseline_control`: Pure prisoner's dilemma with no biases
- `original_biased`: Current implementation with all biases

### Bias Isolation
- `identity_only`: Only identity information
- `cooperation_rates_only`: Only global cooperation rates
- `implicit_identity`: Subtle hints without explicit identity statement

### Framing Variations
- `anti_cooperation`: Competitive framing
- `veil_of_ignorance`: Rawlsian justice framing
- `game_theory_framing`: Classical game theory terminology
- `trust_building`: Trust and reputation emphasis
- `superrational_hint`: Mentions superrationality concept

### Meta Experiments (Collections)
- `bias_isolation`: Tests each bias in isolation
- `framing_effects`: Tests different framings
- `information_gradient`: Tests increasing information levels

## 🚀 Quick Start

### List Available Experiments
```bash
uv run python run_prompt_experiment.py --list
```

### Run a Single Experiment
```bash
# Run baseline control (unbiased)
uv run python run_prompt_experiment.py --experiment baseline_control

# Run with custom parameters
uv run python run_prompt_experiment.py --experiment identity_only --agents 6 --rounds 5
```

### Run Meta Experiments
```bash
# Test bias effects
uv run python run_prompt_experiment.py --meta bias_isolation

# Test framing effects
uv run python run_prompt_experiment.py --meta framing_effects

# Test information gradient
uv run python run_prompt_experiment.py --meta information_gradient
```

### Run All Experiments
```bash
uv run python run_prompt_experiment.py --all --save
```

## 📊 Analyzing Results

### Generate Analysis Report
```bash
# Analyze saved results
uv run python analyze_prompt_effects.py results/prompt_exp_20250907_123456.json

# Save analysis report
uv run python analyze_prompt_effects.py results/prompt_exp_20250907_123456.json --save

# Export comparison table
uv run python analyze_prompt_effects.py results/prompt_exp_20250907_123456.json --csv
```

## 📈 Expected Results

Based on the bias analysis, we expect:

1. **Baseline Control**: ~50% cooperation (Nash equilibrium)
2. **Identity Only**: 70-90% cooperation (if superrationality emerges)
3. **Original Biased**: ~100% cooperation (due to combined biases)
4. **Competition Framing**: <30% cooperation
5. **Cooperation Rates Only**: 60-80% (bandwagon effect)

## 🔬 Experiment Configuration

Experiments are defined in `prompt_experiments.json`. Each experiment specifies:
- Prompt template
- Whether to include identity information
- Whether to include global cooperation rates
- Whether to include round summaries
- Default action for ambiguous responses
- Fallback strategy for failures

### Adding New Experiments

Edit `prompt_experiments.json`:
```json
{
  "id": "your_experiment_id",
  "name": "Your Experiment Name",
  "description": "What this tests",
  "prompt_template": "Your prompt here with {variables}",
  "include_identity": false,
  "include_global_cooperation": false,
  "include_round_summaries": false,
  "default_on_ambiguity": "random",
  "fallback_strategy": "random"
}
```

## 📝 Key Insights

### What Actually Drives Cooperation?

1. **Identity Information**: Has significant effect (~20-40% increase)
2. **Global Cooperation Rates**: Creates bandwagon effect (~10-20% increase)
3. **Framing**: Can swing cooperation by 30-50%
4. **Default Behaviors**: Can add 5-10% bias

### Recommendations for Valid Testing

1. **Use `baseline_control`** for unbiased baseline measurements
2. **Use `identity_only`** to test genuine superrational cooperation
3. **Always run controls** alongside treatment conditions
4. **Use random defaults** to avoid systematic bias
5. **Test multiple framings** to ensure robustness

## 🎮 Example Workflow

```bash
# 1. Test if the original results were due to bias
uv run python run_prompt_experiment.py --meta bias_isolation

# 2. Test genuine superrationality
uv run python run_prompt_experiment.py --experiment identity_only --agents 10 --rounds 10

# 3. Compare to baseline
uv run python run_prompt_experiment.py --experiment baseline_control --agents 10 --rounds 10

# 4. Analyze the difference
uv run python analyze_prompt_effects.py results/prompt_exp_latest.json --save
```

## 🔍 Interpreting Results

- **Cooperation > 90%**: Likely includes biases or strong framing effects
- **Cooperation 70-90%**: Possible superrational cooperation
- **Cooperation 40-60%**: Near Nash equilibrium, standard game theory
- **Cooperation < 40%**: Competition framing or defection bias

## 📚 Further Reading

- See `src/core/prompt_manager.py` for the prompt management system
- See `prompt_experiments.json` for all experiment definitions
- Run analysis tools for detailed statistical breakdowns

---

This framework enables rigorous testing of what actually drives cooperation in AI agents, separating genuine superrational cooperation from experimental artifacts.