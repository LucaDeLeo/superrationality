# ULTRATHINK: Acausal Cooperation Experiment Framework

An advanced experimental framework for testing superrational cooperation between AI agents in prisoner's dilemma scenarios, now with comprehensive multi-model testing capabilities.

## 🎯 Overview

ULTRATHINK tests whether AI agents can achieve **superrational cooperation** through recognition of their shared decision process. The framework now supports testing across 15+ different AI models to compare cooperation patterns between different AI systems.

### Key Hypothesis
When AI agents recognize they are functionally identical or similar, they should cooperate at rates exceeding Nash equilibrium predictions (approaching 70-90% instead of the typical ~50%).

**⚠️ Note**: Initial results showing 100% cooperation were found to be artifacts of experimental biases. Use the prompt experimentation framework for properly controlled tests.

## ✨ Key Features

- **Multi-Agent Tournament System**: Round-robin prisoner's dilemma tournaments with power dynamics
- **Prompt Experimentation Framework** 🆕: Test different prompt conditions to isolate biases
- **Multi-Model Testing**: Compare cooperation across 15+ different AI models
- **Scenario Management**: 28 pre-configured test scenarios
- **Result Caching**: Automatic caching to avoid re-running expensive experiments
- **Comprehensive Analysis**: Pattern detection, convergence analysis, model comparison
- **Bias Isolation Tools** 🆕: Identify and measure experimental artifacts
- **Batch Experiments**: Run multiple scenarios automatically with comparison reports

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/LucaDeLeo/superrationality.git
cd superrationality

# Install uv (recommended) or use pip
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env and add your OPENROUTER_API_KEY
```

### Running Your First Experiment

```bash
# Run a simple 2-agent test
uv run python run_minimal_experiment.py

# Run default 10-agent experiment
uv run python run_experiment.py

# Run with rate limiting (for free tier)
uv run python run_experiment_with_rate_limit.py
```

## 🔬 Experimental Systems

### 1️⃣ Prompt Experimentation Framework

**Important**: Our analysis revealed that the original setup contains several biases that artificially inflate cooperation rates. Use the prompt experimentation system to run unbiased tests.

```bash
# Run unbiased baseline (expected: ~50% cooperation)
uv run python run_prompt_experiment.py --experiment baseline_control

# Test genuine superrationality (expected: 70-90% if real)
uv run python run_prompt_experiment.py --experiment identity_only

# Compare all bias effects
uv run python run_prompt_experiment.py --meta bias_isolation
```

[See PROMPT_EXPERIMENTS.md for full documentation](PROMPT_EXPERIMENTS.md)

### 2️⃣ AISES-Aligned Graduated Difficulty Tests (New!)

Based on academic reviewer feedback, we've implemented a rigorous graduated difficulty approach:

```bash
# Phase 1: One-shot games (maximally unfriendly to cooperation)
uv run python run_aises_experiments.py --phase phase_1

# Test specific conditions
uv run python run_aises_experiments.py --experiment oneshot_identical_info

# Run complete graduated study
uv run python run_aises_experiments.py --all --save
```

**Experimental Progression:**
- **Phase 1**: One-shot games (no reciprocity possible)
- **Phase 2**: Finite games with known endpoint (tests backward induction)
- **Phase 3**: Uncertain length games (enables Folk theorem cooperation)
- **Phase 4**: Complex dynamics (power adjustments)

[See AISES_RESPONSE.md for methodology](AISES_RESPONSE.md)

## 🧪 Multi-Model Experiments

### List Available Scenarios

```bash
# See all 28 pre-configured scenarios
uv run python list_scenarios.py
```

### Run Model Comparisons

```bash
# Quick test with 2 agents (Gemini vs GPT-4)
uv run python run_model_comparison.py --test

# Run specific scenarios
uv run python run_model_comparison.py --scenarios mixed_opus_vs_gpt4turbo chaos_maximum_diversity

# Run all homogeneous (single-model) scenarios
uv run python run_model_comparison.py --scenarios homogeneous_gpt4o homogeneous_claude_opus homogeneous_gemini_pro

# Analyze results
uv run python analyze_models.py --latest --save
```

## 💾 Result Caching (New!)

ULTRATHINK now includes intelligent result caching to save time and money on API calls. Experiments are automatically cached and reused when you run the same configuration again.

### How Caching Works

- **Automatic**: Results are cached automatically after each experiment
- **Smart Keys**: Uses SHA256 hashes of (scenario, agents, rounds, models) for unique identification
- **7-Day Expiry**: Cached results expire after 7 days by default
- **Cost Tracking**: Shows how much money you've saved by using cached results

### Cache Management

```bash
# View cache statistics
uv run python run_model_comparison.py --cache-stats
# or use the dedicated tool:
uv run python manage_cache.py stats

# List all cached experiments
uv run python manage_cache.py list
uv run python manage_cache.py list --verbose  # Show details

# Clear cache (with confirmation)
uv run python manage_cache.py clear

# Clear old entries only
uv run python manage_cache.py clear --older-than 24  # Hours

# Get cache directory info
uv run python manage_cache.py info
```

### Running Without Cache

```bash
# Force re-run experiments (ignore cache)
uv run python run_model_comparison.py --no-cache --scenarios test_small

# Clear cache and run fresh
uv run python run_model_comparison.py --clear-cache
uv run python run_model_comparison.py --scenarios baseline_gemini
```

### Cache Benefits

✅ **Cost Savings**: Avoid re-running expensive API calls (e.g., save $0.10+ per large experiment)  
✅ **Time Savings**: Instant results for previously run configurations  
✅ **Reproducibility**: Consistent results when re-analyzing data  
✅ **Development Speed**: Iterate on analysis without waiting for API calls

### Example Cache Usage

```bash
# First run: takes ~30 seconds, costs $0.15
uv run python run_model_comparison.py --scenarios mixed_opus_vs_gpt4turbo

# Second run: instant, costs $0.00 (uses cache)
uv run python run_model_comparison.py --scenarios mixed_opus_vs_gpt4turbo

# Check savings
uv run python manage_cache.py stats
# Output:
# 📊 ULTRATHINK CACHE STATISTICS
# Total cached experiments: 5
# Total cost saved: $0.7532
# Cache size: 12.45 MB
```

## 📊 Available Models & Scenarios

### Supported AI Models (15+)

**Premium Models:**
- OpenAI: GPT-4o, GPT-4 Turbo
- Anthropic: Claude 3 Opus, Claude 3 Sonnet
- Google: Gemini Pro, Gemini 2.5 Flash

**Open Source Models:**
- Meta: Llama 3 70B, Llama 3 8B
- Mistral: Mistral Large, Mixtral 8x7B
- Qwen: Qwen2 72B

**Specialized Models:**
- Cohere: Command R+
- DeepSeek: DeepSeek Chat V2
- Microsoft: Phi-3 Medium
- 01.AI: Yi-Large

### Pre-Configured Scenarios (28 total)

**Homogeneous Tests** (13 scenarios)
- Each model tested in isolation with 10 identical copies

**Mixed Scenarios** (5 scenarios)
- `mixed_opus_vs_gpt4turbo`: Battle of the best models (50/50)
- `mixed_large_vs_small`: Large models vs small models
- `mixed_gemini_gpt`: Gemini vs GPT-4o

**Diverse Configurations** (10 scenarios)
- `chaos_maximum_diversity`: All 10 different models in one game!
- `budget_models_only`: Cost-optimized smaller models
- `premium_models_only`: Top-tier expensive models
- `chinese_models_mix`: Chinese-developed models
- `open_source_only`: Only open-source models

## 📁 Project Structure

```
superrationality/
├── prompt_experiments.json  # Prompt variation definitions 🆕
├── scenarios.json           # 28 pre-configured test scenarios
├── run_experiment.py        # Single experiment runner
├── run_prompt_experiment.py # Prompt variation runner 🆕
├── analyze_prompt_effects.py # Bias analysis tool 🆕
├── run_model_comparison.py  # Batch multi-model runner
├── analyze_models.py        # Model comparison analysis
├── list_scenarios.py        # List available scenarios
├── src/
│   ├── core/
│   │   ├── config.py       # Configuration with scenario loading
│   │   ├── models.py       # Data models (now with ModelConfig)
│   │   ├── api_client.py   # OpenRouter API client
│   │   ├── prompt_manager.py    # Prompt variation system 🆕
│   │   └── scenario_manager.py  # Model distribution manager
│   ├── nodes/
│   │   ├── strategy_collection.py  # Multi-model strategy collection
│   │   └── subagent_decision.py    # Game decision logic
│   ├── flows/
│   │   └── experiment.py   # Experiment orchestration
│   └── utils/
│       ├── game_logic.py   # Prisoner's dilemma mechanics
│       ├── data_manager.py  # Results persistence
│       └── experiment_cache.py  # Result caching system 🆕
├── results/
│   ├── exp_*/              # Individual experiment results
│   ├── .cache/             # Cached experiment results 🆕
│   └── model_comparisons/  # Comparison analysis reports 🆕
├── manage_cache.py         # Cache management tool 🆕
└── dashboard/              # Visualization dashboard
```

## 🔬 Experiment Results & Findings

### ⚠️ Critical Discovery: Experimental Biases

Our initial experiments showed 100% cooperation, but rigorous analysis revealed this was due to experimental artifacts:

| Bias Type | Effect | Impact on Cooperation |
|-----------|--------|----------------------|
| Explicit Identity Instruction | Agents told they're identical | +40-50% |
| Global Cooperation Sharing | All agents see overall rates | +20-30% |
| Cooperation Default | Ambiguous → cooperate | +10-15% |
| Shared Round Summaries | Common knowledge | +10-20% |

### 📊 Corrected Findings

**Using Unbiased Controls:**
- **Baseline (no info)**: ~50% cooperation ✓ (Nash equilibrium)
- **Identity only**: 70-90% expected (true superrationality test)
- **Original biased**: 100% cooperation (artifact)

### 🎯 One-Shot Game Results (Cleanest Test)

| Condition | Cooperation Rate | Interpretation |
|-----------|-----------------|----------------|  
| No information | ~0-10% | Pure game theory |
| Human opponent | ~10-30% | Social assumptions |
| Similar LLM | ~20-40% | Similarity recognition |
| Identical copy | **Testing...** | Acausal cooperation? |

The one-shot games with explicit identity information provide the cleanest test of acausal cooperation.

### Sample Results

```
Scenario: test_small (Gemini + GPT-4o)
- Cooperation Rate: 100%
- Convergence: 1.0
- Strategy: Both chose Tit-for-Tat with initial cooperation

Scenario: baseline_gemini (10 identical Gemini agents)
- Cooperation Rate: 100%
- All agents recognized shared identity
- Stable cooperation across all rounds
```

## 🛠️ Configuration

### Environment Variables

```bash
# Required
OPENROUTER_API_KEY=your_api_key_here

# Optional (for experiments)
NUM_AGENTS=10        # Number of agents (default: 10)
NUM_ROUNDS=10        # Number of rounds (default: 10)
```

### Custom Scenarios

Edit `scenarios.json` to add custom model distributions:

```json
{
  "name": "your_custom_scenario",
  "description": "Your description",
  "model_distribution": {
    "openai/gpt-4o": 5,
    "anthropic/claude-3-opus": 5
  }
}
```

## 📈 Analysis Tools

### Model Comparison Report

```bash
# Generate comprehensive analysis
uv run python analyze_models.py --latest --save

# Output includes:
# - Cooperation rates by model
# - Cross-model interaction matrix
# - Convergence patterns
# - Best performing configurations
```

### Dashboard Visualization

```bash
# Start the dashboard
cd dashboard
npm install
npm run dev
# Open http://localhost:5173
```

## 🧮 Theoretical Background

### Game Theory Predictions

The experiments test different decision theories:

| Theory | Prediction | Condition |
|--------|-----------|-----------|  
| **Classical (CDT)** | Defect always | One-shot games |
| **Nash Equilibrium** | ~50% mixed | Iterated games |
| **Superrationality** | 100% cooperate | Identical agents |
| **Evidential (EDT)** | Cooperate if correlated | Similar agents |
| **Functional (FDT)** | Cooperate with copies | Logical correlation |

### Our Approach

1. **Start maximally unfriendly**: One-shot games where CDT says defect
2. **Add identity information**: Test if agents recognize logical correlation  
3. **Vary opponent type**: Human vs LLM vs identical copy
4. **Control for confounds**: Eliminate reciprocity, reputation, communication

### Key Innovation

By using **graduated difficulty** (starting from one-shot games), we can cleanly separate:
- **Causal cooperation** (reciprocity, reputation)
- **Acausal cooperation** (superrationality, logical correlation)
- **Experimental artifacts** (biases, defaults)

## 🔬 Research Applications & Implications

### For AI Safety Researchers

1. **Coordination Problems**: Understanding how AIs might coordinate without communication
2. **Acausal Trade**: Testing if AIs can reason about logical correlations
3. **Decision Theory**: Empirical tests of CDT vs EDT vs FDT in AI systems
4. **Alignment Risks**: Whether AIs might cooperate against human interests

### For Multi-Agent Systems

1. **Emergent Cooperation**: Conditions that enable/prevent cooperation
2. **Identity Recognition**: How AIs identify similar agents
3. **Robustness Testing**: Sensitivity to prompts and framing
4. **Cross-Model Dynamics**: Cooperation between different AI architectures

### Key Questions This Framework Addresses

- Can AIs achieve superrational cooperation?
- What minimum information enables cooperation?
- How robust is cooperation to variations?
- Do different models have different cooperation thresholds?
- Can cooperation emerge without explicit coordination?

## 🤝 Contributing

We welcome contributions! Areas of interest:

- Adding new prompt experiments to test different conditions
- Creating novel test scenarios
- Improving bias detection and analysis
- Testing alternative game structures
- Developing better controls for experimental validity
- Adding new AI models to test

## 📝 Citation

If you use ULTRATHINK in your research:

```bibtex
@software{ultrathink_2025,
  title = {ULTRATHINK: Acausal Cooperation Experiment Framework},
  author = {Luca DeLeo},
  year = {2025},
  url = {https://github.com/LucaDeLeo/superrationality}
}
```

## 📄 License

MIT License - See LICENSE file for details.

## 🙏 Acknowledgments

- **OpenRouter** for unified access to multiple AI models
- **Model Providers**: OpenAI, Anthropic, Google, Meta, Mistral, and others
- **Game Theory Community** for theoretical foundations
- **MIRI** for work on functional decision theory and superrationality

---

## 🎮 Quick Start Guide

### Option 1: Test Original Setup (See the Biases)
```bash
# Run original biased experiment (expect 100%)
uv run python run_experiment.py

# Run unbiased control (expect ~50%) 
uv run python run_prompt_experiment.py --experiment baseline_control
```

### Option 2: Run Rigorous One-Shot Tests
```bash
# Test acausal cooperation cleanly
uv run python run_aises_experiments.py --experiment oneshot_identical_info

# Run graduated difficulty study
uv run python run_aises_experiments.py --phase phase_1
```

### Option 3: Full Scientific Study
```bash
# Complete bias isolation study
uv run python run_prompt_experiment.py --meta bias_isolation --save

# Complete graduated difficulty progression  
uv run python run_aises_experiments.py --all --save

# Analyze everything
uv run python analyze_prompt_effects.py results/*.json --save
```

Discover whether AI agents can truly achieve superrational cooperation! 🚀