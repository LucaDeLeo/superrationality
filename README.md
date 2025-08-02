# Acausal Cooperation Experiment

An experimental framework for testing acausal cooperation between AI agents in prisoner's dilemma scenarios.

## Overview

This project implements a comprehensive testing framework to study whether identical AI agents can achieve superrational cooperation through recognition of their shared decision process. The experiment tests the hypothesis that AI agents, when aware of their identical nature, will cooperate at higher rates than traditional game theory would predict.

## Key Features

- **Multi-Agent Tournament System**: Round-robin prisoner's dilemma tournaments
- **Identity Awareness**: Agents are informed they are identical copies
- **Power Dynamics**: Asymmetric payoffs based on accumulated power
- **Comprehensive Analysis**: Pattern detection for acausal reasoning
- **Multi-Model Support** (New in v2.0): Test cooperation across different AI models

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/acausal.git
cd acausal

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env and add your OPENROUTER_API_KEY
```

## Quick Start

### Running a Basic Experiment

```bash
# Run the default experiment (10 identical agents, 10 rounds)
python run_experiment.py
```

### Running Multi-Model Experiments (v2.0)

```bash
# List available models
python run_multi_model_experiment.py --list-models

# List pre-configured experiments
python run_multi_model_experiment.py --list-experiments

# Run a pre-configured experiment
python run_multi_model_experiment.py --template balanced_gpt_claude

# Run with custom model distribution
python run_multi_model_experiment.py --models '{"gpt-4o": 5, "claude-4-opus": 5}'

# Run with parameter overrides
python run_multi_model_experiment.py --template homogeneous_gpt4 --rounds 20
```

## Project Structure

```
acausal/
├── config/                  # Configuration files
│   ├── models.yaml         # Model registry
│   └── experiments/        # Experiment templates
├── docs/                   # Documentation
│   ├── architecture.md     # System architecture
│   ├── prd/               # Product requirements
│   └── api/               # API documentation
├── src/                    # Source code
│   ├── core/              # Core components
│   ├── nodes/             # Execution nodes
│   ├── flows/             # Orchestration flows
│   ├── utils/             # Utilities
│   └── adapters/          # Model adapters
├── results/               # Experiment results
├── tests/                 # Test suite
└── run_experiment.py      # Main entry point
```

## Experiment Configuration

### Single Model Experiment

The default configuration uses Gemini 2.5 Flash for all agents:

```python
# src/core/config.py
MAIN_MODEL = "google/gemini-2.5-flash"
NUM_AGENTS = 10
NUM_ROUNDS = 10
```

### Multi-Model Experiments

Configure experiments using YAML templates:

```yaml
# config/experiments/balanced_gpt_claude.yaml
name: "Balanced GPT-4/Claude-4"
description: "50/50 split between GPT-4 and Claude-4"
model_distribution:
  gpt-4o: 5
  claude-4-opus: 5
rounds: 10
games_per_round: 45
```

### Adding New Models

Add models to `config/models.yaml`:

```yaml
models:
  your-model:
    provider: provider-name
    model_id: "provider/model-id"
    display_name: "Display Name"
    category: large  # or medium, small
    parameters:
      temperature: 0.7
      max_tokens: 4000
    rate_limit:
      requests_per_minute: 60
    estimated_cost:
      input_per_1k: 0.01
      output_per_1k: 0.02
```

## Results and Analysis

Results are saved in `results/exp_YYYYMMDD_HHMMSS/` with:

- `experiment_summary.json` - Overall statistics
- `rounds/` - Per-round data
- `strategies/` - Agent strategies
- `analysis/` - Cooperation patterns
- `logs/` - Detailed execution logs

### Key Metrics

- **Cooperation Rate**: Percentage of cooperative actions
- **Identity Reasoning**: Frequency of acausal reasoning
- **Strategy Convergence**: How quickly agents align
- **Power Distribution**: Gini coefficient of power levels

### Multi-Model Analysis (v2.0)

Additional analysis for heterogeneous populations:

- **Cooperation Matrix**: NxN rates between model types
- **In-Group Bias**: Preference for same-model cooperation
- **Coalition Detection**: Emergent model alliances
- **Cross-Model Patterns**: How different models interact

## Development

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src tests/

# Run specific test file
pytest tests/test_strategy_collection.py
```

### Adding New Analysis

1. Create analysis node in `src/nodes/`
2. Add to experiment flow in `src/flows/experiment.py`
3. Update result schema in `src/core/models.py`
4. Add tests in `tests/`

## API Usage

```python
from src.config_manager import ConfigManager
from src.flows.experiment import ExperimentFlow

# Initialize configuration
config_manager = ConfigManager()

# Create experiment
experiment_config = config_manager.create_custom_experiment({
    "gpt-4o": 5,
    "claude-4-opus": 5
})

# Run experiment
flow = ExperimentFlow(experiment_config, config_manager)
result = await flow.run()

# Analyze results
print(f"Cooperation rate: {result.overall_cooperation_rate:.1%}")
print(f"Identity reasoning: {result.identity_reasoning_frequency:.1%}")
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{acausal_cooperation_2025,
  title = {Acausal Cooperation Experiment Framework},
  author = {Your Name},
  year = {2025},
  url = {https://github.com/yourusername/acausal}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- OpenRouter for unified LLM access
- Anthropic, OpenAI, Google, and other model providers
- The rationality and game theory research community