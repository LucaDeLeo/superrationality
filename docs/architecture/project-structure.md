# Project Structure

```plaintext
acausal/
├── src/
│   ├── __init__.py
│   ├── experiment.py         # Main ExperimentFlow
│   ├── nodes.py             # StrategyCollection, SubagentDecision
│   ├── game_logic.py        # Payoff calculations
│   ├── api_client.py        # OpenRouter integration
│   └── analysis.py          # Simple analysis
├── configs/                  # Experiment configurations
│   ├── baseline.yaml
│   ├── high_temp.yaml
│   └── low_temp.yaml
├── results/                  # Experiment outputs (gitignored)
│   └── .gitkeep
├── run_experiment.py         # Single experiment runner
├── run_all_experiments.py    # Multiple experiment runner
├── compare_results.py        # Result comparison for paper
├── test_experiment.py        # Basic tests
├── requirements.txt          # Dependencies
├── .env.example             # API key template
└── README.md                # Documentation
```
