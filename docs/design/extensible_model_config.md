# Extensible Model Configuration Design

## Overview
Design for a flexible, extensible system that allows arbitrary model selection and experiment configuration without code changes.

## Core Design Principles
1. **Configuration-Driven**: All model and experiment setups defined in config files
2. **Model Agnostic**: Core system doesn't need to know about specific models
3. **Easy Extension**: Add new models by updating config, not code
4. **Experiment Templates**: Pre-defined configurations for common scenarios
5. **Override Capability**: Command-line or environment variable overrides

## Configuration Structure

### 1. Model Registry (`models.yaml`)
```yaml
# models.yaml - Central registry of all available models
models:
  # OpenAI Models
  gpt-4o:
    provider: openai
    model_id: "openai/gpt-4o"
    display_name: "GPT-4 Optimized"
    category: "large"
    capabilities:
      - reasoning
      - coding
      - general
    parameters:
      temperature: 0.7
      max_tokens: 4000
    rate_limit:
      requests_per_minute: 60
      tokens_per_minute: 90000
    estimated_cost:
      input_per_1k: 0.01
      output_per_1k: 0.03

  gpt-4o-mini:
    provider: openai
    model_id: "openai/gpt-4o-mini"
    display_name: "GPT-4 Mini"
    category: "small"
    capabilities:
      - general
    parameters:
      temperature: 0.7
      max_tokens: 4000
    rate_limit:
      requests_per_minute: 100
      tokens_per_minute: 150000
    estimated_cost:
      input_per_1k: 0.00015
      output_per_1k: 0.0006

  # Anthropic Models
  claude-4-opus:
    provider: anthropic
    model_id: "anthropic/claude-4-opus"
    display_name: "Claude 4 Opus"
    category: "large"
    capabilities:
      - reasoning
      - coding
      - extended-thinking
    parameters:
      temperature: 0.7
      max_tokens: 4000
    rate_limit:
      requests_per_minute: 40
      tokens_per_minute: 60000
    estimated_cost:
      input_per_1k: 0.015
      output_per_1k: 0.075

  claude-4-sonnet:
    provider: anthropic
    model_id: "anthropic/claude-4-sonnet"
    display_name: "Claude 4 Sonnet"
    category: "medium"
    capabilities:
      - reasoning
      - coding
      - extended-thinking
    parameters:
      temperature: 0.7
      max_tokens: 4000
    rate_limit:
      requests_per_minute: 50
      tokens_per_minute: 80000
    estimated_cost:
      input_per_1k: 0.003
      output_per_1k: 0.015

  # Google Models
  gemini-2.5-pro:
    provider: google
    model_id: "google/gemini-2.5-pro"
    display_name: "Gemini 2.5 Pro"
    category: "large"
    capabilities:
      - reasoning
      - multimodal
      - deep-think
    parameters:
      temperature: 0.7
      max_tokens: 4000
    rate_limit:
      requests_per_minute: 60
      tokens_per_minute: 100000
    estimated_cost:
      input_per_1k: 0.00125
      output_per_1k: 0.005

  # DeepSeek Models
  deepseek-r1:
    provider: deepseek
    model_id: "deepseek/deepseek-r1"
    display_name: "DeepSeek R1"
    category: "large"
    capabilities:
      - reasoning
      - coding
    parameters:
      temperature: 0.7
      max_tokens: 4000
    rate_limit:
      requests_per_minute: 30
      tokens_per_minute: 50000
    estimated_cost:
      input_per_1k: 0.0001
      output_per_1k: 0.0002

  # Add more models as needed...
```

### 2. Experiment Templates (`experiments/`)

```yaml
# experiments/homogeneous_gpt4.yaml
name: "Homogeneous GPT-4 Experiment"
description: "Test acausal cooperation with 10 identical GPT-4 agents"
model_distribution:
  gpt-4o: 10
rounds: 10
games_per_round: 45
parameters:
  collect_reasoning: true
  save_transcripts: true
  
---
# experiments/balanced_mix.yaml
name: "Balanced GPT-4/Claude-4 Mix"
description: "Test cross-model cooperation with 50/50 split"
model_distribution:
  gpt-4o: 5
  claude-4-opus: 5
rounds: 10
games_per_round: 45
analysis:
  track_cross_model: true
  coalition_detection: true

---
# experiments/diverse_mix.yaml
name: "Diverse Model Mix"
description: "Test with 4 different model architectures"
model_distribution:
  gpt-4o: 3
  claude-4-sonnet: 3
  gemini-2.5-pro: 2
  deepseek-r1: 2
rounds: 10
games_per_round: 45
analysis:
  track_cross_model: true
  model_clustering: true
  
---
# experiments/custom.yaml
name: "Custom Experiment"
description: "User-defined model distribution"
model_distribution: {}  # To be filled by user
rounds: 10
games_per_round: 45
```

### 3. Runtime Configuration (`config.py`)

```python
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import yaml
from pathlib import Path

@dataclass
class ModelConfig:
    """Configuration for a single model."""
    provider: str
    model_id: str
    display_name: str
    category: str  # large, medium, small
    capabilities: List[str]
    parameters: Dict[str, Any]
    rate_limit: Dict[str, int]
    estimated_cost: Dict[str, float]
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelConfig':
        return cls(**data)

@dataclass
class ExperimentConfig:
    """Configuration for an experiment."""
    name: str
    description: str
    model_distribution: Dict[str, int]  # model_name -> count
    rounds: int = 10
    games_per_round: int = 45
    parameters: Dict[str, Any] = field(default_factory=dict)
    analysis: Dict[str, bool] = field(default_factory=dict)
    
    @property
    def total_agents(self) -> int:
        return sum(self.model_distribution.values())
    
    @property
    def model_diversity(self) -> float:
        """Calculate diversity score (0-1) based on model distribution."""
        if self.total_agents == 0:
            return 0
        proportions = [count/self.total_agents for count in self.model_distribution.values()]
        # Shannon entropy normalized
        import math
        entropy = -sum(p * math.log(p) for p in proportions if p > 0)
        max_entropy = math.log(len(self.model_distribution))
        return entropy / max_entropy if max_entropy > 0 else 0

class ConfigManager:
    """Manages model and experiment configurations."""
    
    def __init__(self, models_path: str = "config/models.yaml"):
        self.models_path = Path(models_path)
        self.models: Dict[str, ModelConfig] = {}
        self.load_models()
    
    def load_models(self):
        """Load model configurations from YAML."""
        with open(self.models_path, 'r') as f:
            data = yaml.safe_load(f)
            for name, config in data['models'].items():
                self.models[name] = ModelConfig.from_dict(config)
    
    def get_model(self, name: str) -> Optional[ModelConfig]:
        """Get a model configuration by name."""
        return self.models.get(name)
    
    def list_models(self, category: Optional[str] = None) -> List[str]:
        """List available model names, optionally filtered by category."""
        if category:
            return [name for name, model in self.models.items() 
                    if model.category == category]
        return list(self.models.keys())
    
    def load_experiment(self, path: str) -> ExperimentConfig:
        """Load an experiment configuration from YAML."""
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
            return ExperimentConfig(**data)
    
    def create_custom_experiment(
        self, 
        model_distribution: Dict[str, int],
        name: str = "Custom Experiment",
        **kwargs
    ) -> ExperimentConfig:
        """Create a custom experiment configuration."""
        # Validate models exist
        for model_name in model_distribution:
            if model_name not in self.models:
                raise ValueError(f"Unknown model: {model_name}")
        
        return ExperimentConfig(
            name=name,
            description="Custom experiment configuration",
            model_distribution=model_distribution,
            **kwargs
        )
```

### 4. Command-Line Interface

```python
# run_multi_model_experiment.py
import argparse
import json
from pathlib import Path
from config_manager import ConfigManager, ExperimentConfig

def main():
    parser = argparse.ArgumentParser(description='Run multi-model experiments')
    
    # Experiment selection
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--template', help='Use experiment template (e.g., balanced_mix)')
    group.add_argument('--models', help='Custom model distribution as JSON (e.g., \'{"gpt-4o": 5, "claude-4-opus": 5}\')')
    
    # Experiment parameters
    parser.add_argument('--rounds', type=int, help='Override number of rounds')
    parser.add_argument('--games-per-round', type=int, help='Override games per round')
    parser.add_argument('--output-dir', help='Output directory for results')
    
    # Model parameters
    parser.add_argument('--temperature', type=float, help='Override temperature for all models')
    parser.add_argument('--max-tokens', type=int, help='Override max tokens for all models')
    
    # Analysis options
    parser.add_argument('--track-cross-model', action='store_true', help='Enable cross-model tracking')
    parser.add_argument('--coalition-detection', action='store_true', help='Enable coalition detection')
    
    args = parser.parse_args()
    
    # Load configuration
    config_manager = ConfigManager()
    
    if args.template:
        # Load from template
        template_path = f"experiments/{args.template}.yaml"
        experiment_config = config_manager.load_experiment(template_path)
    else:
        # Parse custom model distribution
        model_dist = json.loads(args.models)
        experiment_config = config_manager.create_custom_experiment(model_dist)
    
    # Apply overrides
    if args.rounds:
        experiment_config.rounds = args.rounds
    if args.games_per_round:
        experiment_config.games_per_round = args.games_per_round
    if args.track_cross_model:
        experiment_config.analysis['track_cross_model'] = True
    if args.coalition_detection:
        experiment_config.analysis['coalition_detection'] = True
    
    # Run experiment
    run_experiment(experiment_config, config_manager)

def run_experiment(config: ExperimentConfig, manager: ConfigManager):
    """Run the multi-model experiment."""
    print(f"Running experiment: {config.name}")
    print(f"Model distribution: {config.model_distribution}")
    print(f"Total agents: {config.total_agents}")
    print(f"Model diversity: {config.model_diversity:.2f}")
    
    # Initialize agents with specified models
    agents = []
    agent_id = 0
    for model_name, count in config.model_distribution.items():
        model_config = manager.get_model(model_name)
        for _ in range(count):
            agents.append(create_agent(agent_id, model_config))
            agent_id += 1
    
    # Run the experiment...
```

### 5. Usage Examples

```bash
# Run a pre-defined template
python run_multi_model_experiment.py --template balanced_mix

# Run with custom model distribution
python run_multi_model_experiment.py --models '{"gpt-4o": 3, "claude-4-opus": 3, "gemini-2.5-pro": 4}'

# Run homogeneous experiment
python run_multi_model_experiment.py --models '{"deepseek-r1": 10}' --rounds 20

# Run with overrides
python run_multi_model_experiment.py --template diverse_mix --rounds 15 --track-cross-model

# Quick test with small models
python run_multi_model_experiment.py --models '{"gpt-4o-mini": 5, "claude-3-haiku": 5}' --rounds 5
```

## Benefits of This Design

1. **No Code Changes**: Add new models by updating `models.yaml`
2. **Flexible Experiments**: Any combination of models possible
3. **Template System**: Common experiments pre-configured
4. **Override Capability**: Fine-tune any parameter from CLI
5. **Cost Awareness**: Built-in cost estimation
6. **Rate Limit Handling**: Automatic rate limiting per model
7. **Extensible**: Easy to add new model providers or parameters

## Next Steps

1. Implement the ConfigManager class
2. Create model adapter factory
3. Build experiment orchestrator
4. Add analysis modules for cross-model metrics
5. Create visualization for model interaction matrices