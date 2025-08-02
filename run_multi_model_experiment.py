#!/usr/bin/env python3
"""
Run multi-model experiments with flexible configuration.

Examples:
    # Run a pre-defined experiment
    python run_multi_model_experiment.py --template homogeneous_gpt4
    
    # Run with custom models
    python run_multi_model_experiment.py --models '{"gpt-4o": 5, "claude-4-opus": 5}'
    
    # Override parameters
    python run_multi_model_experiment.py --template balanced_gpt_claude --rounds 20
    
    # List available models
    python run_multi_model_experiment.py --list-models
    
    # List available experiments
    python run_multi_model_experiment.py --list-experiments
"""

import argparse
import json
import yaml
import sys
from pathlib import Path
from typing import Dict, Optional
import asyncio

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.config_manager import ConfigManager, ExperimentConfig


def list_models(config_manager: ConfigManager, category: Optional[str] = None):
    """List available models."""
    print("\nAvailable Models:")
    print("=" * 60)
    
    models = config_manager.list_models(category=category)
    
    for model_name in sorted(models):
        model = config_manager.get_model(model_name)
        print(f"\n{model_name}:")
        print(f"  Display Name: {model.display_name}")
        print(f"  Provider: {model.provider}")
        print(f"  Category: {model.category}")
        print(f"  Cost: ${model.estimated_cost['input_per_1k']:.4f} input / ${model.estimated_cost['output_per_1k']:.4f} output per 1K tokens")


def list_experiments(experiments_dir: Path):
    """List available experiment templates."""
    print("\nAvailable Experiment Templates:")
    print("=" * 60)
    
    for exp_file in experiments_dir.glob("*.yaml"):
        if exp_file.name == "custom_template.yaml":
            continue
            
        with open(exp_file, 'r') as f:
            data = yaml.safe_load(f)
            
        print(f"\n{exp_file.stem}:")
        print(f"  Name: {data.get('name', 'N/A')}")
        print(f"  Description: {data.get('description', 'N/A')}")
        
        if 'model_distribution' in data and data['model_distribution']:
            print("  Models:")
            for model, count in data['model_distribution'].items():
                print(f"    - {model}: {count}")


def validate_experiment(config: ExperimentConfig, manager: ConfigManager) -> bool:
    """Validate experiment configuration."""
    # Check all models exist
    for model_name in config.model_distribution:
        if not manager.get_model(model_name):
            print(f"Error: Unknown model '{model_name}'")
            return False
    
    # Check minimum agents
    if config.total_agents < 2:
        print("Error: Need at least 2 agents for an experiment")
        return False
    
    # Estimate cost
    total_cost = estimate_experiment_cost(config, manager)
    print(f"\nEstimated experiment cost: ${total_cost:.2f}")
    
    if total_cost > 100:
        response = input("This experiment may cost over $100. Continue? (y/N): ")
        if response.lower() != 'y':
            return False
    
    return True


def estimate_experiment_cost(config: ExperimentConfig, manager: ConfigManager) -> float:
    """Estimate the cost of running an experiment."""
    total_cost = 0.0
    
    # Assume average tokens per interaction
    avg_input_tokens = 1000  # prompt
    avg_output_tokens = 500  # response
    
    # Calculate interactions
    strategy_collections = config.total_agents * config.rounds
    games = config.games_per_round * config.rounds
    
    for model_name, count in config.model_distribution.items():
        model = manager.get_model(model_name)
        if not model:
            continue
            
        # Cost per agent for strategies
        agent_strategy_cost = (
            model.estimated_cost['input_per_1k'] * avg_input_tokens / 1000 +
            model.estimated_cost['output_per_1k'] * avg_output_tokens / 1000
        ) * config.rounds
        
        # Total for this model type
        model_cost = agent_strategy_cost * count
        total_cost += model_cost
    
    return total_cost


async def run_experiment(config: ExperimentConfig, manager: ConfigManager):
    """Run the multi-model experiment."""
    print(f"\nRunning experiment: {config.name}")
    print(f"=" * 60)
    print(f"Model distribution:")
    for model, count in config.model_distribution.items():
        print(f"  {model}: {count} agents")
    print(f"Total agents: {config.total_agents}")
    print(f"Model diversity: {config.model_diversity:.2f}")
    print(f"Rounds: {config.rounds}")
    print(f"Games per round: {config.games_per_round}")
    
    # TODO: Implement actual experiment running
    # This would integrate with the existing experiment framework
    print("\n[Experiment would run here]")
    print("This is a demonstration of the configuration system.")
    

def main():
    parser = argparse.ArgumentParser(
        description='Run multi-model experiments',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Action arguments
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--template', help='Use experiment template from config/experiments/')
    group.add_argument('--models', help='Custom model distribution as JSON')
    group.add_argument('--list-models', action='store_true', help='List available models')
    group.add_argument('--list-experiments', action='store_true', help='List experiment templates')
    
    # Filter arguments
    parser.add_argument('--category', help='Filter models by category (large, medium, small)')
    
    # Experiment parameters
    parser.add_argument('--rounds', type=int, help='Override number of rounds')
    parser.add_argument('--games-per-round', type=int, help='Override games per round')
    parser.add_argument('--name', help='Name for custom experiment')
    
    # Output options
    parser.add_argument('--output-dir', help='Output directory for results')
    parser.add_argument('--dry-run', action='store_true', help='Show configuration without running')
    
    args = parser.parse_args()
    
    # Setup paths
    config_dir = Path(__file__).parent / "config"
    models_path = config_dir / "models.yaml"
    experiments_dir = config_dir / "experiments"
    
    # Initialize config manager
    config_manager = ConfigManager(models_path=models_path)
    
    # Handle list commands
    if args.list_models:
        list_models(config_manager, category=args.category)
        return
    
    if args.list_experiments:
        list_experiments(experiments_dir)
        return
    
    # Load or create experiment config
    if args.template:
        template_path = experiments_dir / f"{args.template}.yaml"
        if not template_path.exists():
            print(f"Error: Template '{args.template}' not found")
            return 1
        
        experiment_config = config_manager.load_experiment(template_path)
        
    elif args.models:
        try:
            model_dist = json.loads(args.models)
        except json.JSONDecodeError as e:
            print(f"Error parsing models JSON: {e}")
            return 1
        
        name = args.name or "Custom Experiment"
        experiment_config = config_manager.create_custom_experiment(
            model_distribution=model_dist,
            name=name
        )
    else:
        parser.print_help()
        return 1
    
    # Apply overrides
    if args.rounds:
        experiment_config.rounds = args.rounds
    if args.games_per_round:
        experiment_config.games_per_round = args.games_per_round
    
    # Validate experiment
    if not validate_experiment(experiment_config, config_manager):
        return 1
    
    # Dry run - just show config
    if args.dry_run:
        print("\nExperiment Configuration:")
        print(yaml.dump(experiment_config.__dict__, default_flow_style=False))
        return 0
    
    # Run experiment
    try:
        asyncio.run(run_experiment(experiment_config, config_manager))
    except KeyboardInterrupt:
        print("\nExperiment interrupted by user")
        return 1
    except Exception as e:
        print(f"\nError running experiment: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())