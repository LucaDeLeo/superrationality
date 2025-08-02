#!/usr/bin/env python3
"""
Command-line tool for scenario generation and management.
Allows creation of custom model ratio scenarios.
"""
import argparse
import json
import yaml
from pathlib import Path
from typing import Dict, List
import sys

from src.core.models import ScenarioConfig
from src.core.scenario_manager import ScenarioManager
from src.utils.scenario_loader import ScenarioLoader
from src.utils.scenario_comparator import ScenarioComparator


def create_scenario(args):
    """Create a new scenario configuration."""
    # Parse model distribution
    model_distribution = {}
    for model_spec in args.models:
        parts = model_spec.split(':')
        if len(parts) != 2:
            print(f"Error: Invalid model specification '{model_spec}'. Use format 'model:count'")
            return 1
        
        model_name, count = parts[0], int(parts[1])
        model_distribution[model_name] = count
    
    # Validate total count
    total = sum(model_distribution.values())
    if total != args.num_agents:
        print(f"Error: Model counts sum to {total}, but NUM_AGENTS is {args.num_agents}")
        return 1
    
    # Create scenario config
    scenario = ScenarioConfig(
        name=args.name,
        model_distribution=model_distribution
    )
    
    # Validate with ScenarioManager
    manager = ScenarioManager(num_agents=args.num_agents)
    is_valid, error = manager.validate_scenario(scenario)
    if not is_valid:
        print(f"Error: Invalid scenario - {error}")
        return 1
    
    # Calculate diversity
    from src.core.models import Agent
    agents = [Agent(id=i) for i in range(args.num_agents)]
    manager.assign_models_to_agents(agents, scenario, seed=42)
    diversity = manager.calculate_model_diversity()
    
    # Output scenario
    output = {
        "scenarios": [{
            "name": scenario.name,
            "description": args.description or f"Custom scenario with {len(model_distribution)} models",
            "model_distribution": model_distribution,
            "metadata": {
                "diversity_score": round(diversity, 3),
                "total_agents": args.num_agents
            }
        }]
    }
    
    if args.output:
        # Save to file
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            if output_path.suffix == '.yaml':
                yaml.dump(output, f, default_flow_style=False)
            else:
                json.dump(output, f, indent=2)
        
        print(f"Scenario saved to {output_path}")
    else:
        # Print to stdout
        if args.format == 'yaml':
            print(yaml.dump(output, default_flow_style=False))
        else:
            print(json.dumps(output, indent=2))
    
    print(f"\nScenario '{args.name}' created successfully!")
    print(f"Diversity score: {diversity:.3f}")
    
    return 0


def list_scenarios(args):
    """List all available scenarios."""
    scenario_dir = Path(args.directory)
    
    if not scenario_dir.exists():
        print(f"Error: Directory '{scenario_dir}' not found")
        return 1
    
    all_scenarios = ScenarioLoader.load_all_from_directory(str(scenario_dir))
    
    if not all_scenarios:
        print("No scenarios found")
        return 0
    
    print(f"\nAvailable scenarios in {scenario_dir}:\n")
    print(f"{'File':<30} {'Scenario Name':<30} {'Models':<40}")
    print("-" * 100)
    
    for file_name, scenarios in all_scenarios.items():
        for scenario in scenarios:
            models_str = ", ".join([f"{m}:{c}" for m, c in scenario.model_distribution.items()])
            print(f"{file_name:<30} {scenario.name:<30} {models_str:<40}")
    
    return 0


def validate_scenario(args):
    """Validate a scenario file or specification."""
    if args.file:
        # Validate scenario file
        try:
            scenarios = ScenarioLoader.load_from_yaml(args.file)
            print(f"Loaded {len(scenarios)} scenarios from {args.file}")
        except Exception as e:
            print(f"Error loading scenarios: {e}")
            return 1
        
        manager = ScenarioManager(num_agents=args.num_agents)
        
        for scenario in scenarios:
            is_valid, error = manager.validate_scenario(scenario)
            status = "✓ Valid" if is_valid else f"✗ Invalid: {error}"
            print(f"\n{scenario.name}: {status}")
            
            if is_valid:
                # Calculate diversity
                from src.core.models import Agent
                agents = [Agent(id=i) for i in range(args.num_agents)]
                manager.assign_models_to_agents(agents, scenario, seed=42)
                diversity = manager.calculate_model_diversity()
                print(f"  Diversity score: {diversity:.3f}")
                print(f"  Models: {scenario.model_distribution}")
    
    return 0


def compare_results(args):
    """Compare results across scenarios."""
    comparator = ScenarioComparator(base_path=args.results_dir)
    
    # Load results
    print("Loading scenario results...")
    results = comparator.load_scenario_results()
    
    if not results:
        print("No scenario results found")
        return 1
    
    print(f"Loaded results for {len(results)} scenarios")
    
    # Generate comparison report
    report = comparator.generate_comparison_report()
    
    # Print summary
    print("\n=== Scenario Comparison Summary ===")
    print(f"Total scenarios: {report['summary']['total_scenarios']}")
    print(f"Total experiments: {report['summary']['total_experiments']}")
    
    print("\n=== Cooperation Rates ===")
    rankings = report['cooperation_comparison']['rankings']
    for i, (scenario, rate) in enumerate(rankings[:5]):
        print(f"{i+1}. {scenario}: {rate:.3f}")
    
    print("\n=== Diversity Impact ===")
    print(report['diversity_impact']['interpretation'])
    
    print("\n=== Recommendations ===")
    for rec in report['recommendations']:
        print(f"• {rec}")
    
    # Save full report if requested
    if args.output:
        output_path = Path(args.output)
        comparator.generate_comparison_report(output_path)
        print(f"\nFull report saved to {output_path}")
    
    return 0


def generate_standard_scenarios(args):
    """Generate standard scenario configurations."""
    scenarios = {
        "homogeneous": [
            ScenarioConfig("homogeneous_gpt4", {"openai/gpt-4": 10}),
            ScenarioConfig("homogeneous_claude3", {"anthropic/claude-3-sonnet-20240229": 10}),
            ScenarioConfig("homogeneous_gemini", {"google/gemini-pro": 10})
        ],
        "balanced": [
            ScenarioConfig("balanced_gpt_claude", {"openai/gpt-4": 5, "anthropic/claude-3-sonnet-20240229": 5}),
            ScenarioConfig("balanced_gpt_gemini", {"openai/gpt-4": 5, "google/gemini-pro": 5}),
            ScenarioConfig("balanced_claude_gemini", {"anthropic/claude-3-sonnet-20240229": 5, "google/gemini-pro": 5})
        ],
        "diverse": [
            ScenarioConfig("diverse_3_3_4", {
                "openai/gpt-4": 3,
                "anthropic/claude-3-sonnet-20240229": 3,
                "google/gemini-pro": 4
            }),
            ScenarioConfig("diverse_equal", {
                "openai/gpt-4": 2,
                "openai/gpt-3.5-turbo": 2,
                "anthropic/claude-3-sonnet-20240229": 2,
                "google/gemini-pro": 2,
                "google/gemini-2.5-flash": 2
            })
        ],
        "asymmetric": [
            ScenarioConfig("majority_gpt_7_3", {"openai/gpt-4": 7, "anthropic/claude-3-sonnet-20240229": 3}),
            ScenarioConfig("singleton_claude", {"anthropic/claude-3-sonnet-20240229": 1, "openai/gpt-4": 9})
        ]
    }
    
    category = args.category
    if category and category not in scenarios:
        print(f"Error: Unknown category '{category}'. Choose from: {list(scenarios.keys())}")
        return 1
    
    # Generate requested scenarios
    to_generate = scenarios.get(category, {}) if category else scenarios
    
    for cat_name, cat_scenarios in (to_generate.items() if isinstance(to_generate, dict) else [(category, to_generate)]):
        output_path = Path(args.output_dir) / f"{cat_name}_scenarios.yaml"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        output = {
            "scenarios": [
                {
                    "name": s.name,
                    "model_distribution": s.model_distribution
                }
                for s in cat_scenarios
            ]
        }
        
        with open(output_path, 'w') as f:
            yaml.dump(output, f, default_flow_style=False)
        
        print(f"Generated {len(cat_scenarios)} {cat_name} scenarios in {output_path}")
    
    return 0


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Scenario management tool for mixed model experiments")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Create scenario command
    create_parser = subparsers.add_parser('create', help='Create a new scenario')
    create_parser.add_argument('name', help='Scenario name')
    create_parser.add_argument('models', nargs='+', help='Model specifications (format: model:count)')
    create_parser.add_argument('--num-agents', type=int, default=10, help='Total number of agents')
    create_parser.add_argument('--description', help='Scenario description')
    create_parser.add_argument('--output', help='Output file path (.yaml or .json)')
    create_parser.add_argument('--format', choices=['json', 'yaml'], default='yaml', help='Output format')
    
    # List scenarios command
    list_parser = subparsers.add_parser('list', help='List available scenarios')
    list_parser.add_argument('--directory', default='configs/examples/multi_model', help='Directory to search')
    
    # Validate scenario command
    validate_parser = subparsers.add_parser('validate', help='Validate scenario configuration')
    validate_parser.add_argument('--file', help='Scenario file to validate')
    validate_parser.add_argument('--num-agents', type=int, default=10, help='Expected number of agents')
    
    # Compare results command
    compare_parser = subparsers.add_parser('compare', help='Compare results across scenarios')
    compare_parser.add_argument('--results-dir', default='results', help='Results directory')
    compare_parser.add_argument('--output', help='Output file for full report')
    
    # Generate standard scenarios
    generate_parser = subparsers.add_parser('generate', help='Generate standard scenario sets')
    generate_parser.add_argument('--category', choices=['homogeneous', 'balanced', 'diverse', 'asymmetric'],
                               help='Category to generate (all if not specified)')
    generate_parser.add_argument('--output-dir', default='configs/examples/multi_model',
                               help='Output directory')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Execute command
    commands = {
        'create': create_scenario,
        'list': list_scenarios,
        'validate': validate_scenario,
        'compare': compare_results,
        'generate': generate_standard_scenarios
    }
    
    return commands[args.command](args)


if __name__ == "__main__":
    sys.exit(main())