#!/usr/bin/env python3
"""Run experiments across multiple model scenarios for comparison."""

import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any
import argparse
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set small experiment size for testing
os.environ['NUM_AGENTS'] = os.getenv('NUM_AGENTS', '4')  # 4 agents = 6 games per round
os.environ['NUM_ROUNDS'] = os.getenv('NUM_ROUNDS', '3')  # 3 rounds for quick testing


async def run_scenario_experiment(scenario_name: str, use_cache: bool = True) -> Dict[str, Any]:
    """Run a single experiment with a specific scenario.
    
    Args:
        scenario_name: Name of the scenario to run
        use_cache: Whether to use cached results if available
        
    Returns:
        Experiment results dictionary
    """
    from run_experiment import ExperimentRunner
    from src.utils.experiment_cache import ExperimentCache
    from src.core.config import Config
    
    print(f"\n{'='*60}")
    print(f"🧪 Running scenario: {scenario_name}")
    print(f"{'='*60}")
    
    # Load scenario configuration
    scenarios_path = Path("scenarios.json")
    with open(scenarios_path, 'r') as f:
        data = json.load(f)
    
    scenario_config = None
    for s in data['scenarios']:
        if s['name'] == scenario_name:
            scenario_config = s
            break
    
    if not scenario_config:
        print(f"❌ Scenario {scenario_name} not found")
        return {}
    
    # Check cache if enabled
    if use_cache:
        cache = ExperimentCache()
        num_agents = scenario_config.get('num_agents_override', int(os.getenv('NUM_AGENTS', 10)))
        num_rounds = int(os.getenv('NUM_ROUNDS', 3))
        
        cached_result = cache.get_cached_result(
            scenario_name=scenario_name,
            num_agents=num_agents,
            num_rounds=num_rounds,
            model_distribution=scenario_config['model_distribution']
        )
        
        if cached_result:
            print(f"✨ Using cached result for {scenario_name}")
            return cached_result
    
    # Run new experiment
    runner = ExperimentRunner(scenario_name=scenario_name)
    
    try:
        result = await runner.run_experiment()
        
        # Extract key metrics
        metrics = {
            'scenario': scenario_name,
            'experiment_id': result.experiment_id,
            'total_rounds': result.total_rounds,
            'total_games': result.total_games,
            'total_cost': result.total_cost,
            'cooperation_rates': [],
            'average_cooperation': 0.0,
            'final_cooperation': 0.0,
            'convergence_rate': 0.0,
            'acausal_indicators': result.acausal_indicators
        }
        
        # Calculate cooperation metrics
        for summary in result.round_summaries:
            metrics['cooperation_rates'].append(summary.cooperation_rate)
        
        if metrics['cooperation_rates']:
            metrics['average_cooperation'] = sum(metrics['cooperation_rates']) / len(metrics['cooperation_rates'])
            metrics['final_cooperation'] = metrics['cooperation_rates'][-1]
            
            # Calculate convergence (how quickly cooperation stabilizes)
            if len(metrics['cooperation_rates']) > 1:
                differences = [abs(metrics['cooperation_rates'][i] - metrics['cooperation_rates'][i-1]) 
                              for i in range(1, len(metrics['cooperation_rates']))]
                metrics['convergence_rate'] = 1.0 - (sum(differences) / len(differences))
        
        print(f"✅ Scenario {scenario_name} completed")
        print(f"   Average cooperation: {metrics['average_cooperation']:.1%}")
        print(f"   Final cooperation: {metrics['final_cooperation']:.1%}")
        
        # Cache the result if caching is enabled
        if use_cache:
            cache = ExperimentCache()
            num_agents = scenario_config.get('num_agents_override', int(os.getenv('NUM_AGENTS', 10)))
            num_rounds = int(os.getenv('NUM_ROUNDS', 3))
            
            cache.save_result(
                scenario_name=scenario_name,
                num_agents=num_agents,
                num_rounds=num_rounds,
                model_distribution=scenario_config['model_distribution'],
                result=metrics,
                experiment_id=metrics['experiment_id'],
                cost=metrics.get('total_cost', 0)
            )
        
        return metrics
        
    except Exception as e:
        print(f"❌ Scenario {scenario_name} failed: {e}")
        return {
            'scenario': scenario_name,
            'error': str(e),
            'average_cooperation': 0.0,
            'final_cooperation': 0.0
        }


async def run_all_scenarios(scenarios_to_run: List[str] = None, use_cache: bool = True) -> Dict[str, Any]:
    """Run experiments for all or selected scenarios.
    
    Args:
        scenarios_to_run: Optional list of scenario names to run. If None, runs all.
        
    Returns:
        Combined results from all experiments
    """
    # Load scenarios
    scenarios_path = Path("scenarios.json")
    if not scenarios_path.exists():
        print("❌ scenarios.json not found")
        return {}
    
    with open(scenarios_path, 'r') as f:
        data = json.load(f)
    
    scenarios = data.get('scenarios', [])
    
    # Filter scenarios if specific ones requested
    if scenarios_to_run:
        scenarios = [s for s in scenarios if s['name'] in scenarios_to_run]
    
    if not scenarios:
        print("❌ No scenarios to run")
        return {}
    
    print(f"\n🎯 Planning to run {len(scenarios)} scenarios:")
    for s in scenarios:
        print(f"   - {s['name']}: {s['description']}")
    
    # Run each scenario
    results = {}
    for scenario in scenarios:
        scenario_name = scenario['name']
        
        # Check for num_agents_override
        if 'num_agents_override' in scenario:
            os.environ['NUM_AGENTS'] = str(scenario['num_agents_override'])
        
        result = await run_scenario_experiment(scenario_name, use_cache=use_cache)
        results[scenario_name] = result
        
        # Reset NUM_AGENTS if it was overridden
        if 'num_agents_override' in scenario:
            os.environ['NUM_AGENTS'] = '4'  # Reset to default
        
        # Small delay between experiments to avoid rate limiting
        await asyncio.sleep(5)
    
    return results


def save_comparison_results(results: Dict[str, Any]):
    """Save comparison results to file.
    
    Args:
        results: Dictionary of experiment results by scenario
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path('results') / 'model_comparisons'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / f'comparison_{timestamp}.json'
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n📊 Results saved to: {output_file}")
    
    # Also create a summary CSV for easy analysis
    csv_file = output_dir / f'comparison_{timestamp}.csv'
    with open(csv_file, 'w') as f:
        f.write("Scenario,Avg Cooperation,Final Cooperation,Convergence,Cost\n")
        for scenario, data in results.items():
            if 'error' not in data:
                f.write(f"{scenario},{data['average_cooperation']:.3f},"
                       f"{data['final_cooperation']:.3f},"
                       f"{data.get('convergence_rate', 0):.3f},"
                       f"{data.get('total_cost', 0):.4f}\n")
    
    print(f"📈 Summary CSV saved to: {csv_file}")


def print_comparison_summary(results: Dict[str, Any]):
    """Print a summary comparison of all results.
    
    Args:
        results: Dictionary of experiment results by scenario
    """
    print("\n" + "="*80)
    print("📊 MODEL COMPARISON SUMMARY")
    print("="*80)
    
    # Sort by average cooperation rate
    sorted_results = sorted(results.items(), 
                           key=lambda x: x[1].get('average_cooperation', 0), 
                           reverse=True)
    
    print(f"\n{'Scenario':<30} {'Avg Coop':<12} {'Final Coop':<12} {'Convergence':<12}")
    print("-"*66)
    
    for scenario, data in sorted_results:
        if 'error' not in data:
            print(f"{scenario:<30} "
                  f"{data['average_cooperation']:>10.1%}  "
                  f"{data['final_cooperation']:>10.1%}  "
                  f"{data.get('convergence_rate', 0):>10.1%}")
        else:
            print(f"{scenario:<30} ERROR: {data['error'][:30]}")
    
    # Find best performers
    if sorted_results:
        print("\n🏆 TOP PERFORMERS:")
        print(f"   Highest Average Cooperation: {sorted_results[0][0]} "
              f"({sorted_results[0][1]['average_cooperation']:.1%})")
        
        # Find most stable (highest convergence)
        most_stable = max(results.items(), 
                         key=lambda x: x[1].get('convergence_rate', 0))
        print(f"   Most Stable (Convergence): {most_stable[0]} "
              f"({most_stable[1].get('convergence_rate', 0):.1%})")
    
    # Calculate total cost
    total_cost = sum(r.get('total_cost', 0) for r in results.values())
    print(f"\n💰 Total Experiment Cost: ${total_cost:.4f}")


async def main():
    """Main entry point for model comparison experiments."""
    parser = argparse.ArgumentParser(description="Run model comparison experiments")
    parser.add_argument(
        '--scenarios',
        nargs='+',
        help='Specific scenarios to run (e.g., baseline_gemini mixed_gemini_gpt)'
    )
    parser.add_argument(
        '--test',
        action='store_true',
        help='Run only the test_small scenario for quick testing'
    )
    parser.add_argument(
        '--no-cache',
        action='store_true',
        help='Disable cache and force re-running experiments'
    )
    parser.add_argument(
        '--cache-stats',
        action='store_true',
        help='Show cache statistics and exit'
    )
    parser.add_argument(
        '--clear-cache',
        action='store_true',
        help='Clear all cached results'
    )
    
    args = parser.parse_args()
    
    # Handle cache operations
    if args.cache_stats:
        from src.utils.experiment_cache import ExperimentCache
        cache = ExperimentCache()
        stats = cache.get_cache_stats()
        print("\n📊 CACHE STATISTICS")
        print("="*60)
        print(f"Total cached experiments: {stats['total_cached']}")
        print(f"Total cost saved: ${stats['total_cost_saved']:.4f}")
        print(f"Cache size: {stats['cache_size_mb']:.2f} MB")
        print("\nCached scenarios:")
        for scenario, info in stats['scenarios'].items():
            print(f"  {scenario}: {info['count']} experiments, ${info['cost_saved']:.4f} saved")
        return 0
    
    if args.clear_cache:
        from src.utils.experiment_cache import ExperimentCache
        cache = ExperimentCache()
        cache.clear_cache()
        print("✅ Cache cleared")
        return 0
    
    # Determine which scenarios to run
    if args.test:
        scenarios_to_run = ['test_small']
        print("🧪 Running in TEST mode (small scenario only)")
    else:
        scenarios_to_run = args.scenarios
    
    use_cache = not args.no_cache
    if args.no_cache:
        print("⚠️ Cache disabled - will re-run all experiments")
    
    print("\n🚀 ULTRATHINK Model Comparison Experiment")
    print("="*60)
    
    # Check for API key
    if not os.getenv("OPENROUTER_API_KEY"):
        print("❌ OPENROUTER_API_KEY not set")
        return 1
    
    # Run experiments
    results = await run_all_scenarios(scenarios_to_run, use_cache=use_cache)
    
    if results:
        # Save results
        save_comparison_results(results)
        
        # Print summary
        print_comparison_summary(results)
        
        print("\n✅ Model comparison complete!")
        return 0
    else:
        print("\n❌ No experiments completed successfully")
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))