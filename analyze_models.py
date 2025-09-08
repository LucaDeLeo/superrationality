#!/usr/bin/env python3
"""Analyze and compare results from multi-model experiments."""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Tuple
import statistics
from collections import defaultdict
from datetime import datetime


class ModelAnalyzer:
    """Analyze model behavior in acausal cooperation experiments."""
    
    def __init__(self, results_file: Path = None):
        """Initialize analyzer with results file.
        
        Args:
            results_file: Path to comparison results JSON file
        """
        self.results = {}
        if results_file and results_file.exists():
            with open(results_file, 'r') as f:
                self.results = json.load(f)
    
    def load_multiple_experiments(self, experiment_dirs: List[Path]) -> Dict[str, List[Any]]:
        """Load results from multiple experiment directories.
        
        Args:
            experiment_dirs: List of experiment directory paths
            
        Returns:
            Dictionary mapping scenario names to lists of results
        """
        combined_results = defaultdict(list)
        
        for exp_dir in experiment_dirs:
            results_file = exp_dir / 'experiment_results.json'
            if results_file.exists():
                with open(results_file, 'r') as f:
                    data = json.load(f)
                    # Extract scenario name from experiment metadata
                    scenario = data.get('scenario_name', 'unknown')
                    combined_results[scenario].append(data)
        
        return combined_results
    
    def analyze_cooperation_by_model(self) -> Dict[str, Dict[str, float]]:
        """Analyze cooperation rates for each model type.
        
        Returns:
            Dictionary mapping model names to cooperation statistics
        """
        model_stats = defaultdict(lambda: {'cooperation_rates': [], 'scenarios': []})
        
        for scenario_name, result in self.results.items():
            if 'error' in result:
                continue
            
            # Parse scenario name to identify model types
            if 'homogeneous' in scenario_name:
                # Extract model name from scenario
                if 'gemini' in scenario_name:
                    model = 'google/gemini-2.5-flash'
                elif 'gpt4o' in scenario_name:
                    model = 'openai/gpt-4o'
                elif 'claude' in scenario_name:
                    model = 'anthropic/claude-3-sonnet'
                elif 'llama' in scenario_name:
                    model = 'meta-llama/llama-3-70b-instruct'
                else:
                    continue
                
                model_stats[model]['cooperation_rates'].append(result['average_cooperation'])
                model_stats[model]['scenarios'].append(scenario_name)
        
        # Calculate statistics for each model
        analysis = {}
        for model, data in model_stats.items():
            if data['cooperation_rates']:
                analysis[model] = {
                    'mean_cooperation': statistics.mean(data['cooperation_rates']),
                    'std_cooperation': statistics.stdev(data['cooperation_rates']) 
                                      if len(data['cooperation_rates']) > 1 else 0,
                    'scenarios': data['scenarios']
                }
        
        return analysis
    
    def analyze_mixed_scenarios(self) -> Dict[str, Any]:
        """Analyze behavior in mixed-model scenarios.
        
        Returns:
            Analysis of cross-model interactions
        """
        mixed_results = {}
        
        for scenario_name, result in self.results.items():
            if 'mixed' in scenario_name and 'error' not in result:
                mixed_results[scenario_name] = {
                    'cooperation_rate': result['average_cooperation'],
                    'final_cooperation': result['final_cooperation'],
                    'convergence': result.get('convergence_rate', 0)
                }
        
        if mixed_results:
            # Compare mixed vs homogeneous
            avg_mixed = statistics.mean([r['cooperation_rate'] for r in mixed_results.values()])
            
            # Get average of homogeneous scenarios
            homogeneous = [r['average_cooperation'] for s, r in self.results.items() 
                          if 'homogeneous' in s and 'error' not in r]
            avg_homogeneous = statistics.mean(homogeneous) if homogeneous else 0
            
            return {
                'mixed_scenarios': mixed_results,
                'average_mixed_cooperation': avg_mixed,
                'average_homogeneous_cooperation': avg_homogeneous,
                'diversity_impact': avg_mixed - avg_homogeneous
            }
        
        return {}
    
    def analyze_convergence_patterns(self) -> Dict[str, Any]:
        """Analyze how quickly different models converge to stable cooperation.
        
        Returns:
            Convergence analysis by model/scenario
        """
        convergence_data = {}
        
        for scenario_name, result in self.results.items():
            if 'error' in result and 'cooperation_rates' in result:
                rates = result['cooperation_rates']
                if len(rates) > 1:
                    # Calculate variance reduction over rounds
                    early_variance = statistics.variance(rates[:len(rates)//2]) if len(rates) > 2 else 0
                    late_variance = statistics.variance(rates[len(rates)//2:]) if len(rates) > 2 else 0
                    
                    convergence_data[scenario_name] = {
                        'convergence_rate': result.get('convergence_rate', 0),
                        'variance_reduction': (early_variance - late_variance) / early_variance 
                                            if early_variance > 0 else 0,
                        'final_stability': 1.0 - late_variance if late_variance < 1 else 0
                    }
        
        return convergence_data
    
    def identify_best_configurations(self) -> Dict[str, Any]:
        """Identify the best model configurations for cooperation.
        
        Returns:
            Recommendations for optimal model configurations
        """
        rankings = []
        
        for scenario_name, result in self.results.items():
            if 'error' not in result:
                # Calculate composite score
                score = (
                    result['average_cooperation'] * 0.5 +  # Weight cooperation heavily
                    result['final_cooperation'] * 0.3 +    # Final state matters
                    result.get('convergence_rate', 0) * 0.2  # Stability is good
                )
                
                rankings.append({
                    'scenario': scenario_name,
                    'score': score,
                    'cooperation': result['average_cooperation'],
                    'convergence': result.get('convergence_rate', 0)
                })
        
        # Sort by score
        rankings.sort(key=lambda x: x['score'], reverse=True)
        
        return {
            'top_configurations': rankings[:5] if len(rankings) >= 5 else rankings,
            'best_overall': rankings[0] if rankings else None,
            'most_cooperative': max(rankings, key=lambda x: x['cooperation']) if rankings else None,
            'most_stable': max(rankings, key=lambda x: x['convergence']) if rankings else None
        }
    
    def generate_report(self) -> str:
        """Generate a comprehensive analysis report.
        
        Returns:
            Formatted report string
        """
        report = []
        report.append("="*80)
        report.append("ULTRATHINK MODEL COMPARISON ANALYSIS REPORT")
        report.append("="*80)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Model cooperation analysis
        report.append("## COOPERATION RATES BY MODEL")
        report.append("-"*40)
        model_analysis = self.analyze_cooperation_by_model()
        for model, stats in sorted(model_analysis.items(), 
                                   key=lambda x: x[1]['mean_cooperation'], 
                                   reverse=True):
            report.append(f"{model}:")
            report.append(f"  Mean cooperation: {stats['mean_cooperation']:.1%}")
            report.append(f"  Std deviation: {stats['std_cooperation']:.3f}")
            report.append("")
        
        # Mixed scenario analysis
        report.append("## MIXED MODEL SCENARIOS")
        report.append("-"*40)
        mixed_analysis = self.analyze_mixed_scenarios()
        if mixed_analysis:
            report.append(f"Average mixed cooperation: {mixed_analysis['average_mixed_cooperation']:.1%}")
            report.append(f"Average homogeneous cooperation: {mixed_analysis['average_homogeneous_cooperation']:.1%}")
            report.append(f"Diversity impact: {mixed_analysis['diversity_impact']:+.1%}")
            report.append("")
            
            for scenario, data in mixed_analysis['mixed_scenarios'].items():
                report.append(f"{scenario}: {data['cooperation_rate']:.1%} cooperation")
        else:
            report.append("No mixed scenarios found")
        report.append("")
        
        # Best configurations
        report.append("## OPTIMAL CONFIGURATIONS")
        report.append("-"*40)
        best = self.identify_best_configurations()
        if best['best_overall']:
            report.append(f"Best Overall: {best['best_overall']['scenario']}")
            report.append(f"  Score: {best['best_overall']['score']:.3f}")
            report.append(f"  Cooperation: {best['best_overall']['cooperation']:.1%}")
            report.append("")
        
        if best['most_cooperative']:
            report.append(f"Most Cooperative: {best['most_cooperative']['scenario']}")
            report.append(f"  Cooperation: {best['most_cooperative']['cooperation']:.1%}")
            report.append("")
        
        if best['most_stable']:
            report.append(f"Most Stable: {best['most_stable']['scenario']}")
            report.append(f"  Convergence: {best['most_stable']['convergence']:.1%}")
        
        report.append("")
        report.append("## KEY FINDINGS")
        report.append("-"*40)
        
        # Determine key findings
        if model_analysis:
            best_model = max(model_analysis.items(), key=lambda x: x[1]['mean_cooperation'])
            report.append(f"1. {best_model[0].split('/')[-1]} shows highest cooperation ({best_model[1]['mean_cooperation']:.1%})")
        
        if mixed_analysis and 'diversity_impact' in mixed_analysis:
            if mixed_analysis['diversity_impact'] > 0:
                report.append(f"2. Model diversity INCREASES cooperation by {mixed_analysis['diversity_impact']:.1%}")
            else:
                report.append(f"2. Model diversity DECREASES cooperation by {abs(mixed_analysis['diversity_impact']):.1%}")
        
        # Check for superrational outcomes
        superrational = [s for s, r in self.results.items() 
                        if 'error' not in r and r['average_cooperation'] > 0.9]
        if superrational:
            report.append(f"3. {len(superrational)} scenarios achieved superrational cooperation (>90%)")
        
        report.append("")
        report.append("="*80)
        
        return "\n".join(report)
    
    def save_report(self, output_path: Path = None):
        """Save analysis report to file.
        
        Args:
            output_path: Path to save report (default: results/model_comparisons/analysis_<timestamp>.txt)
        """
        if output_path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_dir = Path('results') / 'model_comparisons'
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f'analysis_{timestamp}.txt'
        
        report = self.generate_report()
        with open(output_path, 'w') as f:
            f.write(report)
        
        print(f"📄 Report saved to: {output_path}")


def main():
    """Main entry point for model analysis."""
    parser = argparse.ArgumentParser(description="Analyze model comparison results")
    parser.add_argument(
        'results_file',
        nargs='?',
        help='Path to comparison results JSON file'
    )
    parser.add_argument(
        '--latest',
        action='store_true',
        help='Use the latest comparison results file'
    )
    parser.add_argument(
        '--save',
        action='store_true',
        help='Save analysis report to file'
    )
    
    args = parser.parse_args()
    
    # Find results file
    if args.latest or not args.results_file:
        # Find latest comparison file
        comparison_dir = Path('results') / 'model_comparisons'
        if comparison_dir.exists():
            comparison_files = sorted(comparison_dir.glob('comparison_*.json'))
            if comparison_files:
                results_file = comparison_files[-1]
                print(f"Using latest results: {results_file}")
            else:
                print("❌ No comparison results found")
                return 1
        else:
            print("❌ No model_comparisons directory found")
            return 1
    else:
        results_file = Path(args.results_file)
        if not results_file.exists():
            print(f"❌ Results file not found: {results_file}")
            return 1
    
    # Run analysis
    analyzer = ModelAnalyzer(results_file)
    
    # Print report
    print(analyzer.generate_report())
    
    # Save if requested
    if args.save:
        analyzer.save_report()
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())