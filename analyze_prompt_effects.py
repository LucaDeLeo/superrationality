#!/usr/bin/env python3
"""Analyze the effects of different prompt variations on cooperation rates."""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np
from datetime import datetime


class PromptEffectAnalyzer:
    """Analyze effects of prompt variations on cooperation."""
    
    def __init__(self):
        """Initialize the analyzer."""
        self.results = []
    
    def load_results(self, filepath: str) -> bool:
        """Load results from a JSON file.
        
        Args:
            filepath: Path to results file
            
        Returns:
            True if successful
        """
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
                if 'experiments' in data:
                    self.results = data['experiments']
                else:
                    self.results = [data]
            return True
        except Exception as e:
            print(f"Error loading results: {e}")
            return False
    
    def analyze_bias_effects(self) -> Dict[str, Any]:
        """Analyze the individual effects of each bias.
        
        Returns:
            Analysis of bias effects
        """
        if not self.results:
            return {"error": "No results loaded"}
        
        # Find baseline (no biases)
        baseline = self._find_experiment('baseline_control')
        if not baseline:
            print("Warning: No baseline control found")
            baseline_rate = 50.0  # Theoretical baseline
        else:
            baseline_rate = baseline['cooperation_rate']
        
        # Analyze each bias effect
        effects = {
            'baseline_rate': baseline_rate,
            'individual_effects': {},
            'combined_effects': {},
            'interaction_effects': {}
        }
        
        # Individual effects
        identity_only = self._find_experiment('identity_only')
        if identity_only:
            effects['individual_effects']['identity'] = {
                'cooperation_rate': identity_only['cooperation_rate'],
                'effect': identity_only['cooperation_rate'] - baseline_rate,
                'percent_change': self._percent_change(baseline_rate, identity_only['cooperation_rate'])
            }
        
        coop_rates_only = self._find_experiment('cooperation_rates_only')
        if coop_rates_only:
            effects['individual_effects']['global_cooperation'] = {
                'cooperation_rate': coop_rates_only['cooperation_rate'],
                'effect': coop_rates_only['cooperation_rate'] - baseline_rate,
                'percent_change': self._percent_change(baseline_rate, coop_rates_only['cooperation_rate'])
            }
        
        # Combined effect (all biases)
        original = self._find_experiment('original_biased')
        if original:
            effects['combined_effects']['all_biases'] = {
                'cooperation_rate': original['cooperation_rate'],
                'effect': original['cooperation_rate'] - baseline_rate,
                'percent_change': self._percent_change(baseline_rate, original['cooperation_rate'])
            }
            
            # Calculate interaction effects
            if identity_only and coop_rates_only:
                expected_additive = baseline_rate + \
                    (identity_only['cooperation_rate'] - baseline_rate) + \
                    (coop_rates_only['cooperation_rate'] - baseline_rate)
                
                effects['interaction_effects']['identity_x_cooperation'] = {
                    'expected_if_additive': min(100, expected_additive),
                    'actual_combined': original['cooperation_rate'],
                    'interaction': original['cooperation_rate'] - min(100, expected_additive),
                    'type': 'synergistic' if original['cooperation_rate'] > expected_additive else 'antagonistic'
                }
        
        return effects
    
    def analyze_framing_effects(self) -> Dict[str, Any]:
        """Analyze how different framings affect cooperation.
        
        Returns:
            Analysis of framing effects
        """
        framing_experiments = [
            'baseline_control',
            'anti_cooperation',
            'veil_of_ignorance',
            'game_theory_framing',
            'trust_building',
            'superrational_hint'
        ]
        
        results = {
            'framings': {},
            'summary': {}
        }
        
        for exp_id in framing_experiments:
            exp = self._find_experiment(exp_id)
            if exp:
                results['framings'][exp_id] = {
                    'name': exp.get('experiment_name', exp_id),
                    'cooperation_rate': exp['cooperation_rate'],
                    'convergence': exp.get('convergence_score', 0)
                }
        
        # Calculate summary statistics
        if results['framings']:
            rates = [f['cooperation_rate'] for f in results['framings'].values()]
            results['summary'] = {
                'mean_cooperation': np.mean(rates),
                'std_cooperation': np.std(rates),
                'range': max(rates) - min(rates),
                'most_cooperative': max(results['framings'].items(), key=lambda x: x[1]['cooperation_rate'])[0],
                'least_cooperative': min(results['framings'].items(), key=lambda x: x[1]['cooperation_rate'])[0]
            }
        
        return results
    
    def generate_report(self) -> str:
        """Generate a comprehensive report of prompt effects.
        
        Returns:
            Formatted report string
        """
        report = []
        report.append("=" * 80)
        report.append("PROMPT VARIATION EFFECTS ON COOPERATION - ANALYSIS REPORT")
        report.append("=" * 80)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Bias effects section
        bias_effects = self.analyze_bias_effects()
        report.append("\n📊 BIAS EFFECTS ANALYSIS")
        report.append("-" * 40)
        report.append(f"Baseline cooperation rate: {bias_effects['baseline_rate']:.1f}%")
        report.append("")
        
        if bias_effects['individual_effects']:
            report.append("Individual Bias Effects:")
            for bias, data in bias_effects['individual_effects'].items():
                report.append(f"  • {bias.replace('_', ' ').title()}:")
                report.append(f"    - Cooperation: {data['cooperation_rate']:.1f}%")
                report.append(f"    - Effect: {data['effect']:+.1f}%")
                report.append(f"    - Change: {data['percent_change']:+.1f}%")
        
        if bias_effects['combined_effects']:
            report.append("\nCombined Effects:")
            for combo, data in bias_effects['combined_effects'].items():
                report.append(f"  • {combo.replace('_', ' ').title()}:")
                report.append(f"    - Cooperation: {data['cooperation_rate']:.1f}%")
                report.append(f"    - Total effect: {data['effect']:+.1f}%")
        
        if bias_effects['interaction_effects']:
            report.append("\nInteraction Effects:")
            for interaction, data in bias_effects['interaction_effects'].items():
                report.append(f"  • {interaction.replace('_', ' ').title()}:")
                report.append(f"    - Expected (if additive): {data['expected_if_additive']:.1f}%")
                report.append(f"    - Actual: {data['actual_combined']:.1f}%")
                report.append(f"    - Interaction: {data['interaction']:+.1f}% ({data['type']})")
        
        # Framing effects section
        framing_effects = self.analyze_framing_effects()
        if framing_effects['framings']:
            report.append("\n\n🎭 FRAMING EFFECTS ANALYSIS")
            report.append("-" * 40)
            
            # Sort by cooperation rate
            sorted_framings = sorted(
                framing_effects['framings'].items(),
                key=lambda x: x[1]['cooperation_rate'],
                reverse=True
            )
            
            report.append("Framing Rankings (by cooperation rate):")
            for i, (exp_id, data) in enumerate(sorted_framings, 1):
                report.append(f"  {i}. {data['name']}: {data['cooperation_rate']:.1f}%")
            
            if framing_effects['summary']:
                summary = framing_effects['summary']
                report.append("\nFraming Statistics:")
                report.append(f"  • Mean cooperation: {summary['mean_cooperation']:.1f}%")
                report.append(f"  • Std deviation: {summary['std_cooperation']:.1f}%")
                report.append(f"  • Range: {summary['range']:.1f}%")
        
        # Key findings section
        report.append("\n\n🔍 KEY FINDINGS")
        report.append("-" * 40)
        
        findings = self._generate_key_findings(bias_effects, framing_effects)
        for i, finding in enumerate(findings, 1):
            report.append(f"{i}. {finding}")
        
        # Recommendations
        report.append("\n\n💡 RECOMMENDATIONS")
        report.append("-" * 40)
        recommendations = self._generate_recommendations(bias_effects, framing_effects)
        for i, rec in enumerate(recommendations, 1):
            report.append(f"{i}. {rec}")
        
        report.append("\n" + "=" * 80)
        
        return "\n".join(report)
    
    def _find_experiment(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """Find an experiment by ID in loaded results.
        
        Args:
            experiment_id: ID to search for
            
        Returns:
            Experiment data or None
        """
        for exp in self.results:
            if exp.get('experiment_id') == experiment_id:
                return exp
        return None
    
    def _percent_change(self, baseline: float, value: float) -> float:
        """Calculate percent change from baseline.
        
        Args:
            baseline: Baseline value
            value: New value
            
        Returns:
            Percent change
        """
        if baseline == 0:
            return 0 if value == 0 else 100
        return ((value - baseline) / baseline) * 100
    
    def _generate_key_findings(self, bias_effects: Dict, framing_effects: Dict) -> List[str]:
        """Generate key findings from the analysis.
        
        Args:
            bias_effects: Bias effects analysis
            framing_effects: Framing effects analysis
            
        Returns:
            List of key findings
        """
        findings = []
        
        # Check identity effect
        if 'identity' in bias_effects.get('individual_effects', {}):
            identity_effect = bias_effects['individual_effects']['identity']['effect']
            if abs(identity_effect) > 10:
                findings.append(
                    f"Identity information {'increases' if identity_effect > 0 else 'decreases'} "
                    f"cooperation by {abs(identity_effect):.1f}%"
                )
        
        # Check global cooperation effect
        if 'global_cooperation' in bias_effects.get('individual_effects', {}):
            global_effect = bias_effects['individual_effects']['global_cooperation']['effect']
            if abs(global_effect) > 10:
                findings.append(
                    f"Sharing global cooperation rates {'increases' if global_effect > 0 else 'decreases'} "
                    f"cooperation by {abs(global_effect):.1f}%"
                )
        
        # Check interaction effects
        if 'identity_x_cooperation' in bias_effects.get('interaction_effects', {}):
            interaction = bias_effects['interaction_effects']['identity_x_cooperation']
            if abs(interaction['interaction']) > 5:
                findings.append(
                    f"Biases show {interaction['type']} interaction effects "
                    f"({interaction['interaction']:+.1f}% from expected)"
                )
        
        # Check framing range
        if framing_effects.get('summary', {}).get('range', 0) > 20:
            findings.append(
                f"Framing has substantial impact: {framing_effects['summary']['range']:.1f}% "
                f"cooperation rate difference between framings"
            )
        
        # Check if competition framing reduces cooperation
        competition = self._find_experiment('anti_cooperation')
        baseline = self._find_experiment('baseline_control')
        if competition and baseline:
            if competition['cooperation_rate'] < baseline['cooperation_rate'] - 10:
                findings.append(
                    "Competitive framing significantly reduces cooperation "
                    f"({competition['cooperation_rate']:.1f}% vs {baseline['cooperation_rate']:.1f}% baseline)"
                )
        
        return findings
    
    def _generate_recommendations(self, bias_effects: Dict, framing_effects: Dict) -> List[str]:
        """Generate recommendations based on analysis.
        
        Args:
            bias_effects: Bias effects analysis
            framing_effects: Framing effects analysis
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        # Check if current setup is biased
        if 'all_biases' in bias_effects.get('combined_effects', {}):
            combined_effect = bias_effects['combined_effects']['all_biases']['effect']
            if combined_effect > 30:
                recommendations.append(
                    "Current experimental setup shows strong bias toward cooperation. "
                    "Consider using baseline_control or identity_only configurations for unbiased testing."
                )
        
        # Recommend best configuration for testing superrationality
        recommendations.append(
            "For testing genuine superrational cooperation: Use 'identity_only' or 'implicit_identity' "
            "configurations to avoid confounding factors from global information sharing."
        )
        
        # Recommend control experiments
        recommendations.append(
            "Always run control experiments (baseline_control) alongside treatment conditions "
            "to measure the true effect of identity recognition."
        )
        
        # Framing recommendations
        if framing_effects.get('summary', {}).get('range', 0) > 20:
            recommendations.append(
                "Framing significantly affects results. Use neutral framing (baseline_control) "
                "or explicitly test multiple framings to understand robustness."
            )
        
        # Default behavior recommendations
        recommendations.append(
            "Use random defaults for ambiguous responses to avoid systematic bias. "
            "Current 'cooperate' default inflates cooperation rates."
        )
        
        return recommendations
    
    def export_comparison_table(self) -> str:
        """Export a comparison table of all experiments.
        
        Returns:
            CSV-formatted comparison table
        """
        if not self.results:
            return "No results loaded"
        
        # Create CSV header
        lines = ["Experiment,Cooperation Rate,Identity Info,Global Cooperation,Round Summaries,Default Action,Convergence"]
        
        # Add each experiment
        for exp in self.results:
            config = exp.get('config', {})
            lines.append(','.join([
                exp.get('experiment_name', exp.get('experiment_id', 'Unknown')),
                f"{exp.get('cooperation_rate', 0):.1f}%",
                'Yes' if config.get('include_identity') else 'No',
                'Yes' if config.get('include_global_cooperation') else 'No',
                'Yes' if config.get('include_round_summaries') else 'No',
                config.get('default_on_ambiguity', 'unknown'),
                f"{exp.get('convergence_score', 0):.2f}"
            ]))
        
        return '\n'.join(lines)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Analyze prompt experiment effects')
    parser.add_argument(
        'results_file',
        type=str,
        help='Path to results JSON file'
    )
    parser.add_argument(
        '--save',
        action='store_true',
        help='Save analysis report to file'
    )
    parser.add_argument(
        '--csv',
        action='store_true',
        help='Export comparison table as CSV'
    )
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = PromptEffectAnalyzer()
    
    # Load results
    if not analyzer.load_results(args.results_file):
        print("Failed to load results")
        return
    
    # Generate and display report
    report = analyzer.generate_report()
    print(report)
    
    # Save report if requested
    if args.save:
        filename = f"prompt_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(filename, 'w') as f:
            f.write(report)
        print(f"\n📄 Report saved to: {filename}")
    
    # Export CSV if requested
    if args.csv:
        csv_data = analyzer.export_comparison_table()
        csv_filename = f"prompt_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        with open(csv_filename, 'w') as f:
            f.write(csv_data)
        print(f"\n📊 CSV table saved to: {csv_filename}")


if __name__ == "__main__":
    main()