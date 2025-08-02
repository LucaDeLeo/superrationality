"""
Utility for comparing results across different scenarios.
Enables cross-scenario analysis and reporting.
"""
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
from collections import defaultdict
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class ScenarioResult:
    """Container for scenario-specific results."""
    scenario_name: str
    experiment_id: str
    model_distribution: Dict[str, int]
    diversity_score: float
    overall_cooperation_rate: float
    cooperation_trend: float
    final_round_cooperation: float
    coalition_count: int
    cross_model_cooperation: Optional[float] = None
    same_model_cooperation: Optional[float] = None
    path: Optional[Path] = None


class ScenarioComparator:
    """Compares and analyzes results across multiple scenarios."""
    
    def __init__(self, base_path: str = "results"):
        """
        Initialize comparator.
        
        Args:
            base_path: Base directory containing scenario results
        """
        self.base_path = Path(base_path)
        self.scenarios_path = self.base_path / "scenarios"
        self.scenario_results: Dict[str, List[ScenarioResult]] = {}
        
    def load_scenario_results(self, scenario_name: Optional[str] = None) -> Dict[str, List[ScenarioResult]]:
        """
        Load results for one or all scenarios.
        
        Args:
            scenario_name: Specific scenario to load, or None for all
            
        Returns:
            Dictionary mapping scenario names to lists of results
        """
        if scenario_name:
            # Load specific scenario
            scenario_path = self.scenarios_path / scenario_name
            if scenario_path.exists():
                self.scenario_results[scenario_name] = self._load_scenario_experiments(scenario_path)
        else:
            # Load all scenarios
            if self.scenarios_path.exists():
                for scenario_dir in self.scenarios_path.iterdir():
                    if scenario_dir.is_dir():
                        scenario_name = scenario_dir.name
                        self.scenario_results[scenario_name] = self._load_scenario_experiments(scenario_dir)
        
        return self.scenario_results
    
    def _load_scenario_experiments(self, scenario_path: Path) -> List[ScenarioResult]:
        """Load all experiments for a scenario."""
        results = []
        
        for exp_dir in scenario_path.iterdir():
            if exp_dir.is_dir() and exp_dir.name.startswith("exp_"):
                try:
                    result = self._load_experiment_result(exp_dir)
                    if result:
                        results.append(result)
                except Exception as e:
                    logger.error(f"Failed to load experiment {exp_dir}: {e}")
        
        return results
    
    def _load_experiment_result(self, exp_path: Path) -> Optional[ScenarioResult]:
        """Load a single experiment result."""
        # Look for experiment results file
        result_files = list(exp_path.glob("experiment_results*.json"))
        if not result_files:
            return None
        
        with open(result_files[0], 'r') as f:
            data = json.load(f)
        
        # Load scenario config if available
        scenario_config_path = exp_path / "scenario_config.json"
        if scenario_config_path.exists():
            with open(scenario_config_path, 'r') as f:
                scenario_data = json.load(f)
        else:
            scenario_data = {}
        
        # Extract key metrics
        round_summaries = data.get("round_summaries", [])
        if not round_summaries:
            return None
        
        cooperation_rates = [r["cooperation_rate"] for r in round_summaries]
        
        # Calculate cooperation trend
        if len(cooperation_rates) > 1:
            rounds = np.arange(len(cooperation_rates))
            trend = np.polyfit(rounds, cooperation_rates, 1)[0]
        else:
            trend = 0.0
        
        # Extract coalition data if available
        coalition_count = 0
        # This would come from analysis results
        
        return ScenarioResult(
            scenario_name=data.get("scenario_name", scenario_data.get("scenario_name", "unknown")),
            experiment_id=data["experiment_id"],
            model_distribution=scenario_data.get("model_distribution", {}),
            diversity_score=scenario_data.get("diversity_score", 0.0),
            overall_cooperation_rate=np.mean(cooperation_rates),
            cooperation_trend=trend,
            final_round_cooperation=cooperation_rates[-1] if cooperation_rates else 0.0,
            coalition_count=coalition_count,
            path=exp_path
        )
    
    def compare_scenarios(self, metric: str = "overall_cooperation_rate") -> Dict:
        """
        Compare scenarios by a specific metric.
        
        Args:
            metric: Metric to compare (e.g., 'overall_cooperation_rate', 'diversity_score')
            
        Returns:
            Comparison results
        """
        if not self.scenario_results:
            self.load_scenario_results()
        
        comparison = {
            "metric": metric,
            "scenario_averages": {},
            "scenario_std": {},
            "best_scenario": None,
            "worst_scenario": None,
            "rankings": []
        }
        
        # Calculate averages per scenario
        for scenario_name, results in self.scenario_results.items():
            if results:
                values = [getattr(r, metric) for r in results if hasattr(r, metric)]
                if values:
                    comparison["scenario_averages"][scenario_name] = np.mean(values)
                    comparison["scenario_std"][scenario_name] = np.std(values)
        
        # Rank scenarios
        if comparison["scenario_averages"]:
            sorted_scenarios = sorted(
                comparison["scenario_averages"].items(),
                key=lambda x: x[1],
                reverse=True
            )
            comparison["rankings"] = sorted_scenarios
            comparison["best_scenario"] = sorted_scenarios[0][0]
            comparison["worst_scenario"] = sorted_scenarios[-1][0]
        
        return comparison
    
    def analyze_diversity_impact(self) -> Dict:
        """
        Analyze how diversity affects cooperation across all scenarios.
        
        Returns:
            Dictionary with diversity impact analysis
        """
        if not self.scenario_results:
            self.load_scenario_results()
        
        # Collect diversity and cooperation pairs
        diversity_cooperation_pairs = []
        
        for scenario_name, results in self.scenario_results.items():
            for result in results:
                if result.diversity_score is not None:
                    diversity_cooperation_pairs.append(
                        (result.diversity_score, result.overall_cooperation_rate)
                    )
        
        if not diversity_cooperation_pairs:
            return {"error": "No diversity data available"}
        
        # Group by diversity level
        diversity_groups = defaultdict(list)
        for diversity, cooperation in diversity_cooperation_pairs:
            if diversity == 0:
                group = "homogeneous"
            elif diversity < 0.7:
                group = "low_diversity"
            elif diversity < 1.0:
                group = "moderate_diversity"
            else:
                group = "high_diversity"
            
            diversity_groups[group].append(cooperation)
        
        # Calculate statistics per group
        group_stats = {}
        for group, values in diversity_groups.items():
            if values:
                group_stats[group] = {
                    "mean": np.mean(values),
                    "std": np.std(values),
                    "count": len(values),
                    "min": min(values),
                    "max": max(values)
                }
        
        # Calculate overall correlation
        if len(diversity_cooperation_pairs) > 2:
            diversities, cooperations = zip(*diversity_cooperation_pairs)
            correlation = np.corrcoef(diversities, cooperations)[0, 1]
        else:
            correlation = None
        
        return {
            "total_experiments": len(diversity_cooperation_pairs),
            "diversity_groups": group_stats,
            "overall_correlation": correlation,
            "interpretation": self._interpret_diversity_impact(group_stats, correlation)
        }
    
    def generate_comparison_report(self, output_path: Optional[Path] = None) -> Dict:
        """
        Generate comprehensive comparison report across all scenarios.
        
        Args:
            output_path: Optional path to save report
            
        Returns:
            Dictionary with full comparison report
        """
        if not self.scenario_results:
            self.load_scenario_results()
        
        report = {
            "summary": {
                "total_scenarios": len(self.scenario_results),
                "total_experiments": sum(len(results) for results in self.scenario_results.values()),
                "scenarios_analyzed": list(self.scenario_results.keys())
            },
            "cooperation_comparison": self.compare_scenarios("overall_cooperation_rate"),
            "trend_comparison": self.compare_scenarios("cooperation_trend"),
            "diversity_impact": self.analyze_diversity_impact(),
            "scenario_profiles": self._generate_scenario_profiles(),
            "recommendations": self._generate_recommendations()
        }
        
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2)
            logger.info(f"Saved comparison report to {output_path}")
        
        return report
    
    def find_best_scenario_for_metric(self, target_metric: str, 
                                    target_value: float) -> Optional[str]:
        """
        Find scenario that best achieves a target metric value.
        
        Args:
            target_metric: Metric to optimize
            target_value: Target value to achieve
            
        Returns:
            Name of best matching scenario
        """
        if not self.scenario_results:
            self.load_scenario_results()
        
        best_scenario = None
        best_distance = float('inf')
        
        for scenario_name, results in self.scenario_results.items():
            if results:
                values = [getattr(r, target_metric) for r in results 
                         if hasattr(r, target_metric)]
                if values:
                    avg_value = np.mean(values)
                    distance = abs(avg_value - target_value)
                    
                    if distance < best_distance:
                        best_distance = distance
                        best_scenario = scenario_name
        
        return best_scenario
    
    def get_scenario_filtering_options(self) -> Dict[str, List]:
        """
        Get available filtering options for scenarios.
        
        Returns:
            Dictionary of filter options
        """
        if not self.scenario_results:
            self.load_scenario_results()
        
        # Collect unique values
        model_types = set()
        diversity_levels = set()
        
        for results in self.scenario_results.values():
            for result in results:
                # Collect model types
                for model in result.model_distribution.keys():
                    model_types.add(model)
                
                # Categorize diversity
                if result.diversity_score == 0:
                    diversity_levels.add("homogeneous")
                elif result.diversity_score < 0.7:
                    diversity_levels.add("low")
                elif result.diversity_score < 1.0:
                    diversity_levels.add("moderate")
                else:
                    diversity_levels.add("high")
        
        return {
            "scenarios": list(self.scenario_results.keys()),
            "model_types": sorted(model_types),
            "diversity_levels": sorted(diversity_levels),
            "metrics": ["overall_cooperation_rate", "cooperation_trend", 
                       "diversity_score", "coalition_count"]
        }
    
    def _generate_scenario_profiles(self) -> Dict:
        """Generate detailed profiles for each scenario."""
        profiles = {}
        
        for scenario_name, results in self.scenario_results.items():
            if not results:
                continue
            
            # Calculate aggregate statistics
            cooperation_rates = [r.overall_cooperation_rate for r in results]
            diversity_scores = [r.diversity_score for r in results]
            
            profiles[scenario_name] = {
                "experiment_count": len(results),
                "model_distribution": results[0].model_distribution if results else {},
                "cooperation": {
                    "mean": np.mean(cooperation_rates),
                    "std": np.std(cooperation_rates),
                    "range": (min(cooperation_rates), max(cooperation_rates))
                },
                "diversity": {
                    "value": np.mean(diversity_scores),
                    "category": self._categorize_diversity(np.mean(diversity_scores))
                },
                "stability": {
                    "trend_mean": np.mean([r.cooperation_trend for r in results]),
                    "trend_positive": sum(1 for r in results if r.cooperation_trend > 0)
                }
            }
        
        return profiles
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on comparison results."""
        recommendations = []
        
        # Analyze cooperation comparison
        coop_comparison = self.compare_scenarios("overall_cooperation_rate")
        if coop_comparison["best_scenario"]:
            recommendations.append(
                f"Scenario '{coop_comparison['best_scenario']}' achieves highest cooperation rates"
            )
        
        # Analyze diversity impact
        diversity_impact = self.analyze_diversity_impact()
        if diversity_impact.get("overall_correlation"):
            corr = diversity_impact["overall_correlation"]
            if corr > 0.3:
                recommendations.append("Higher model diversity correlates with increased cooperation")
            elif corr < -0.3:
                recommendations.append("Higher model diversity correlates with decreased cooperation")
        
        # Analyze stability
        trend_comparison = self.compare_scenarios("cooperation_trend")
        stable_scenarios = [s for s, t in trend_comparison.get("rankings", []) if t > -0.01]
        if stable_scenarios:
            recommendations.append(f"{len(stable_scenarios)} scenarios show stable or improving cooperation")
        
        return recommendations
    
    def _categorize_diversity(self, diversity_score: float) -> str:
        """Categorize diversity score into levels."""
        if diversity_score == 0:
            return "homogeneous"
        elif diversity_score < 0.5:
            return "low"
        elif diversity_score < 0.8:
            return "moderate"
        else:
            return "high"
    
    def _interpret_diversity_impact(self, group_stats: Dict, correlation: Optional[float]) -> str:
        """Interpret diversity impact results."""
        if not group_stats:
            return "Insufficient data for diversity analysis"
        
        # Compare homogeneous vs diverse
        homo_mean = group_stats.get("homogeneous", {}).get("mean", 0)
        high_div_mean = group_stats.get("high_diversity", {}).get("mean", 0)
        
        if homo_mean > high_div_mean + 0.1:
            interpretation = "Homogeneous scenarios show higher cooperation than diverse scenarios"
        elif high_div_mean > homo_mean + 0.1:
            interpretation = "Diverse scenarios show higher cooperation than homogeneous scenarios"
        else:
            interpretation = "Model diversity shows no clear impact on cooperation rates"
        
        if correlation is not None:
            if abs(correlation) > 0.5:
                interpretation += f" (strong correlation: {correlation:.3f})"
            elif abs(correlation) > 0.3:
                interpretation += f" (moderate correlation: {correlation:.3f})"
        
        return interpretation