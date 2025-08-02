"""Analyzer for model diversity impact on cooperation rates."""
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import json
from pathlib import Path
from scipy import stats
import logging

from src.core.models import RoundSummary, ExperimentResult
from src.core.scenario_manager import ScenarioManager

logger = logging.getLogger(__name__)


@dataclass
class DiversityImpactResult:
    """Results from diversity impact analysis."""
    scenario_name: str
    diversity_score: float
    overall_cooperation_rate: float
    cooperation_by_round: List[float]
    cooperation_trend: float  # Slope of cooperation over rounds
    correlation_coefficient: float  # Correlation between diversity and cooperation
    p_value: float  # Statistical significance
    model_proportions: Dict[str, float]


class DiversityAnalyzer:
    """Analyzes the impact of model diversity on cooperation dynamics."""
    
    def __init__(self):
        self.results: List[DiversityImpactResult] = []
    
    def analyze_experiment(self, experiment_result: ExperimentResult, 
                         scenario_manager: ScenarioManager) -> DiversityImpactResult:
        """
        Analyze a single experiment for diversity impact.
        
        Args:
            experiment_result: Results from an experiment
            scenario_manager: Manager with scenario information
            
        Returns:
            DiversityImpactResult with analysis
        """
        # Get diversity score
        diversity_score = scenario_manager.calculate_model_diversity()
        
        # Extract cooperation rates by round
        cooperation_by_round = [
            summary.cooperation_rate 
            for summary in experiment_result.round_summaries
        ]
        
        # Calculate overall cooperation rate
        overall_cooperation = np.mean(cooperation_by_round) if cooperation_by_round else 0.0
        
        # Calculate cooperation trend (linear regression)
        if len(cooperation_by_round) > 1:
            rounds = np.arange(1, len(cooperation_by_round) + 1)
            slope, intercept, r_value, p_value, std_err = stats.linregress(rounds, cooperation_by_round)
            cooperation_trend = slope
        else:
            cooperation_trend = 0.0
            r_value = 0.0
            p_value = 1.0
        
        result = DiversityImpactResult(
            scenario_name=scenario_manager.current_scenario.name if scenario_manager.current_scenario else "unknown",
            diversity_score=diversity_score,
            overall_cooperation_rate=overall_cooperation,
            cooperation_by_round=cooperation_by_round,
            cooperation_trend=cooperation_trend,
            correlation_coefficient=r_value,
            p_value=p_value,
            model_proportions=scenario_manager.get_model_proportions()
        )
        
        self.results.append(result)
        return result
    
    def compare_scenarios(self, scenario_results: List[DiversityImpactResult]) -> Dict:
        """
        Compare multiple scenarios to identify diversity impact patterns.
        
        Args:
            scenario_results: List of results from different scenarios
            
        Returns:
            Dictionary with comparative analysis
        """
        if not scenario_results:
            return {"error": "No scenarios to compare"}
        
        # Extract data for correlation analysis
        diversity_scores = [r.diversity_score for r in scenario_results]
        cooperation_rates = [r.overall_cooperation_rate for r in scenario_results]
        cooperation_trends = [r.cooperation_trend for r in scenario_results]
        
        # Calculate correlation between diversity and cooperation
        if len(diversity_scores) > 2:
            corr_coef, corr_pval = stats.pearsonr(diversity_scores, cooperation_rates)
            trend_corr, trend_pval = stats.pearsonr(diversity_scores, cooperation_trends)
        else:
            corr_coef = corr_pval = trend_corr = trend_pval = None
        
        # Group by diversity level
        homogeneous = [r for r in scenario_results if r.diversity_score == 0]
        balanced = [r for r in scenario_results if 0.5 < r.diversity_score < 0.8]
        diverse = [r for r in scenario_results if r.diversity_score >= 0.8]
        
        comparison = {
            "total_scenarios": len(scenario_results),
            "diversity_cooperation_correlation": {
                "coefficient": corr_coef,
                "p_value": corr_pval,
                "significant": corr_pval < 0.05 if corr_pval is not None else False
            },
            "diversity_trend_correlation": {
                "coefficient": trend_corr,
                "p_value": trend_pval,
                "significant": trend_pval < 0.05 if trend_pval is not None else False
            },
            "cooperation_by_diversity_level": {
                "homogeneous": {
                    "count": len(homogeneous),
                    "avg_cooperation": np.mean([r.overall_cooperation_rate for r in homogeneous]) if homogeneous else None,
                    "avg_trend": np.mean([r.cooperation_trend for r in homogeneous]) if homogeneous else None
                },
                "balanced": {
                    "count": len(balanced),
                    "avg_cooperation": np.mean([r.overall_cooperation_rate for r in balanced]) if balanced else None,
                    "avg_trend": np.mean([r.cooperation_trend for r in balanced]) if balanced else None
                },
                "diverse": {
                    "count": len(diverse),
                    "avg_cooperation": np.mean([r.overall_cooperation_rate for r in diverse]) if diverse else None,
                    "avg_trend": np.mean([r.cooperation_trend for r in diverse]) if diverse else None
                }
            },
            "best_performing_scenario": max(scenario_results, key=lambda x: x.overall_cooperation_rate).scenario_name,
            "highest_diversity_scenario": max(scenario_results, key=lambda x: x.diversity_score).scenario_name,
            "most_stable_scenario": min(scenario_results, key=lambda x: abs(x.cooperation_trend)).scenario_name
        }
        
        return comparison
    
    def track_diversity_evolution(self, round_summaries: List[RoundSummary], 
                                scenario_manager: ScenarioManager) -> Dict:
        """
        Track how diversity affects cooperation evolution across rounds.
        
        Args:
            round_summaries: Summaries for each round
            scenario_manager: Manager with scenario information
            
        Returns:
            Dictionary with evolution analysis
        """
        diversity = scenario_manager.calculate_model_diversity()
        
        # Extract cooperation rates
        cooperation_rates = [s.cooperation_rate for s in round_summaries]
        
        # Analyze phases
        early_phase = cooperation_rates[:3] if len(cooperation_rates) >= 3 else cooperation_rates
        mid_phase = cooperation_rates[3:7] if len(cooperation_rates) >= 7 else []
        late_phase = cooperation_rates[7:] if len(cooperation_rates) > 7 else []
        
        evolution = {
            "diversity_score": diversity,
            "scenario": scenario_manager.current_scenario.name if scenario_manager.current_scenario else "unknown",
            "cooperation_evolution": {
                "early_phase": {
                    "rounds": len(early_phase),
                    "avg_cooperation": np.mean(early_phase) if early_phase else 0,
                    "std_cooperation": np.std(early_phase) if len(early_phase) > 1 else 0
                },
                "mid_phase": {
                    "rounds": len(mid_phase),
                    "avg_cooperation": np.mean(mid_phase) if mid_phase else 0,
                    "std_cooperation": np.std(mid_phase) if len(mid_phase) > 1 else 0
                },
                "late_phase": {
                    "rounds": len(late_phase),
                    "avg_cooperation": np.mean(late_phase) if late_phase else 0,
                    "std_cooperation": np.std(late_phase) if len(late_phase) > 1 else 0
                }
            },
            "stability_metrics": {
                "overall_variance": np.var(cooperation_rates) if len(cooperation_rates) > 1 else 0,
                "max_change": max(abs(cooperation_rates[i] - cooperation_rates[i-1]) 
                                for i in range(1, len(cooperation_rates))) if len(cooperation_rates) > 1 else 0,
                "convergence_achieved": np.std(late_phase) < np.std(early_phase) if late_phase and len(early_phase) > 1 else False
            }
        }
        
        return evolution
    
    def generate_diversity_visualization_data(self) -> Dict:
        """
        Generate data for visualizing diversity impact.
        
        Returns:
            Dictionary with visualization-ready data
        """
        if not self.results:
            return {"error": "No results to visualize"}
        
        viz_data = {
            "scatter_data": {
                "x_diversity": [r.diversity_score for r in self.results],
                "y_cooperation": [r.overall_cooperation_rate for r in self.results],
                "labels": [r.scenario_name for r in self.results]
            },
            "trend_data": {
                "x_diversity": [r.diversity_score for r in self.results],
                "y_trend": [r.cooperation_trend for r in self.results],
                "labels": [r.scenario_name for r in self.results]
            },
            "cooperation_by_round": {
                r.scenario_name: {
                    "diversity": r.diversity_score,
                    "cooperation_rates": r.cooperation_by_round
                }
                for r in self.results
            }
        }
        
        return viz_data
    
    def save_analysis(self, output_path: Path) -> None:
        """
        Save diversity analysis results to file.
        
        Args:
            output_path: Path to save results
        """
        analysis_data = {
            "individual_results": [
                {
                    "scenario_name": r.scenario_name,
                    "diversity_score": r.diversity_score,
                    "overall_cooperation_rate": r.overall_cooperation_rate,
                    "cooperation_by_round": r.cooperation_by_round,
                    "cooperation_trend": r.cooperation_trend,
                    "correlation_coefficient": r.correlation_coefficient,
                    "p_value": r.p_value,
                    "model_proportions": r.model_proportions
                }
                for r in self.results
            ],
            "comparative_analysis": self.compare_scenarios(self.results) if self.results else {},
            "visualization_data": self.generate_diversity_visualization_data()
        }
        
        with open(output_path, 'w') as f:
            json.dump(analysis_data, f, indent=2)
        
        logger.info(f"Saved diversity analysis to {output_path}")