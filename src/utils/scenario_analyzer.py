"""
Scenario-specific analysis features for mixed model experiments.
Provides detailed analysis capabilities for multi-model scenarios.
"""
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import json
from pathlib import Path
import logging

from src.utils.cross_model_analyzer import CrossModelAnalyzer
from src.utils.diversity_analyzer import DiversityAnalyzer
from src.core.scenario_manager import ScenarioManager

logger = logging.getLogger(__name__)


class ScenarioAnalyzer:
    """Comprehensive analysis for mixed model scenarios."""
    
    def __init__(self, scenario_manager: ScenarioManager):
        """
        Initialize analyzer with scenario context.
        
        Args:
            scenario_manager: Manager with scenario configuration
        """
        self.scenario_manager = scenario_manager
        self.cross_model_analyzer = CrossModelAnalyzer()
        self.diversity_analyzer = DiversityAnalyzer()
        
    def create_inter_model_cooperation_heatmap(self, game_results: List[Dict], 
                                             strategy_records: List[Dict]) -> Dict:
        """
        Create heatmap showing cooperation rates between each model pair.
        
        Args:
            game_results: List of game results
            strategy_records: List of strategy records with model info
            
        Returns:
            Dictionary with heatmap data
        """
        # Load data into cross-model analyzer
        self.cross_model_analyzer.load_data(game_results, strategy_records)
        
        # Get cooperation matrix
        matrix_df = self.cross_model_analyzer.calculate_cooperation_matrix()
        
        # Convert to heatmap format
        models = list(matrix_df.index)
        heatmap_data = {
            "type": "heatmap",
            "title": f"Inter-Model Cooperation: {self.scenario_manager.current_scenario.name if self.scenario_manager.current_scenario else 'Unknown'}",
            "models": models,
            "matrix": [],
            "annotations": [],
            "color_scale": {
                "min": 0.0,
                "max": 1.0,
                "scheme": "RdYlGn"
            }
        }
        
        # Build matrix and annotations
        for i, model1 in enumerate(models):
            row = []
            for j, model2 in enumerate(models):
                value = matrix_df.loc[model1, model2]
                if not np.isnan(value):
                    row.append(float(value))
                    heatmap_data["annotations"].append({
                        "x": j,
                        "y": i,
                        "text": f"{value:.2f}"
                    })
                else:
                    row.append(None)
            heatmap_data["matrix"].append(row)
        
        # Add scenario metadata
        heatmap_data["scenario_metadata"] = {
            "diversity_score": self.scenario_manager.calculate_model_diversity(),
            "model_distribution": self.scenario_manager.get_model_proportions()
        }
        
        return heatmap_data
    
    def track_minority_model_performance(self, game_results: List[Dict], 
                                       round_summaries: List[Dict]) -> Dict:
        """
        Track performance of minority models in the scenario.
        
        Args:
            game_results: List of game results
            round_summaries: List of round summaries
            
        Returns:
            Dictionary with minority model analysis
        """
        if not self.scenario_manager.current_scenario:
            return {"error": "No scenario loaded"}
        
        # Identify minority models (those with < 30% representation)
        proportions = self.scenario_manager.get_model_proportions()
        minority_models = [model for model, prop in proportions.items() if prop < 0.3]
        
        if not minority_models:
            return {"message": "No minority models in this scenario"}
        
        # Track performance metrics
        minority_performance = {}
        
        for model in minority_models:
            # Get agents with this model
            minority_agents = [agent_id for agent_id, assigned_model 
                             in self.scenario_manager.model_assignments.items() 
                             if assigned_model == model]
            
            # Calculate cooperation rates when playing as minority
            minority_games = []
            majority_games = []
            
            for game in game_results:
                if game.get("player1_id") in minority_agents:
                    if self.scenario_manager.get_agent_model(game.get("player2_id")) in minority_models:
                        minority_games.append(game.get("player1_action") == "COOPERATE")
                    else:
                        majority_games.append(game.get("player1_action") == "COOPERATE")
                        
                if game.get("player2_id") in minority_agents:
                    if self.scenario_manager.get_agent_model(game.get("player1_id")) in minority_models:
                        minority_games.append(game.get("player2_action") == "COOPERATE")
                    else:
                        majority_games.append(game.get("player2_action") == "COOPERATE")
            
            # Calculate rates
            minority_coop_rate = np.mean(minority_games) if minority_games else 0
            majority_coop_rate = np.mean(majority_games) if majority_games else 0
            
            minority_performance[model] = {
                "representation": proportions[model],
                "agent_count": len(minority_agents),
                "cooperation_with_minority": round(minority_coop_rate, 3),
                "cooperation_with_majority": round(majority_coop_rate, 3),
                "cooperation_difference": round(majority_coop_rate - minority_coop_rate, 3),
                "total_games": len(minority_games) + len(majority_games)
            }
        
        # Analyze trends
        interpretation = self._interpret_minority_performance(minority_performance)
        
        return {
            "minority_models": minority_performance,
            "interpretation": interpretation,
            "scenario": self.scenario_manager.current_scenario.name
        }
    
    def detect_model_dominance(self, game_results: List[Dict], 
                             strategy_records: List[Dict]) -> Dict:
        """
        Detect which model drives cooperation in the scenario.
        
        Args:
            game_results: List of game results
            strategy_records: List of strategy records
            
        Returns:
            Dictionary with dominance analysis
        """
        # Load data
        self.cross_model_analyzer.load_data(game_results, strategy_records)
        
        # Get average cooperation by model
        model_cooperation = self.cross_model_analyzer.calculate_average_cooperation_by_model()
        
        # Calculate influence scores
        influence_scores = {}
        
        for model, stats in model_cooperation.items():
            # Influence = cooperation rate * representation
            representation = self.scenario_manager.get_model_proportions().get(model, 0)
            cooperation_rate = stats["avg_cooperation"]
            
            influence_scores[model] = {
                "cooperation_rate": cooperation_rate,
                "representation": representation,
                "influence_score": cooperation_rate * representation,
                "total_games": stats["total_games"]
            }
        
        # Rank by influence
        ranked_models = sorted(influence_scores.items(), 
                             key=lambda x: x[1]["influence_score"], 
                             reverse=True)
        
        # Detect dominance patterns
        if ranked_models:
            dominant_model = ranked_models[0][0]
            dominance_score = ranked_models[0][1]["influence_score"]
            
            # Check if dominance is significant
            if len(ranked_models) > 1:
                second_score = ranked_models[1][1]["influence_score"]
                dominance_ratio = dominance_score / second_score if second_score > 0 else float('inf')
            else:
                dominance_ratio = float('inf')
        else:
            dominant_model = None
            dominance_ratio = 0
        
        return {
            "dominant_model": dominant_model,
            "dominance_ratio": round(dominance_ratio, 2),
            "model_rankings": [
                {
                    "model": model,
                    "influence_score": round(scores["influence_score"], 3),
                    "cooperation_rate": round(scores["cooperation_rate"], 3),
                    "representation": round(scores["representation"], 3)
                }
                for model, scores in ranked_models
            ],
            "interpretation": self._interpret_dominance(dominant_model, dominance_ratio)
        }
    
    def analyze_strategy_convergence_by_model(self, strategy_records: List[Dict]) -> Dict:
        """
        Track how strategies converge or diverge by model composition.
        
        Args:
            strategy_records: List of strategy records
            
        Returns:
            Dictionary with convergence analysis
        """
        # Group strategies by model and round
        strategies_by_model_round = defaultdict(lambda: defaultdict(list))
        
        for record in strategy_records:
            model = record.get("model", "unknown")
            round_num = record.get("round", 0)
            strategy = record.get("strategy", "").lower()
            
            # Extract cooperation intent
            if "cooperate" in strategy:
                strategies_by_model_round[model][round_num].append(1)
            elif "defect" in strategy:
                strategies_by_model_round[model][round_num].append(0)
        
        # Calculate convergence metrics
        convergence_analysis = {}
        
        for model in strategies_by_model_round:
            rounds = sorted(strategies_by_model_round[model].keys())
            
            if len(rounds) < 2:
                continue
            
            # Calculate variance over rounds
            variances = []
            for round_num in rounds:
                strategies = strategies_by_model_round[model][round_num]
                if len(strategies) > 1:
                    variances.append(np.var(strategies))
            
            # Convergence = reduction in variance
            if len(variances) > 1:
                early_variance = np.mean(variances[:len(variances)//2])
                late_variance = np.mean(variances[len(variances)//2:])
                convergence_rate = 1 - (late_variance / early_variance) if early_variance > 0 else 0
            else:
                convergence_rate = 0
            
            convergence_analysis[model] = {
                "rounds_analyzed": len(rounds),
                "early_variance": round(early_variance, 3) if 'early_variance' in locals() else None,
                "late_variance": round(late_variance, 3) if 'late_variance' in locals() else None,
                "convergence_rate": round(convergence_rate, 3),
                "converged": convergence_rate > 0.5
            }
        
        # Compare convergence across models
        comparison = self._compare_model_convergence(convergence_analysis)
        
        return {
            "model_convergence": convergence_analysis,
            "comparison": comparison,
            "scenario": self.scenario_manager.current_scenario.name if self.scenario_manager.current_scenario else "unknown"
        }
    
    def generate_scenario_dashboard(self, experiment_data: Dict) -> Dict:
        """
        Generate comprehensive dashboard for scenario analysis.
        
        Args:
            experiment_data: Complete experiment data including games, strategies, etc.
            
        Returns:
            Dictionary with dashboard data
        """
        game_results = experiment_data.get("games", [])
        strategy_records = experiment_data.get("strategies", [])
        round_summaries = experiment_data.get("round_summaries", [])
        
        dashboard = {
            "scenario_name": self.scenario_manager.current_scenario.name if self.scenario_manager.current_scenario else "unknown",
            "model_diversity": self.scenario_manager.calculate_model_diversity(),
            "model_distribution": self.scenario_manager.get_model_proportions(),
            "analyses": {}
        }
        
        # Run all analyses
        try:
            dashboard["analyses"]["cooperation_heatmap"] = self.create_inter_model_cooperation_heatmap(
                game_results, strategy_records
            )
        except Exception as e:
            logger.error(f"Failed to create cooperation heatmap: {e}")
            dashboard["analyses"]["cooperation_heatmap"] = {"error": str(e)}
        
        try:
            dashboard["analyses"]["minority_performance"] = self.track_minority_model_performance(
                game_results, round_summaries
            )
        except Exception as e:
            logger.error(f"Failed to track minority performance: {e}")
            dashboard["analyses"]["minority_performance"] = {"error": str(e)}
        
        try:
            dashboard["analyses"]["model_dominance"] = self.detect_model_dominance(
                game_results, strategy_records
            )
        except Exception as e:
            logger.error(f"Failed to detect model dominance: {e}")
            dashboard["analyses"]["model_dominance"] = {"error": str(e)}
        
        try:
            dashboard["analyses"]["strategy_convergence"] = self.analyze_strategy_convergence_by_model(
                strategy_records
            )
        except Exception as e:
            logger.error(f"Failed to analyze strategy convergence: {e}")
            dashboard["analyses"]["strategy_convergence"] = {"error": str(e)}
        
        # Add summary insights
        dashboard["insights"] = self._generate_scenario_insights(dashboard["analyses"])
        
        return dashboard
    
    def _interpret_minority_performance(self, performance: Dict) -> str:
        """Interpret minority model performance patterns."""
        if not performance:
            return "No minority models to analyze"
        
        # Check if minorities cooperate more with majorities
        avg_diff = np.mean([p["cooperation_difference"] for p in performance.values()])
        
        if avg_diff > 0.1:
            return "Minority models show appeasement behavior, cooperating more with majority models"
        elif avg_diff < -0.1:
            return "Minority models show defensive behavior, cooperating less with majority models"
        else:
            return "Minority models show no significant behavioral difference based on opponent type"
    
    def _interpret_dominance(self, dominant_model: Optional[str], ratio: float) -> str:
        """Interpret model dominance patterns."""
        if not dominant_model:
            return "No clear dominant model detected"
        
        if ratio > 2.0:
            return f"{dominant_model} strongly dominates cooperation dynamics"
        elif ratio > 1.5:
            return f"{dominant_model} moderately influences cooperation patterns"
        else:
            return "Multiple models share influence over cooperation dynamics"
    
    def _compare_model_convergence(self, convergence_data: Dict) -> Dict:
        """Compare convergence patterns across models."""
        if not convergence_data:
            return {"error": "No convergence data available"}
        
        converged_models = [m for m, data in convergence_data.items() if data["converged"]]
        avg_convergence = np.mean([data["convergence_rate"] for data in convergence_data.values()])
        
        return {
            "converged_models": converged_models,
            "convergence_ratio": len(converged_models) / len(convergence_data) if convergence_data else 0,
            "average_convergence_rate": round(avg_convergence, 3),
            "interpretation": "High strategy convergence across models" if avg_convergence > 0.5 else "Models maintain diverse strategies"
        }
    
    def _generate_scenario_insights(self, analyses: Dict) -> List[str]:
        """Generate key insights from scenario analyses."""
        insights = []
        
        # Diversity insight
        diversity = self.scenario_manager.calculate_model_diversity()
        if diversity == 0:
            insights.append("Homogeneous scenario provides baseline for model behavior")
        elif diversity > 1.0:
            insights.append("High model diversity creates complex interaction dynamics")
        
        # Dominance insight
        dominance = analyses.get("model_dominance", {})
        if dominance.get("dominant_model"):
            insights.append(f"{dominance['dominant_model']} acts as cooperation anchor in this scenario")
        
        # Minority insight
        minority = analyses.get("minority_performance", {})
        if "appeasement" in minority.get("interpretation", ""):
            insights.append("Minority models adapt cooperative strategies when outnumbered")
        
        # Convergence insight
        convergence = analyses.get("strategy_convergence", {})
        if convergence.get("comparison", {}).get("average_convergence_rate", 0) > 0.6:
            insights.append("Models converge toward similar strategies despite diversity")
        
        return insights