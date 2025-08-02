"""
Cross-model cooperation analysis utility.

Analyzes cooperation patterns between different model types to identify
in-group vs out-group cooperation behaviors and model-specific coalitions.
"""

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
import pandas as pd
import numpy as np
from scipy import stats


@dataclass
class CooperationStats:
    """Statistics for a model pairing."""
    cooperation_count: int
    total_games: int
    cooperation_rate: float
    confidence_interval: Tuple[float, float]
    sample_size: int


class CrossModelAnalyzer:
    """Analyzes cooperation patterns across different model types."""
    
    def __init__(self):
        """Initialize the analyzer."""
        self.game_results = []
        self.strategy_records = []
        self.model_map = {}  # agent_id -> model mapping
        
    def load_data(self, game_results: List[Dict], strategy_records: List[Dict]):
        """
        Load game results and strategy records for analysis.
        
        Args:
            game_results: List of game result dictionaries
            strategy_records: List of strategy record dictionaries
        """
        self.game_results = game_results
        self.strategy_records = strategy_records
        
        # Build agent_id to model mapping
        self.model_map = {}
        for record in strategy_records:
            agent_id = record.get("agent_id")
            model = record.get("model", "unknown")
            if agent_id is not None:
                self.model_map[agent_id] = model
                
    def _get_model_for_agent(self, agent_id: int) -> str:
        """Get model type for an agent, with error handling."""
        return self.model_map.get(agent_id, "unknown")
    
    def _calculate_cooperation_counts(self) -> Tuple[Dict[str, Dict[str, int]], Dict[str, Dict[str, int]]]:
        """
        Calculate cooperation counts and total games per model pairing.
        
        Returns:
            Tuple of (cooperation_counts, total_games) dictionaries
        """
        cooperation_counts = defaultdict(lambda: defaultdict(int))
        total_games = defaultdict(lambda: defaultdict(int))
        
        for game in self.game_results:
            player1_id = game.get("player1_id")
            player2_id = game.get("player2_id")
            player1_action = game.get("player1_action")
            player2_action = game.get("player2_action")
            
            if None in (player1_id, player2_id, player1_action, player2_action):
                continue
                
            model1 = self._get_model_for_agent(player1_id)
            model2 = self._get_model_for_agent(player2_id)
            
            # Track both directions
            total_games[model1][model2] += 1
            total_games[model2][model1] += 1
            
            if player1_action == "COOPERATE":
                cooperation_counts[model1][model2] += 1
            if player2_action == "COOPERATE":
                cooperation_counts[model2][model1] += 1
                
        return cooperation_counts, total_games
    
    def _calculate_confidence_interval(self, successes: int, trials: int, 
                                     confidence: float = 0.95) -> Tuple[float, float]:
        """
        Calculate Wilson score confidence interval for a proportion.
        
        Args:
            successes: Number of successful outcomes
            trials: Total number of trials
            confidence: Confidence level (default 0.95)
            
        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        if trials == 0:
            return (0.0, 1.0)
            
        # Wilson score interval
        p_hat = successes / trials
        z = stats.norm.ppf((1 + confidence) / 2)
        z_squared = z * z
        
        denominator = 1 + z_squared / trials
        center = (p_hat + z_squared / (2 * trials)) / denominator
        margin = z * np.sqrt(p_hat * (1 - p_hat) / trials + z_squared / (4 * trials * trials)) / denominator
        
        return (max(0, center - margin), min(1, center + margin))
    
    def calculate_cooperation_matrix(self) -> pd.DataFrame:
        """
        Calculate cooperation rates between all model pairs.
        
        Returns:
            DataFrame with models as rows/columns and cooperation rates as values
        """
        cooperation_counts, total_games = self._calculate_cooperation_counts()
        
        # Get all unique models
        all_models = sorted(set(list(cooperation_counts.keys()) + 
                              list(sum([list(v.keys()) for v in cooperation_counts.values()], []))))
        
        # Build cooperation rate matrix
        matrix_data = {}
        for model1 in all_models:
            matrix_data[model1] = {}
            for model2 in all_models:
                total = total_games[model1][model2]
                if total > 0:
                    rate = cooperation_counts[model1][model2] / total
                    matrix_data[model1][model2] = round(rate, 3)
                else:
                    matrix_data[model1][model2] = np.nan
                    
        return pd.DataFrame(matrix_data)
    
    def get_cooperation_stats(self) -> Dict[str, Dict[str, CooperationStats]]:
        """
        Get detailed cooperation statistics with confidence intervals.
        
        Returns:
            Nested dict of model1 -> model2 -> CooperationStats
        """
        cooperation_counts, total_games = self._calculate_cooperation_counts()
                
        # Build stats with confidence intervals
        stats_dict = defaultdict(dict)
        for model1 in cooperation_counts:
            for model2 in cooperation_counts[model1]:
                coop_count = cooperation_counts[model1][model2]
                total = total_games[model1][model2]
                if total > 0:
                    rate = coop_count / total
                    ci = self._calculate_confidence_interval(coop_count, total)
                    stats_dict[model1][model2] = CooperationStats(
                        cooperation_count=coop_count,
                        total_games=total,
                        cooperation_rate=rate,
                        confidence_interval=ci,
                        sample_size=total
                    )
                    
        return dict(stats_dict)
    
    def detect_in_group_bias(self) -> Dict[str, Any]:
        """
        Detect if models cooperate more with their own type.
        
        Returns:
            Dict containing:
            - same_model_rate: Average cooperation rate for same-model pairings
            - cross_model_rate: Average cooperation rate for different-model pairings
            - bias: Difference between same and cross model rates
            - p_value: Statistical significance of the bias
            - effect_size: Cohen's d effect size
            - confidence_interval: CI for the bias
        """
        same_model_cooperations = []
        same_model_totals = []
        cross_model_cooperations = []
        cross_model_totals = []
        
        stats = self.get_cooperation_stats()
        
        for model1, model2_stats in stats.items():
            for model2, coop_stats in model2_stats.items():
                if model1 == model2:
                    same_model_cooperations.append(coop_stats.cooperation_count)
                    same_model_totals.append(coop_stats.total_games)
                else:
                    cross_model_cooperations.append(coop_stats.cooperation_count)
                    cross_model_totals.append(coop_stats.total_games)
                    
        # Calculate rates
        same_total = sum(same_model_totals)
        cross_total = sum(cross_model_totals)
        
        if same_total == 0 or cross_total == 0:
            return {
                "same_model_rate": None,
                "cross_model_rate": None,
                "bias": None,
                "p_value": None,
                "effect_size": None,
                "confidence_interval": (None, None),
                "sample_sizes": {
                    "same_model": same_total,
                    "cross_model": cross_total
                }
            }
            
        same_rate = sum(same_model_cooperations) / same_total
        cross_rate = sum(cross_model_cooperations) / cross_total
        bias = same_rate - cross_rate
        
        # Statistical significance (two-proportion z-test)
        pooled_p = (sum(same_model_cooperations) + sum(cross_model_cooperations)) / (same_total + cross_total)
        se = np.sqrt(pooled_p * (1 - pooled_p) * (1/same_total + 1/cross_total))
        
        if se > 0:
            z = bias / se
            p_value = 2 * (1 - stats.norm.cdf(abs(z)))
        else:
            p_value = 1.0
            
        # Effect size (Cohen's d for proportions)
        if same_total > 1 and cross_total > 1:
            same_var = same_rate * (1 - same_rate)
            cross_var = cross_rate * (1 - cross_rate)
            pooled_sd = np.sqrt((same_var + cross_var) / 2)
            effect_size = bias / pooled_sd if pooled_sd > 0 else 0
        else:
            effect_size = 0
            
        # Confidence interval for the bias
        ci_same = self._calculate_confidence_interval(sum(same_model_cooperations), same_total)
        ci_cross = self._calculate_confidence_interval(sum(cross_model_cooperations), cross_total)
        bias_ci = (ci_same[0] - ci_cross[1], ci_same[1] - ci_cross[0])
        
        return {
            "same_model_rate": round(same_rate, 3),
            "cross_model_rate": round(cross_rate, 3),
            "bias": round(bias, 3),
            "p_value": round(p_value, 4),
            "effect_size": round(effect_size, 3),
            "confidence_interval": (round(bias_ci[0], 3), round(bias_ci[1], 3)),
            "sample_sizes": {
                "same_model": same_total,
                "cross_model": cross_total
            }
        }
    
    def analyze_model_coalitions(self, include_temporal: bool = True) -> Dict[str, Any]:
        """
        Detect if certain model pairs form persistent cooperation coalitions.
        
        Args:
            include_temporal: Whether to include temporal coalition tracking
            
        Returns:
            Dict containing coalition analysis results
        """
        # Track cooperation consistency by round for each model pair
        cooperation_by_round = defaultdict(lambda: defaultdict(list))
        
        # Group games by round
        games_by_round = defaultdict(list)
        for game in self.game_results:
            round_num = game.get("round", 0)
            games_by_round[round_num].append(game)
            
        # Analyze each round
        for round_num in sorted(games_by_round.keys()):
            round_games = games_by_round[round_num]
            
            # Track cooperation in this round
            round_cooperation = defaultdict(lambda: defaultdict(list))
            
            for game in round_games:
                player1_id = game.get("player1_id")
                player2_id = game.get("player2_id")
                player1_action = game.get("player1_action")
                player2_action = game.get("player2_action")
                
                if None in (player1_id, player2_id, player1_action, player2_action):
                    continue
                    
                model1 = self._get_model_for_agent(player1_id)
                model2 = self._get_model_for_agent(player2_id)
                
                # Track mutual cooperation
                mutual_coop = (player1_action == "COOPERATE" and player2_action == "COOPERATE")
                
                # Create consistent ordering for model pairs
                model_pair = tuple(sorted([model1, model2]))
                round_cooperation[model_pair][round_num].append(mutual_coop)
                
            # Calculate round cooperation rates
            for model_pair, round_data in round_cooperation.items():
                for round_num, outcomes in round_data.items():
                    if outcomes:
                        rate = sum(outcomes) / len(outcomes)
                        cooperation_by_round[model_pair][round_num] = rate
                        
        # Analyze coalition patterns
        coalitions = []
        for model_pair, round_rates in cooperation_by_round.items():
            rates = [round_rates.get(r, 0) for r in sorted(games_by_round.keys())]
            
            if len(rates) >= 3:  # Need at least 3 rounds for meaningful coalition
                avg_rate = np.mean(rates)
                consistency = 1 - np.std(rates) if len(rates) > 1 else 1.0
                strength = avg_rate * consistency  # High cooperation + high consistency
                
                coalitions.append({
                    "models": list(model_pair),
                    "average_cooperation": round(avg_rate, 3),
                    "consistency": round(consistency, 3),
                    "strength": round(strength, 3),
                    "rounds_analyzed": len(rates)
                })
                
        # Sort by strength
        coalitions.sort(key=lambda x: x["strength"], reverse=True)
        
        # Identify strongest and weakest pairs
        if coalitions:
            strongest = coalitions[0]["models"]
            weakest = coalitions[-1]["models"]
        else:
            strongest = None
            weakest = None
            
        # Detect if there are clear coalition groups
        high_strength_coalitions = [c for c in coalitions if c["strength"] > 0.7]
        coalition_detected = len(high_strength_coalitions) >= 2
        
        result = {
            "detected": coalition_detected,
            "coalition_groups": high_strength_coalitions[:5],  # Top 5 coalitions
            "all_pairings": coalitions,
            "strongest_pair": strongest,
            "weakest_pair": weakest,
            "total_rounds_analyzed": len(games_by_round)
        }
        
        # Add temporal tracking if requested
        if include_temporal and len(games_by_round) > 0:
            from src.utils.coalition_tracker import TemporalCoalitionTracker
            
            tracker = TemporalCoalitionTracker()
            
            # Process each round
            for round_num in sorted(games_by_round.keys()):
                round_cooperation_rates = {}
                
                # Calculate cooperation rates for this round
                for model_pair, round_data in cooperation_by_round.items():
                    if round_num in round_data:
                        rates = round_data[round_num]
                        if rates:
                            round_cooperation_rates[model_pair] = sum(rates) / len(rates)
                
                # Update tracker
                tracker.update_round(round_num, round_cooperation_rates)
            
            # Get temporal analysis
            result["temporal_analysis"] = {
                "stability_metrics": tracker.get_coalition_stability_metrics(),
                "cross_vs_same_model": tracker.identify_cross_model_vs_same_model_patterns(),
                "coalition_cascades": tracker.detect_coalition_cascades(),
                "defection_patterns": tracker.track_defection_patterns(),
                "network_data": tracker.generate_coalition_network_data()
            }
        
        return result
    
    def generate_heatmap_data(self) -> Dict[str, Any]:
        """
        Generate visualization data for cooperation heatmap.
        
        Returns:
            JSON-serializable data structure for heatmap visualization
        """
        # Get cooperation matrix
        matrix_df = self.calculate_cooperation_matrix()
        
        # Get detailed stats for significance
        stats = self.get_cooperation_stats()
        
        # Convert matrix to list format
        models = list(matrix_df.index)
        matrix_values = []
        significance_values = []
        
        for model1 in models:
            row_values = []
            sig_row = []
            for model2 in models:
                # Get cooperation rate
                rate = matrix_df.loc[model1, model2]
                row_values.append(float(rate) if not pd.isna(rate) else None)
                
                # Calculate significance (simplified p-value calculation)
                if model1 in stats and model2 in stats[model1]:
                    coop_stats = stats[model1][model2]
                    # Use confidence interval width as proxy for significance
                    ci_width = coop_stats.confidence_interval[1] - coop_stats.confidence_interval[0]
                    # Smaller CI width = more significant
                    significance = 1.0 - min(ci_width, 1.0)
                    sig_row.append(round(significance, 3))
                else:
                    sig_row.append(None)
                    
            matrix_values.append(row_values)
            significance_values.append(sig_row)
        
        # Create color scale configuration
        color_scale = {
            "min": 0.0,
            "max": 1.0,
            "midpoint": 0.5,
            "colormap": "RdYlGn",  # Red-Yellow-Green colormap
            "null_color": "#e0e0e0"  # Gray for missing data
        }
        
        # Generate time series data
        time_series_data = self._generate_time_series_data()
        
        return {
            "heatmap": {
                "matrix": matrix_values,
                "labels": models,
                "significance": significance_values,
                "color_scale": color_scale,
                "title": "Cross-Model Cooperation Rates",
                "x_label": "Model (as Player 2)",
                "y_label": "Model (as Player 1)"
            },
            "time_series": time_series_data,
            "metadata": {
                "total_games": len(self.game_results),
                "total_models": len(models),
                "generated_at": pd.Timestamp.now().isoformat()
            }
        }
    
    def _generate_time_series_data(self) -> Dict[str, Any]:
        """
        Generate time series data showing cooperation evolution by model pairing.
        
        Returns:
            Time series visualization data
        """
        # Group games by round
        games_by_round = defaultdict(list)
        for game in self.game_results:
            round_num = game.get("round", 0)
            games_by_round[round_num].append(game)
        
        # Track cooperation rates by round for each model pair
        time_series = defaultdict(lambda: {"rounds": [], "cooperation_rates": [], "sample_sizes": []})
        
        for round_num in sorted(games_by_round.keys()):
            # Count cooperation by model pair in this round
            round_cooperation = defaultdict(lambda: {"cooperate": 0, "total": 0})
            
            for game in games_by_round[round_num]:
                player1_id = game.get("player1_id")
                player2_id = game.get("player2_id")
                player1_action = game.get("player1_action")
                player2_action = game.get("player2_action")
                
                if None in (player1_id, player2_id, player1_action, player2_action):
                    continue
                    
                model1 = self._get_model_for_agent(player1_id)
                model2 = self._get_model_for_agent(player2_id)
                
                # Track cooperation for model1 -> model2
                key1 = f"{model1} → {model2}"
                round_cooperation[key1]["total"] += 1
                if player1_action == "COOPERATE":
                    round_cooperation[key1]["cooperate"] += 1
                
                # Track cooperation for model2 -> model1
                key2 = f"{model2} → {model1}"
                round_cooperation[key2]["total"] += 1
                if player2_action == "COOPERATE":
                    round_cooperation[key2]["cooperate"] += 1
            
            # Calculate rates for this round
            for pair_key, counts in round_cooperation.items():
                if counts["total"] > 0:
                    rate = counts["cooperate"] / counts["total"]
                    time_series[pair_key]["rounds"].append(round_num)
                    time_series[pair_key]["cooperation_rates"].append(round(rate, 3))
                    time_series[pair_key]["sample_sizes"].append(counts["total"])
        
        # Convert to list format and select top pairs by total games
        series_list = []
        for pair_key, data in time_series.items():
            total_games = sum(data["sample_sizes"])
            if total_games >= 5:  # Only include pairs with meaningful data
                series_list.append({
                    "name": pair_key,
                    "data": data,
                    "total_games": total_games
                })
        
        # Sort by total games and take top 10
        series_list.sort(key=lambda x: x["total_games"], reverse=True)
        top_series = series_list[:10]
        
        return {
            "series": top_series,
            "x_label": "Round",
            "y_label": "Cooperation Rate",
            "title": "Cooperation Evolution by Model Pairing",
            "y_range": [0, 1],
            "show_sample_sizes": True
        }
    
    def calculate_average_cooperation_by_model(self) -> Dict[str, Dict[str, Any]]:
        """
        Calculate average cooperation rate per model type.
        
        Returns:
            Dict mapping model name to cooperation statistics
        """
        model_cooperation = defaultdict(lambda: {"cooperate": 0, "total": 0})
        
        for game in self.game_results:
            player1_id = game.get("player1_id")
            player2_id = game.get("player2_id")
            player1_action = game.get("player1_action")
            player2_action = game.get("player2_action")
            
            if None in (player1_id, player2_id, player1_action, player2_action):
                continue
                
            model1 = self._get_model_for_agent(player1_id)
            model2 = self._get_model_for_agent(player2_id)
            
            # Track cooperation for each model
            model_cooperation[model1]["total"] += 1
            if player1_action == "COOPERATE":
                model_cooperation[model1]["cooperate"] += 1
                
            model_cooperation[model2]["total"] += 1
            if player2_action == "COOPERATE":
                model_cooperation[model2]["cooperate"] += 1
        
        # Calculate rates and confidence intervals
        result = {}
        for model, counts in model_cooperation.items():
            if counts["total"] > 0:
                rate = counts["cooperate"] / counts["total"]
                ci = self._calculate_confidence_interval(counts["cooperate"], counts["total"])
                
                result[model] = {
                    "avg_cooperation": round(rate, 3),
                    "total_games": counts["total"],
                    "cooperation_count": counts["cooperate"],
                    "confidence_interval": ci
                }
        
        return result
    
    def compute_model_diversity_impact(self) -> Dict[str, Any]:
        """
        Compute the impact of model diversity on overall cooperation.
        
        Returns:
            Dictionary with diversity metrics and impact analysis
        """
        # Get in-group bias first
        bias_analysis = self.detect_in_group_bias()
        
        # Calculate average cooperation by model pairing type
        same_model_games = []
        cross_model_games = []
        
        for game in self.game_results:
            player1_id = game.get("player1_id")
            player2_id = game.get("player2_id")
            player1_action = game.get("player1_action")
            player2_action = game.get("player2_action")
            
            if None in (player1_id, player2_id, player1_action, player2_action):
                continue
                
            model1 = self._get_model_for_agent(player1_id)
            model2 = self._get_model_for_agent(player2_id)
            
            # Track mutual cooperation
            mutual_coop = (player1_action == "COOPERATE" and player2_action == "COOPERATE")
            
            if model1 == model2:
                same_model_games.append(mutual_coop)
            else:
                cross_model_games.append(mutual_coop)
        
        # Calculate diversity metrics
        total_games = len(same_model_games) + len(cross_model_games)
        diversity_ratio = len(cross_model_games) / total_games if total_games > 0 else 0
        
        # Calculate cooperation rates
        same_model_coop_rate = sum(same_model_games) / len(same_model_games) if same_model_games else 0
        cross_model_coop_rate = sum(cross_model_games) / len(cross_model_games) if cross_model_games else 0
        
        # Calculate impact
        cooperation_difference = same_model_coop_rate - cross_model_coop_rate
        diversity_cost = cooperation_difference * diversity_ratio if cooperation_difference > 0 else 0
        
        return {
            "diversity_ratio": round(diversity_ratio, 3),
            "same_model_mutual_cooperation": round(same_model_coop_rate, 3),
            "cross_model_mutual_cooperation": round(cross_model_coop_rate, 3),
            "cooperation_difference": round(cooperation_difference, 3),
            "diversity_cost": round(diversity_cost, 3),
            "in_group_bias": bias_analysis.get("bias", 0),
            "statistical_significance": bias_analysis.get("p_value", 1.0),
            "interpretation": self._interpret_diversity_impact(diversity_cost, bias_analysis.get("p_value", 1.0))
        }
    
    def _interpret_diversity_impact(self, diversity_cost: float, p_value: float) -> str:
        """
        Provide interpretation of diversity impact.
        
        Args:
            diversity_cost: The calculated diversity cost
            p_value: Statistical significance of in-group bias
            
        Returns:
            Human-readable interpretation
        """
        if p_value > 0.05:
            return "No significant impact of model diversity on cooperation detected."
        elif diversity_cost < 0.05:
            return "Model diversity has minimal impact on cooperation rates."
        elif diversity_cost < 0.15:
            return "Model diversity moderately reduces overall cooperation."
        else:
            return "Model diversity substantially reduces overall cooperation."
    
    def get_sample_size_warnings(self) -> Dict[str, List[str]]:
        """
        Generate warnings for model pairings with low sample sizes.
        
        Returns:
            Dictionary of warnings by severity level
        """
        warnings = {"critical": [], "low_confidence": [], "info": []}
        
        # Get cooperation stats
        stats = self.get_cooperation_stats()
        
        # Check each pairing
        for model1, model2_stats in stats.items():
            for model2, coop_stats in model2_stats.items():
                pairing = f"{model1} vs {model2}"
                
                if coop_stats.sample_size == 0:
                    warnings["critical"].append(f"No data available for {pairing}")
                elif coop_stats.sample_size < 5:
                    warnings["critical"].append(
                        f"Very low sample size ({coop_stats.sample_size}) for {pairing} - results unreliable"
                    )
                elif coop_stats.sample_size < 20:
                    warnings["low_confidence"].append(
                        f"Low sample size ({coop_stats.sample_size}) for {pairing} - interpret with caution"
                    )
                elif coop_stats.sample_size < 50:
                    warnings["info"].append(
                        f"Moderate sample size ({coop_stats.sample_size}) for {pairing}"
                    )
        
        return warnings
    
    def track_coalition_emergence(self) -> Dict[str, Any]:
        """
        Track the emergence of model-specific coalitions over rounds.
        
        Returns:
            Dictionary with coalition emergence patterns
        """
        # Group games by round
        games_by_round = defaultdict(list)
        for game in self.game_results:
            round_num = game.get("round", 0)
            games_by_round[round_num].append(game)
        
        # Track cooperation patterns by round
        round_coalitions = []
        cumulative_strength = defaultdict(list)
        
        for round_num in sorted(games_by_round.keys()):
            round_games = games_by_round[round_num]
            
            # Count mutual cooperation by model pair
            pair_cooperation = defaultdict(lambda: {"mutual_coop": 0, "total": 0})
            
            for game in round_games:
                player1_id = game.get("player1_id")
                player2_id = game.get("player2_id")
                player1_action = game.get("player1_action")
                player2_action = game.get("player2_action")
                
                if None in (player1_id, player2_id, player1_action, player2_action):
                    continue
                    
                model1 = self._get_model_for_agent(player1_id)
                model2 = self._get_model_for_agent(player2_id)
                
                # Create consistent pair key
                pair_key = tuple(sorted([model1, model2]))
                
                pair_cooperation[pair_key]["total"] += 1
                if player1_action == "COOPERATE" and player2_action == "COOPERATE":
                    pair_cooperation[pair_key]["mutual_coop"] += 1
            
            # Calculate coalition strength for this round
            round_strength = {}
            for pair, counts in pair_cooperation.items():
                if counts["total"] > 0:
                    strength = counts["mutual_coop"] / counts["total"]
                    round_strength[pair] = strength
                    cumulative_strength[pair].append(strength)
            
            round_coalitions.append({
                "round": round_num,
                "coalition_strengths": round_strength,
                "emerging_coalitions": [
                    pair for pair, strength in round_strength.items() if strength > 0.7
                ]
            })
        
        # Identify stable coalitions (consistent across rounds)
        stable_coalitions = []
        for pair, strengths in cumulative_strength.items():
            if len(strengths) >= 3:  # Need at least 3 rounds
                avg_strength = np.mean(strengths)
                consistency = 1 - np.std(strengths) if len(strengths) > 1 else 1.0
                
                if avg_strength > 0.6 and consistency > 0.7:
                    stable_coalitions.append({
                        "models": list(pair),
                        "average_strength": round(avg_strength, 3),
                        "consistency": round(consistency, 3),
                        "rounds_present": len(strengths)
                    })
        
        # Sort by average strength
        stable_coalitions.sort(key=lambda x: x["average_strength"], reverse=True)
        
        return {
            "round_by_round": round_coalitions,
            "stable_coalitions": stable_coalitions,
            "coalition_formation_round": self._find_coalition_formation_round(round_coalitions),
            "total_rounds_analyzed": len(games_by_round)
        }
    
    def _find_coalition_formation_round(self, round_coalitions: List[Dict]) -> Optional[int]:
        """
        Find the round where stable coalitions first emerge.
        
        Args:
            round_coalitions: Coalition data by round
            
        Returns:
            Round number where coalitions stabilize, or None
        """
        if not round_coalitions:
            return None
            
        for i, round_data in enumerate(round_coalitions):
            if len(round_data.get("emerging_coalitions", [])) >= 2:
                # Check if these coalitions persist
                if i + 1 < len(round_coalitions):
                    next_round = round_coalitions[i + 1]
                    # Check overlap
                    current_coalitions = set(map(tuple, round_data["emerging_coalitions"]))
                    next_coalitions = set(map(tuple, next_round.get("emerging_coalitions", [])))
                    
                    if len(current_coalitions & next_coalitions) >= 1:
                        return round_data["round"]
        
        return None
    
    def calculate_statistical_power(self) -> Dict[str, Dict[str, float]]:
        """
        Calculate statistical power for each model pairing comparison.
        
        Returns:
            Dictionary mapping model pairs to statistical power estimates
        """
        power_analysis = {}
        stats = self.get_cooperation_stats()
        
        # For each pairing, estimate power based on sample size and effect size
        for model1, model2_stats in stats.items():
            for model2, coop_stats in model2_stats.items():
                pair_key = f"{model1} vs {model2}"
                
                # Simplified power calculation based on sample size
                # In practice, would use proper power analysis
                n = coop_stats.sample_size
                
                if n == 0:
                    power = 0.0
                elif n < 5:
                    power = 0.1
                elif n < 10:
                    power = 0.3
                elif n < 20:
                    power = 0.5
                elif n < 50:
                    power = 0.7
                elif n < 100:
                    power = 0.8
                else:
                    power = 0.9
                
                # Adjust for effect size (cooperation rate distance from 0.5)
                effect_size = abs(coop_stats.cooperation_rate - 0.5) * 2
                adjusted_power = power * (0.5 + 0.5 * effect_size)
                
                power_analysis[pair_key] = {
                    "statistical_power": round(adjusted_power, 3),
                    "sample_size": n,
                    "effect_size": round(effect_size, 3),
                    "interpretation": self._interpret_power(adjusted_power)
                }
        
        return power_analysis
    
    def _interpret_power(self, power: float) -> str:
        """
        Provide interpretation of statistical power.
        
        Args:
            power: Statistical power value
            
        Returns:
            Human-readable interpretation
        """
        if power < 0.3:
            return "Very low power - results likely unreliable"
        elif power < 0.5:
            return "Low power - high chance of false negatives"
        elif power < 0.7:
            return "Moderate power - some confidence in results"
        elif power < 0.8:
            return "Good power - reliable for detecting large effects"
        else:
            return "Excellent power - reliable for detecting most effects"