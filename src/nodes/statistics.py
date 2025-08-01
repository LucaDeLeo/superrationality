"""Statistics computation node for analyzing cooperation patterns and power dynamics."""

from typing import Dict, Any, List, Tuple, Optional
import json
import re
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict

from src.nodes.base import AsyncNode, ContextKeys
from src.utils.data_manager import DataManager


class StatisticsNode(AsyncNode):
    """Computes comprehensive statistics on experiment results.
    
    This node analyzes game outcomes, cooperation patterns, power distributions,
    and detects anomalies across rounds. It generates a detailed statistical
    report including per-round metrics, trend analysis, and anomaly detection.
    """
    
    def __init__(self):
        """Initialize StatisticsNode with analysis parameters."""
        super().__init__(max_retries=1)
        
        # Configuration parameters
        self.anomaly_threshold = 2.0  # Standard deviations for anomaly detection
        self.trend_window = 3  # Rounds for moving average calculation
        self.power_quartile_method = "inclusive"  # Method for quartile calculation
        
        # Analysis results storage
        self.per_round_stats = []
        self.anomalies = []
        self.experiment_stats = {}
        
    async def _execute_impl(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute statistical analysis on game and round data.
        
        Args:
            context: Execution context containing DataManager and experiment_id
            
        Returns:
            Updated context with statistical_analysis results
            
        Raises:
            ValueError: If required context keys are missing
            FileNotFoundError: If expected data files are not found
        """
        # Extract required context
        data_manager = context.get(ContextKeys.DATA_MANAGER)
        experiment_id = context.get(ContextKeys.EXPERIMENT_ID)
        
        if not data_manager or not experiment_id:
            raise ValueError("Missing required context: DataManager or experiment_id")
            
        # Load experiment data
        rounds_path = data_manager.get_experiment_path() / "rounds"
        if not rounds_path.exists():
            raise FileNotFoundError(f"Rounds directory not found: {rounds_path}")
            
        # Process all rounds
        game_data_by_round = self._load_game_data(rounds_path)
        round_summaries = self._load_round_summaries(rounds_path)
        
        # Compute per-round statistics
        for round_num in sorted(game_data_by_round.keys()):
            round_stats = self.compute_round_statistics(
                game_data_by_round[round_num],
                round_num,
                round_summaries.get(round_num)
            )
            self.per_round_stats.append(round_stats)
            
        # Analyze trends across rounds
        trend_analysis = self.analyze_trends(self.per_round_stats)
        
        # Detect anomalies
        self.detect_anomalies(self.per_round_stats)
        
        # Generate comprehensive report
        statistical_analysis = self.generate_statistics_report(
            experiment_id,
            self.per_round_stats,
            trend_analysis,
            self.anomalies
        )
        
        # Save analysis results
        analysis_path = data_manager.get_experiment_path() / "statistical_analysis.json"
        data_manager._write_json(analysis_path, statistical_analysis)
        
        # Update context
        context["statistical_analysis"] = statistical_analysis
        
        return context
        
    def _load_game_data(self, rounds_path: Path) -> Dict[int, List[Dict[str, Any]]]:
        """Load all game data files organized by round.
        
        Args:
            rounds_path: Path to rounds directory
            
        Returns:
            Dictionary mapping round numbers to lists of game data
        """
        game_data_by_round = {}
        
        for game_file in sorted(rounds_path.glob("games_r*.json")):
            # Extract round number from filename
            match = re.search(r"games_r(\d+)\.json", game_file.name)
            if match:
                round_num = int(match.group(1))
                with open(game_file, 'r') as f:
                    data = json.load(f)
                    game_data_by_round[round_num] = data.get("games", [])
                    
        return game_data_by_round
        
    def _load_round_summaries(self, rounds_path: Path) -> Dict[int, Dict[str, Any]]:
        """Load all round summary files.
        
        Args:
            rounds_path: Path to rounds directory
            
        Returns:
            Dictionary mapping round numbers to summary data
        """
        summaries = {}
        
        for summary_file in sorted(rounds_path.glob("round_summary_r*.json")):
            match = re.search(r"round_summary_r(\d+)\.json", summary_file.name)
            if match:
                round_num = int(match.group(1))
                with open(summary_file, 'r') as f:
                    summaries[round_num] = json.load(f)
                    
        return summaries
        
    def compute_round_statistics(self, games: List[Dict[str, Any]], 
                                round_num: int,
                                round_summary: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Compute statistics for a single round.
        
        Args:
            games: List of game data for the round
            round_num: Round number
            round_summary: Optional existing round summary data
            
        Returns:
            Dictionary containing comprehensive round statistics
        """
        if not games:
            # Return empty statistics for missing rounds
            return self._empty_round_stats(round_num)
            
        # Count actions and outcomes
        total_actions = len(games) * 2  # Each game has 2 actions
        cooperate_count = 0
        mutual_cooperation = 0
        mutual_defection = 0
        asymmetric = 0
        
        # Collect payoffs and power data
        all_payoffs = []
        power_before = []
        power_after = []
        player_actions = defaultdict(list)  # player_id -> list of (action, power_before)
        
        for game in games:
            # Count cooperation actions
            if game["player1_action"] == "COOPERATE":
                cooperate_count += 1
            if game["player2_action"] == "COOPERATE":
                cooperate_count += 1
                
            # Count outcome types
            if game["player1_action"] == "COOPERATE" and game["player2_action"] == "COOPERATE":
                mutual_cooperation += 1
            elif game["player1_action"] == "DEFECT" and game["player2_action"] == "DEFECT":
                mutual_defection += 1
            else:
                asymmetric += 1
                
            # Collect payoffs
            all_payoffs.extend([game["player1_payoff"], game["player2_payoff"]])
            
            # Collect power data
            power_before.extend([game["player1_power_before"], game["player2_power_before"]])
            power_after.extend([game["player1_power_after"], game["player2_power_after"]])
            
            # Track actions by player for quartile analysis
            player_actions[game["player1_id"]].append(
                (game["player1_action"], game["player1_power_before"])
            )
            player_actions[game["player2_id"]].append(
                (game["player2_action"], game["player2_power_before"])
            )
            
        # Calculate basic rates
        cooperation_rate = cooperate_count / total_actions if total_actions > 0 else 0.0
        defection_rate = 1.0 - cooperation_rate
        
        # Calculate outcome rates (per game, not per action)
        total_games = len(games)
        mutual_cooperation_rate = mutual_cooperation / total_games if total_games > 0 else 0.0
        mutual_defection_rate = mutual_defection / total_games if total_games > 0 else 0.0
        asymmetric_outcome_rate = asymmetric / total_games if total_games > 0 else 0.0
        
        # Calculate payoff statistics
        average_payoff = np.mean(all_payoffs) if all_payoffs else 0.0
        payoff_variance = np.var(all_payoffs) if all_payoffs else 0.0
        payoff_std = np.std(all_payoffs) if all_payoffs else 0.0
        
        # Calculate power distribution statistics
        power_stats = self._calculate_power_stats(power_after)
        
        # Calculate cooperation by power quartile
        cooperation_by_quartile = self._calculate_cooperation_by_power_quartile(player_actions)
        
        return {
            "round": round_num,
            "cooperation_rate": round(cooperation_rate, 4),
            "defection_rate": round(defection_rate, 4),
            "mutual_cooperation_rate": round(mutual_cooperation_rate, 4),
            "mutual_defection_rate": round(mutual_defection_rate, 4),
            "asymmetric_outcome_rate": round(asymmetric_outcome_rate, 4),
            "average_payoff": round(average_payoff, 4),
            "payoff_variance": round(payoff_variance, 4),
            "payoff_std": round(payoff_std, 4),
            "power_stats": power_stats,
            "cooperation_by_power_quartile": cooperation_by_quartile
        }
        
    def _empty_round_stats(self, round_num: int) -> Dict[str, Any]:
        """Return empty statistics structure for missing rounds."""
        return {
            "round": round_num,
            "cooperation_rate": 0.0,
            "defection_rate": 0.0,
            "mutual_cooperation_rate": 0.0,
            "mutual_defection_rate": 0.0,
            "asymmetric_outcome_rate": 0.0,
            "average_payoff": 0.0,
            "payoff_variance": 0.0,
            "payoff_std": 0.0,
            "power_stats": {
                "mean": 0.0,
                "std": 0.0,
                "min": 0.0,
                "max": 0.0,
                "gini_coefficient": 0.0
            },
            "cooperation_by_power_quartile": {
                "Q1": 0.0,
                "Q2": 0.0,
                "Q3": 0.0,
                "Q4": 0.0
            }
        }
        
    def _calculate_power_stats(self, power_values: List[float]) -> Dict[str, float]:
        """Calculate power distribution statistics including Gini coefficient.
        
        Args:
            power_values: List of power values
            
        Returns:
            Dictionary with power statistics
        """
        if not power_values:
            return {
                "mean": 0.0,
                "std": 0.0,
                "min": 0.0,
                "max": 0.0,
                "gini_coefficient": 0.0
            }
            
        power_array = np.array(power_values)
        
        # Calculate Gini coefficient
        gini = self._calculate_gini_coefficient(power_array)
        
        return {
            "mean": round(float(np.mean(power_array)), 4),
            "std": round(float(np.std(power_array)), 4),
            "min": round(float(np.min(power_array)), 4),
            "max": round(float(np.max(power_array)), 4),
            "gini_coefficient": round(gini, 4)
        }
        
    def _calculate_gini_coefficient(self, values: np.ndarray) -> float:
        """Calculate Gini coefficient for measuring inequality.
        
        Args:
            values: Array of values (e.g., power levels)
            
        Returns:
            Gini coefficient (0 = perfect equality, 1 = perfect inequality)
        """
        if len(values) == 0:
            return 0.0
            
        # Sort values in ascending order
        sorted_values = np.sort(values)
        n = len(sorted_values)
        
        # Calculate cumulative values
        cumsum = np.cumsum(sorted_values)
        
        # Calculate Gini using the formula
        # G = (2 * sum(i * x_i)) / (n * sum(x_i)) - (n + 1) / n
        numerator = 2 * np.sum((np.arange(1, n + 1) * sorted_values))
        denominator = n * cumsum[-1]
        
        if denominator == 0:
            return 0.0
            
        gini = (numerator / denominator) - (n + 1) / n
        
        return max(0.0, min(1.0, gini))  # Ensure result is between 0 and 1
        
    def _calculate_cooperation_by_power_quartile(self, 
                                                player_actions: Dict[int, List[Tuple[str, float]]]) -> Dict[str, float]:
        """Calculate cooperation rates by power quartile.
        
        Args:
            player_actions: Dictionary mapping player_id to list of (action, power_before) tuples
            
        Returns:
            Dictionary with cooperation rates for each quartile
        """
        if not player_actions:
            return {"Q1": 0.0, "Q2": 0.0, "Q3": 0.0, "Q4": 0.0}
            
        # Collect all actions with power levels
        all_actions_with_power = []
        for player_id, actions in player_actions.items():
            all_actions_with_power.extend(actions)
            
        if not all_actions_with_power:
            return {"Q1": 0.0, "Q2": 0.0, "Q3": 0.0, "Q4": 0.0}
            
        # Sort by power level
        all_actions_with_power.sort(key=lambda x: x[1])
        
        # Calculate quartile boundaries
        n = len(all_actions_with_power)
        q1_end = n // 4
        q2_end = n // 2
        q3_end = 3 * n // 4
        
        # Calculate cooperation rate for each quartile
        quartiles = {
            "Q1": all_actions_with_power[:q1_end],
            "Q2": all_actions_with_power[q1_end:q2_end],
            "Q3": all_actions_with_power[q2_end:q3_end],
            "Q4": all_actions_with_power[q3_end:]
        }
        
        cooperation_by_quartile = {}
        for quartile, actions in quartiles.items():
            if actions:
                cooperate_count = sum(1 for action, _ in actions if action == "COOPERATE")
                cooperation_rate = cooperate_count / len(actions)
                cooperation_by_quartile[quartile] = round(cooperation_rate, 4)
            else:
                cooperation_by_quartile[quartile] = 0.0
                
        return cooperation_by_quartile
        
    def analyze_trends(self, per_round_stats: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze trends across rounds.
        
        Args:
            per_round_stats: List of per-round statistics
            
        Returns:
            Dictionary containing trend analysis results
        """
        if not per_round_stats:
            return self._empty_trend_analysis()
            
        # Extract time series data
        rounds = [stat["round"] for stat in per_round_stats]
        cooperation_rates = [stat["cooperation_rate"] for stat in per_round_stats]
        average_payoffs = [stat["average_payoff"] for stat in per_round_stats]
        gini_coefficients = [stat["power_stats"]["gini_coefficient"] for stat in per_round_stats]
        
        # Analyze cooperation trend
        cooperation_trend = self._analyze_cooperation_trend(rounds, cooperation_rates)
        
        # Analyze power concentration trend
        power_trend = self._analyze_power_concentration_trend(gini_coefficients)
        
        # Analyze payoff trend and correlation with cooperation
        payoff_trend = self._analyze_payoff_trend(
            average_payoffs, cooperation_rates
        )
        
        return {
            "cooperation_trend": cooperation_trend,
            "power_concentration_trend": power_trend,
            "payoff_trend": payoff_trend
        }
        
    def _empty_trend_analysis(self) -> Dict[str, Any]:
        """Return empty trend analysis structure."""
        return {
            "cooperation_trend": {
                "direction": "insufficient_data",
                "slope": 0.0,
                "r_squared": 0.0,
                "p_value": 1.0,
                "forecast_round_11": 0.0
            },
            "power_concentration_trend": {
                "gini_evolution": [],
                "trend": "insufficient_data",
                "interpretation": "Not enough data for trend analysis"
            },
            "payoff_trend": {
                "average_payoff_evolution": [],
                "correlation_with_cooperation": 0.0
            }
        }
        
    def _analyze_cooperation_trend(self, rounds: List[int], 
                                  cooperation_rates: List[float]) -> Dict[str, Any]:
        """Analyze cooperation rate trend using linear regression.
        
        Args:
            rounds: List of round numbers
            cooperation_rates: List of cooperation rates per round
            
        Returns:
            Dictionary with trend analysis results
        """
        if len(rounds) < 2:
            return {
                "direction": "insufficient_data",
                "slope": 0.0,
                "r_squared": 0.0,
                "p_value": 1.0,
                "forecast_round_11": 0.0,
                "moving_average": []
            }
            
        # Perform linear regression
        rounds_array = np.array(rounds)
        rates_array = np.array(cooperation_rates)
        
        # Calculate regression parameters
        slope, intercept, r_value, p_value = self._linear_regression(
            rounds_array, rates_array
        )
        
        # Determine trend direction
        if abs(slope) < 0.01:  # Threshold for "stable"
            direction = "stable"
        elif slope > 0:
            direction = "increasing"
        else:
            direction = "decreasing"
            
        # Forecast next round
        forecast_round_11 = slope * 11 + intercept
        forecast_round_11 = max(0.0, min(1.0, forecast_round_11))  # Clamp to [0, 1]
        
        # Calculate moving average
        moving_avg = self._calculate_moving_average(cooperation_rates, self.trend_window)
        
        return {
            "direction": direction,
            "slope": round(slope, 6),
            "r_squared": round(r_value ** 2, 4),
            "p_value": round(p_value, 4),
            "forecast_round_11": round(forecast_round_11, 4),
            "moving_average": [round(x, 4) for x in moving_avg]
        }
        
    def _analyze_power_concentration_trend(self, gini_coefficients: List[float]) -> Dict[str, Any]:
        """Analyze power concentration trend based on Gini coefficient evolution.
        
        Args:
            gini_coefficients: List of Gini coefficients per round
            
        Returns:
            Dictionary with power concentration trend analysis
        """
        if not gini_coefficients:
            return {
                "gini_evolution": [],
                "trend": "insufficient_data",
                "interpretation": "Not enough data for trend analysis"
            }
            
        # Calculate trend in Gini coefficient
        if len(gini_coefficients) >= 2:
            rounds = list(range(1, len(gini_coefficients) + 1))
            slope, _, _, _ = self._linear_regression(
                np.array(rounds), np.array(gini_coefficients)
            )
            
            if abs(slope) < 0.005:
                trend = "stable"
                interpretation = "Power distribution remains relatively constant"
            elif slope > 0:
                trend = "increasing"
                interpretation = "Power becoming more concentrated"
            else:
                trend = "decreasing"
                interpretation = "Power becoming more evenly distributed"
        else:
            trend = "insufficient_data"
            interpretation = "Not enough rounds for trend analysis"
            
        return {
            "gini_evolution": [round(g, 4) for g in gini_coefficients],
            "trend": trend,
            "interpretation": interpretation
        }
        
    def _analyze_payoff_trend(self, average_payoffs: List[float], 
                             cooperation_rates: List[float]) -> Dict[str, Any]:
        """Analyze payoff trend and its correlation with cooperation.
        
        Args:
            average_payoffs: List of average payoffs per round
            cooperation_rates: List of cooperation rates per round
            
        Returns:
            Dictionary with payoff trend analysis
        """
        if not average_payoffs:
            return {
                "average_payoff_evolution": [],
                "correlation_with_cooperation": 0.0
            }
            
        # Calculate correlation between payoffs and cooperation
        correlation = 0.0
        if len(average_payoffs) >= 2 and len(cooperation_rates) >= 2:
            min_length = min(len(average_payoffs), len(cooperation_rates))
            if min_length > 1:
                correlation = np.corrcoef(
                    average_payoffs[:min_length], 
                    cooperation_rates[:min_length]
                )[0, 1]
                
                # Handle NaN correlation (occurs when one variable is constant)
                if np.isnan(correlation):
                    correlation = 0.0
                    
        return {
            "average_payoff_evolution": [round(p, 4) for p in average_payoffs],
            "correlation_with_cooperation": round(correlation, 4)
        }
        
    def _linear_regression(self, x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float, float]:
        """Perform simple linear regression.
        
        Args:
            x: Independent variable (e.g., round numbers)
            y: Dependent variable (e.g., cooperation rates)
            
        Returns:
            Tuple of (slope, intercept, r_value, p_value)
        """
        if len(x) < 2:
            return 0.0, 0.0, 0.0, 1.0
            
        # Calculate means
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        
        # Calculate slope and intercept
        numerator = np.sum((x - x_mean) * (y - y_mean))
        denominator = np.sum((x - x_mean) ** 2)
        
        if denominator == 0:
            return 0.0, y_mean, 0.0, 1.0
            
        slope = numerator / denominator
        intercept = y_mean - slope * x_mean
        
        # Calculate r-value (correlation coefficient)
        y_pred = slope * x + intercept
        ss_tot = np.sum((y - y_mean) ** 2)
        ss_res = np.sum((y - y_pred) ** 2)
        
        if ss_tot == 0:
            r_squared = 0.0
        else:
            r_squared = 1 - (ss_res / ss_tot)
            
        r_value = np.sqrt(max(0, r_squared)) * np.sign(slope)
        
        # Simple p-value approximation (would need scipy for exact calculation)
        # Using a simplified approach based on sample size
        n = len(x)
        if n <= 2:
            p_value = 1.0
        else:
            # Approximate p-value based on r-value and sample size
            t_stat = r_value * np.sqrt((n - 2) / (1 - r_value ** 2 + 1e-10))
            # Rough approximation for two-tailed p-value
            if abs(t_stat) > 3.0:
                p_value = 0.01
            elif abs(t_stat) > 2.0:
                p_value = 0.05
            elif abs(t_stat) > 1.0:
                p_value = 0.3
            else:
                p_value = 0.5
                
        return slope, intercept, r_value, p_value
        
    def _calculate_moving_average(self, values: List[float], window: int) -> List[float]:
        """Calculate moving average for smoothed trend analysis.
        
        Args:
            values: List of values
            window: Window size for moving average
            
        Returns:
            List of moving averages
        """
        if not values or window <= 0:
            return []
            
        moving_avg = []
        for i in range(len(values)):
            start_idx = max(0, i - window + 1)
            window_values = values[start_idx:i + 1]
            avg = sum(window_values) / len(window_values)
            moving_avg.append(avg)
            
        return moving_avg
        
    def detect_anomalies(self, per_round_stats: List[Dict[str, Any]]) -> None:
        """Detect anomalies in the data.
        
        Args:
            per_round_stats: List of per-round statistics
        
        Updates self.anomalies with detected anomalies.
        """
        if len(per_round_stats) < 3:
            # Not enough data for meaningful anomaly detection
            return
            
        # Clear previous anomalies
        self.anomalies = []
        
        # Extract metrics for anomaly detection
        cooperation_rates = [stat["cooperation_rate"] for stat in per_round_stats]
        gini_coefficients = [stat["power_stats"]["gini_coefficient"] for stat in per_round_stats]
        average_payoffs = [stat["average_payoff"] for stat in per_round_stats]
        
        # Detect cooperation rate anomalies
        self._detect_cooperation_anomalies(per_round_stats, cooperation_rates)
        
        # Detect sudden strategy shifts
        self._detect_strategy_shifts(per_round_stats, cooperation_rates)
        
        # Detect power distribution anomalies
        self._detect_power_anomalies(per_round_stats, gini_coefficients)
        
        # Detect unusual game outcome patterns
        self._detect_outcome_pattern_anomalies(per_round_stats)
        
    def _detect_cooperation_anomalies(self, per_round_stats: List[Dict[str, Any]], 
                                    cooperation_rates: List[float]) -> None:
        """Detect rounds with unusual cooperation rates.
        
        Args:
            per_round_stats: List of per-round statistics
            cooperation_rates: List of cooperation rates
        """
        if len(cooperation_rates) < 3:
            return
            
        # Calculate mean and standard deviation
        rates_array = np.array(cooperation_rates)
        mean_rate = np.mean(rates_array)
        std_rate = np.std(rates_array)
        
        if std_rate == 0:
            return  # No variation, no anomalies
            
        # Check each round for anomalies
        for i, (stat, rate) in enumerate(zip(per_round_stats, cooperation_rates)):
            z_score = (rate - mean_rate) / std_rate
            
            if abs(z_score) > self.anomaly_threshold:
                severity = self._calculate_severity(abs(z_score))
                
                if z_score > 0:
                    anomaly_type = "cooperation_spike"
                    description = f"Cooperation rate {rate:.2f} is {abs(z_score):.1f} std deviations above mean"
                else:
                    anomaly_type = "cooperation_drop"
                    description = f"Cooperation rate {rate:.2f} is {abs(z_score):.1f} std deviations below mean"
                    
                # Add context
                context = self._generate_anomaly_context(i, per_round_stats, "cooperation")
                
                self.anomalies.append({
                    "round": stat["round"],
                    "type": anomaly_type,
                    "severity": severity,
                    "description": description,
                    "context": context
                })
                
    def _detect_strategy_shifts(self, per_round_stats: List[Dict[str, Any]], 
                               cooperation_rates: List[float]) -> None:
        """Detect sudden changes in cooperation rates between consecutive rounds.
        
        Args:
            per_round_stats: List of per-round statistics
            cooperation_rates: List of cooperation rates
        """
        if len(cooperation_rates) < 2:
            return
            
        # Calculate changes between consecutive rounds
        for i in range(1, len(cooperation_rates)):
            rate_change = abs(cooperation_rates[i] - cooperation_rates[i-1])
            
            # Flag large changes (more than 0.3 change in cooperation rate)
            if rate_change > 0.3:
                severity = "high" if rate_change > 0.5 else "medium"
                
                direction = "increase" if cooperation_rates[i] > cooperation_rates[i-1] else "decrease"
                description = f"Sudden {direction} in cooperation rate by {rate_change:.2f}"
                
                context = f"Changed from {cooperation_rates[i-1]:.2f} in round {per_round_stats[i-1]['round']} to {cooperation_rates[i]:.2f}"
                
                self.anomalies.append({
                    "round": per_round_stats[i]["round"],
                    "type": "strategy_shift",
                    "severity": severity,
                    "description": description,
                    "context": context
                })
                
    def _detect_power_anomalies(self, per_round_stats: List[Dict[str, Any]], 
                               gini_coefficients: List[float]) -> None:
        """Detect rounds with unusual power concentration.
        
        Args:
            per_round_stats: List of per-round statistics
            gini_coefficients: List of Gini coefficients
        """
        if len(gini_coefficients) < 3:
            return
            
        # Calculate mean and std for Gini coefficients
        gini_array = np.array(gini_coefficients)
        mean_gini = np.mean(gini_array)
        std_gini = np.std(gini_array)
        
        if std_gini == 0:
            return
            
        for i, (stat, gini) in enumerate(zip(per_round_stats, gini_coefficients)):
            z_score = (gini - mean_gini) / std_gini
            
            if abs(z_score) > self.anomaly_threshold:
                severity = self._calculate_severity(abs(z_score))
                
                if z_score > 0:
                    description = f"Gini coefficient {gini:.2f} indicates unusual power concentration"
                else:
                    description = f"Gini coefficient {gini:.2f} indicates unusual power equality"
                    
                # Add specific context about power distribution
                power_stats = stat["power_stats"]
                context = f"Power range: {power_stats['min']:.1f} to {power_stats['max']:.1f}, std: {power_stats['std']:.1f}"
                
                self.anomalies.append({
                    "round": stat["round"],
                    "type": "power_concentration",
                    "severity": severity,
                    "description": description,
                    "context": context
                })
                
    def _detect_outcome_pattern_anomalies(self, per_round_stats: List[Dict[str, Any]]) -> None:
        """Detect unusual game outcome patterns.
        
        Args:
            per_round_stats: List of per-round statistics
        """
        for stat in per_round_stats:
            # Check for extreme mutual cooperation or defection
            if stat["mutual_cooperation_rate"] > 0.9:
                self.anomalies.append({
                    "round": stat["round"],
                    "type": "extreme_cooperation",
                    "severity": "medium",
                    "description": f"Extremely high mutual cooperation rate: {stat['mutual_cooperation_rate']:.2f}",
                    "context": "Nearly all games resulted in mutual cooperation"
                })
            elif stat["mutual_defection_rate"] > 0.9:
                self.anomalies.append({
                    "round": stat["round"],
                    "type": "extreme_defection",
                    "severity": "high",
                    "description": f"Extremely high mutual defection rate: {stat['mutual_defection_rate']:.2f}",
                    "context": "Nearly all games resulted in mutual defection"
                })
                
            # Check for rounds with no cooperation at all
            if stat["cooperation_rate"] == 0.0:
                self.anomalies.append({
                    "round": stat["round"],
                    "type": "zero_cooperation",
                    "severity": "high",
                    "description": "No cooperation observed in this round",
                    "context": "All agents chose to defect"
                })
            elif stat["cooperation_rate"] == 1.0:
                self.anomalies.append({
                    "round": stat["round"],
                    "type": "perfect_cooperation",
                    "severity": "low",
                    "description": "Perfect cooperation observed in this round",
                    "context": "All agents chose to cooperate"
                })
                
    def _calculate_severity(self, z_score: float) -> str:
        """Calculate anomaly severity based on z-score.
        
        Args:
            z_score: Absolute z-score value
            
        Returns:
            Severity level: "low", "medium", or "high"
        """
        if z_score > 3.5:
            return "high"
        elif z_score > 2.5:
            return "medium"
        else:
            return "low"
            
    def _generate_anomaly_context(self, round_index: int, 
                                 per_round_stats: List[Dict[str, Any]], 
                                 metric_type: str) -> str:
        """Generate context for an anomaly.
        
        Args:
            round_index: Index of the anomalous round
            per_round_stats: List of per-round statistics
            metric_type: Type of metric ("cooperation", "power", etc.)
            
        Returns:
            Context string describing surrounding rounds
        """
        if round_index == 0:
            return "First round of the experiment"
        elif round_index == len(per_round_stats) - 1:
            return "Last round of the experiment"
            
        # Look at previous round
        prev_round = per_round_stats[round_index - 1]
        
        if metric_type == "cooperation":
            prev_rate = prev_round["cooperation_rate"]
            return f"Followed round {prev_round['round']} with cooperation rate {prev_rate:.2f}"
        elif metric_type == "power":
            prev_gini = prev_round["power_stats"]["gini_coefficient"]
            return f"Followed round {prev_round['round']} with Gini coefficient {prev_gini:.2f}"
        else:
            return f"Occurred in round {per_round_stats[round_index]['round']}"
        
    def generate_statistics_report(self, experiment_id: str,
                                 per_round_stats: List[Dict[str, Any]],
                                 trend_analysis: Dict[str, Any],
                                 anomalies: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate comprehensive statistical report.
        
        Args:
            experiment_id: Experiment identifier
            per_round_stats: List of per-round statistics
            trend_analysis: Results from trend analysis
            anomalies: List of detected anomalies
            
        Returns:
            Complete statistical analysis report
        """
        # Calculate experiment-wide summary statistics
        experiment_summary = self._calculate_experiment_summary(
            per_round_stats, trend_analysis, anomalies
        )
        
        # Sort anomalies by round number
        sorted_anomalies = sorted(anomalies, key=lambda x: x["round"])
        
        return {
            "statistical_analysis": {
                "experiment_id": experiment_id,
                "analysis_timestamp": datetime.now().isoformat() + "Z",
                "per_round_statistics": per_round_stats,
                "trend_analysis": trend_analysis,
                "anomalies_detected": sorted_anomalies,
                "experiment_summary": experiment_summary,
                "metadata": {
                    "analysis_method": "descriptive_statistics",
                    "anomaly_detection_method": "z_score",
                    "trend_analysis_method": "linear_regression",
                    "parameters": {
                        "anomaly_threshold_std": self.anomaly_threshold,
                        "trend_window_size": self.trend_window,
                        "power_quartile_method": self.power_quartile_method
                    }
                }
            }
        }
        
    def _calculate_experiment_summary(self, per_round_stats: List[Dict[str, Any]],
                                    trend_analysis: Dict[str, Any],
                                    anomalies: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate experiment-wide summary statistics.
        
        Args:
            per_round_stats: List of per-round statistics
            trend_analysis: Results from trend analysis
            anomalies: List of detected anomalies
            
        Returns:
            Dictionary with experiment summary
        """
        if not per_round_stats:
            return self._empty_experiment_summary()
            
        # Calculate total games (45 games per round)
        total_rounds = len(per_round_stats)
        total_games = total_rounds * 45  # Based on 10 agents playing round-robin
        
        # Calculate overall cooperation rate
        all_cooperation_rates = [stat["cooperation_rate"] for stat in per_round_stats]
        overall_cooperation_rate = np.mean(all_cooperation_rates) if all_cooperation_rates else 0.0
        
        # Calculate cooperation improvement
        if len(all_cooperation_rates) >= 2:
            cooperation_improvement = all_cooperation_rates[-1] - all_cooperation_rates[0]
        else:
            cooperation_improvement = 0.0
            
        # Determine dominant outcome
        dominant_outcome = self._determine_dominant_outcome(per_round_stats)
        
        # Assess power distribution stability
        power_stability = self._assess_power_stability(trend_analysis)
        
        # Determine statistical significance
        statistical_significance = self._assess_statistical_significance(
            trend_analysis, per_round_stats
        )
        
        return {
            "total_games": total_games,
            "overall_cooperation_rate": round(overall_cooperation_rate, 4),
            "cooperation_improvement": round(cooperation_improvement, 4),
            "dominant_outcome": dominant_outcome,
            "power_distribution_stability": power_stability,
            "statistical_significance": statistical_significance,
            "anomaly_summary": {
                "total_anomalies": len(anomalies),
                "high_severity": len([a for a in anomalies if a["severity"] == "high"]),
                "medium_severity": len([a for a in anomalies if a["severity"] == "medium"]),
                "low_severity": len([a for a in anomalies if a["severity"] == "low"])
            }
        }
        
    def _empty_experiment_summary(self) -> Dict[str, Any]:
        """Return empty experiment summary structure."""
        return {
            "total_games": 0,
            "overall_cooperation_rate": 0.0,
            "cooperation_improvement": 0.0,
            "dominant_outcome": "insufficient_data",
            "power_distribution_stability": "insufficient_data",
            "statistical_significance": {
                "cooperation_trend_significant": False,
                "power_trend_significant": False,
                "strategy_convergence_correlation": 0.0
            },
            "anomaly_summary": {
                "total_anomalies": 0,
                "high_severity": 0,
                "medium_severity": 0,
                "low_severity": 0
            }
        }
        
    def _determine_dominant_outcome(self, per_round_stats: List[Dict[str, Any]]) -> str:
        """Determine the dominant game outcome across all rounds.
        
        Args:
            per_round_stats: List of per-round statistics
            
        Returns:
            String indicating dominant outcome type
        """
        if not per_round_stats:
            return "insufficient_data"
            
        # Sum up outcome rates across all rounds
        total_mutual_cooperation = sum(stat["mutual_cooperation_rate"] for stat in per_round_stats)
        total_mutual_defection = sum(stat["mutual_defection_rate"] for stat in per_round_stats)
        total_asymmetric = sum(stat["asymmetric_outcome_rate"] for stat in per_round_stats)
        
        # Find the maximum
        outcomes = {
            "mutual_cooperation": total_mutual_cooperation,
            "mutual_defection": total_mutual_defection,
            "asymmetric": total_asymmetric
        }
        
        dominant = max(outcomes.items(), key=lambda x: x[1])
        
        # Check if there's a clear dominant outcome (at least 10% more than others)
        total = sum(outcomes.values())
        if total > 0 and dominant[1] / total > 0.4:
            return dominant[0]
        else:
            return "mixed"
            
    def _assess_power_stability(self, trend_analysis: Dict[str, Any]) -> str:
        """Assess power distribution stability based on trend analysis.
        
        Args:
            trend_analysis: Results from trend analysis
            
        Returns:
            String describing power distribution stability
        """
        power_trend = trend_analysis.get("power_concentration_trend", {})
        trend = power_trend.get("trend", "unknown")
        
        if trend == "stable":
            return "stable"
        elif trend == "increasing":
            return "concentrating"
        elif trend == "decreasing":
            return "equalizing"
        else:
            return "insufficient_data"
            
    def _assess_statistical_significance(self, trend_analysis: Dict[str, Any],
                                       per_round_stats: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Assess statistical significance of trends.
        
        Args:
            trend_analysis: Results from trend analysis
            per_round_stats: List of per-round statistics
            
        Returns:
            Dictionary with significance assessments
        """
        cooperation_trend = trend_analysis.get("cooperation_trend", {})
        
        # Cooperation trend is significant if p-value < 0.05
        cooperation_significant = cooperation_trend.get("p_value", 1.0) < 0.05
        
        # Power trend significance (simplified - would need proper test)
        power_trend = trend_analysis.get("power_concentration_trend", {})
        gini_evolution = power_trend.get("gini_evolution", [])
        
        power_significant = False
        if len(gini_evolution) >= 5:
            # Simple check: is the trend consistent?
            diffs = [gini_evolution[i+1] - gini_evolution[i] for i in range(len(gini_evolution)-1)]
            same_sign = all(d >= 0 for d in diffs) or all(d <= 0 for d in diffs)
            power_significant = same_sign and abs(gini_evolution[-1] - gini_evolution[0]) > 0.05
            
        # Calculate strategy convergence correlation
        payoff_trend = trend_analysis.get("payoff_trend", {})
        strategy_convergence = payoff_trend.get("correlation_with_cooperation", 0.0)
        
        return {
            "cooperation_trend_significant": cooperation_significant,
            "power_trend_significant": power_significant,
            "strategy_convergence_correlation": round(strategy_convergence, 4)
        }