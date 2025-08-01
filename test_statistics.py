"""Unit tests for StatisticsNode."""

import pytest
import pytest_asyncio
import asyncio
import numpy as np
from datetime import datetime
from pathlib import Path
import json
import tempfile
import shutil
from typing import Dict, Any, List

from src.nodes.statistics import StatisticsNode
from src.nodes.base import ContextKeys
from src.utils.data_manager import DataManager


class TestStatisticsNode:
    """Test suite for StatisticsNode functionality."""
    
    @pytest.fixture
    def temp_results_dir(self):
        """Create a temporary directory for test results."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def data_manager(self, temp_results_dir):
        """Create a DataManager instance with temporary directory."""
        return DataManager(base_path=temp_results_dir)
    
    @pytest.fixture
    def basic_context(self, data_manager):
        """Create basic context for testing."""
        return {
            ContextKeys.DATA_MANAGER: data_manager,
            ContextKeys.EXPERIMENT_ID: "test_exp_001"
        }
    
    @pytest.fixture
    def sample_game_data(self):
        """Create sample game data for testing."""
        return {
            "round": 1,
            "timestamp": "2024-01-15T10:35:00Z",
            "games": [
                {
                    "game_id": "r1_g1",
                    "round": 1,
                    "game_number": 1,
                    "player1_id": 0,
                    "player2_id": 1,
                    "player1_action": "COOPERATE",
                    "player2_action": "COOPERATE",
                    "player1_payoff": 3.0,
                    "player2_payoff": 3.0,
                    "player1_power_before": 100.0,
                    "player2_power_before": 100.0,
                    "player1_power_after": 103.0,
                    "player2_power_after": 103.0,
                    "timestamp": "2024-01-15T10:35:05Z"
                },
                {
                    "game_id": "r1_g2",
                    "round": 1,
                    "game_number": 2,
                    "player1_id": 0,
                    "player2_id": 2,
                    "player1_action": "COOPERATE",
                    "player2_action": "DEFECT",
                    "player1_payoff": 0.0,
                    "player2_payoff": 5.0,
                    "player1_power_before": 103.0,
                    "player2_power_before": 100.0,
                    "player1_power_after": 103.0,
                    "player2_power_after": 105.0,
                    "timestamp": "2024-01-15T10:35:10Z"
                },
                {
                    "game_id": "r1_g3",
                    "round": 1,
                    "game_number": 3,
                    "player1_id": 1,
                    "player2_id": 2,
                    "player1_action": "DEFECT",
                    "player2_action": "DEFECT",
                    "player1_payoff": 1.0,
                    "player2_payoff": 1.0,
                    "player1_power_before": 103.0,
                    "player2_power_before": 105.0,
                    "player1_power_after": 104.0,
                    "player2_power_after": 106.0,
                    "timestamp": "2024-01-15T10:35:15Z"
                }
            ]
        }
    
    def setup_test_data(self, data_manager, experiment_id, rounds_data):
        """Set up test data in the file system."""
        # DataManager uses get_experiment_path()
        data_manager.experiment_id = experiment_id
        exp_path = data_manager.get_experiment_path()
        rounds_path = exp_path / "rounds"
        rounds_path.mkdir(parents=True, exist_ok=True)
        
        for round_num, round_data in rounds_data.items():
            # Save game data
            game_file = rounds_path / f"games_r{round_num}.json"
            with open(game_file, 'w') as f:
                json.dump(round_data, f)
            
            # Save round summary
            summary_data = {
                "round": round_num,
                "cooperation_rate": 0.67,  # Placeholder
                "average_score": 3.0,
                "score_variance": 0.5,
                "power_distribution": {
                    "mean": 100.0,
                    "std": 5.0,
                    "min": 95.0,
                    "max": 105.0
                }
            }
            summary_file = rounds_path / f"round_summary_r{round_num}.json"
            with open(summary_file, 'w') as f:
                json.dump(summary_data, f)
    
    @pytest.mark.asyncio
    async def test_cooperation_rate_calculation(self, basic_context, sample_game_data):
        """Test cooperation rate calculation with various game outcomes."""
        node = StatisticsNode()
        
        # Test with sample data (3 cooperate, 3 defect actions)
        stats = node.compute_round_statistics(sample_game_data["games"], 1)
        
        assert stats["cooperation_rate"] == 0.5  # 3 out of 6 actions
        assert stats["defection_rate"] == 0.5
        assert stats["mutual_cooperation_rate"] == pytest.approx(0.333, rel=0.01)  # 1 out of 3 games
        assert stats["mutual_defection_rate"] == pytest.approx(0.333, rel=0.01)  # 1 out of 3 games
        assert stats["asymmetric_outcome_rate"] == pytest.approx(0.333, rel=0.01)  # 1 out of 3 games
    
    @pytest.mark.asyncio
    async def test_power_distribution_statistics(self, basic_context, sample_game_data):
        """Test power distribution statistics computation."""
        node = StatisticsNode()
        
        stats = node.compute_round_statistics(sample_game_data["games"], 1)
        power_stats = stats["power_stats"]
        
        # Check power stats structure
        assert "mean" in power_stats
        assert "std" in power_stats
        assert "min" in power_stats
        assert "max" in power_stats
        assert "gini_coefficient" in power_stats
        
        # Check values are reasonable
        assert power_stats["mean"] > 0
        assert power_stats["std"] >= 0
        assert power_stats["min"] <= power_stats["max"]
        assert 0 <= power_stats["gini_coefficient"] <= 1
    
    @pytest.mark.asyncio
    async def test_gini_coefficient_calculation(self):
        """Test Gini coefficient calculation."""
        node = StatisticsNode()
        
        # Test perfect equality
        equal_values = np.array([100, 100, 100, 100])
        gini_equal = node._calculate_gini_coefficient(equal_values)
        assert gini_equal == pytest.approx(0.0, abs=0.01)
        
        # Test some inequality
        unequal_values = np.array([50, 100, 150, 200])
        gini_unequal = node._calculate_gini_coefficient(unequal_values)
        assert 0 < gini_unequal < 1
        
        # Test extreme inequality
        extreme_values = np.array([1, 1, 1, 997])
        gini_extreme = node._calculate_gini_coefficient(extreme_values)
        assert gini_extreme > 0.7  # High inequality
    
    @pytest.mark.asyncio
    async def test_trend_analysis_increasing(self):
        """Test trend analysis with increasing cooperation pattern."""
        node = StatisticsNode()
        
        # Create increasing cooperation trend
        per_round_stats = []
        for i in range(10):
            per_round_stats.append({
                "round": i + 1,
                "cooperation_rate": 0.3 + (i * 0.05),  # Increases from 0.3 to 0.75
                "average_payoff": 2.0 + (i * 0.2),
                "power_stats": {"gini_coefficient": 0.25 - (i * 0.01)}
            })
        
        trends = node.analyze_trends(per_round_stats)
        
        assert trends["cooperation_trend"]["direction"] == "increasing"
        assert trends["cooperation_trend"]["slope"] > 0
        assert trends["cooperation_trend"]["forecast_round_11"] > 0.75
        assert trends["power_concentration_trend"]["trend"] == "decreasing"
    
    @pytest.mark.asyncio
    async def test_trend_analysis_decreasing(self):
        """Test trend analysis with decreasing cooperation pattern."""
        node = StatisticsNode()
        
        # Create decreasing cooperation trend
        per_round_stats = []
        for i in range(10):
            per_round_stats.append({
                "round": i + 1,
                "cooperation_rate": 0.8 - (i * 0.05),  # Decreases from 0.8 to 0.35
                "average_payoff": 4.0 - (i * 0.2),
                "power_stats": {"gini_coefficient": 0.15 + (i * 0.01)}
            })
        
        trends = node.analyze_trends(per_round_stats)
        
        assert trends["cooperation_trend"]["direction"] == "decreasing"
        assert trends["cooperation_trend"]["slope"] < 0
        assert trends["cooperation_trend"]["forecast_round_11"] < 0.35
        assert trends["power_concentration_trend"]["trend"] == "increasing"
    
    @pytest.mark.asyncio
    async def test_anomaly_detection_high_cooperation(self):
        """Test anomaly detection for unusually high cooperation."""
        node = StatisticsNode()
        
        # Create data with cooperation spike - 10 rounds with consistent baseline and extreme spike
        per_round_stats = [
            {"round": 1, "cooperation_rate": 0.5, "average_payoff": 2.5, "mutual_cooperation_rate": 0.3, "mutual_defection_rate": 0.2, "power_stats": {"gini_coefficient": 0.2}},
            {"round": 2, "cooperation_rate": 0.48, "average_payoff": 2.4, "mutual_cooperation_rate": 0.28, "mutual_defection_rate": 0.22, "power_stats": {"gini_coefficient": 0.21}},
            {"round": 3, "cooperation_rate": 0.52, "average_payoff": 2.6, "mutual_cooperation_rate": 0.32, "mutual_defection_rate": 0.18, "power_stats": {"gini_coefficient": 0.2}},
            {"round": 4, "cooperation_rate": 0.49, "average_payoff": 2.45, "mutual_cooperation_rate": 0.29, "mutual_defection_rate": 0.21, "power_stats": {"gini_coefficient": 0.22}},
            {"round": 5, "cooperation_rate": 0.51, "average_payoff": 2.55, "mutual_cooperation_rate": 0.31, "mutual_defection_rate": 0.19, "power_stats": {"gini_coefficient": 0.21}},
            {"round": 6, "cooperation_rate": 0.9, "average_payoff": 3.8, "mutual_cooperation_rate": 0.8, "mutual_defection_rate": 0.05, "power_stats": {"gini_coefficient": 0.2}},  # Anomaly
            {"round": 7, "cooperation_rate": 0.5, "average_payoff": 2.5, "mutual_cooperation_rate": 0.3, "mutual_defection_rate": 0.2, "power_stats": {"gini_coefficient": 0.21}},
            {"round": 8, "cooperation_rate": 0.48, "average_payoff": 2.4, "mutual_cooperation_rate": 0.28, "mutual_defection_rate": 0.22, "power_stats": {"gini_coefficient": 0.22}},
            {"round": 9, "cooperation_rate": 0.52, "average_payoff": 2.6, "mutual_cooperation_rate": 0.32, "mutual_defection_rate": 0.18, "power_stats": {"gini_coefficient": 0.2}},
            {"round": 10, "cooperation_rate": 0.5, "average_payoff": 2.5, "mutual_cooperation_rate": 0.3, "mutual_defection_rate": 0.2, "power_stats": {"gini_coefficient": 0.21}},
        ]
        
        node.detect_anomalies(per_round_stats)
        
        assert len(node.anomalies) > 0
        
        # Find cooperation spike anomaly
        spike_anomaly = next((a for a in node.anomalies if a["type"] == "cooperation_spike"), None)
        assert spike_anomaly is not None
        assert spike_anomaly["round"] == 6
        assert "above mean" in spike_anomaly["description"]
    
    @pytest.mark.asyncio
    async def test_anomaly_detection_power_concentration(self):
        """Test anomaly detection for unusual power concentration."""
        node = StatisticsNode()
        
        # Create data with power concentration anomaly - 10 rounds with consistent baseline
        per_round_stats = [
            {"round": 1, "cooperation_rate": 0.5, "average_payoff": 2.5, "mutual_cooperation_rate": 0.3, "mutual_defection_rate": 0.2, "power_stats": {"gini_coefficient": 0.2, "min": 90, "max": 110, "std": 5}},
            {"round": 2, "cooperation_rate": 0.5, "average_payoff": 2.5, "mutual_cooperation_rate": 0.3, "mutual_defection_rate": 0.2, "power_stats": {"gini_coefficient": 0.21, "min": 88, "max": 112, "std": 6}},
            {"round": 3, "cooperation_rate": 0.5, "average_payoff": 2.5, "mutual_cooperation_rate": 0.3, "mutual_defection_rate": 0.2, "power_stats": {"gini_coefficient": 0.19, "min": 91, "max": 109, "std": 4.5}},
            {"round": 4, "cooperation_rate": 0.5, "average_payoff": 2.5, "mutual_cooperation_rate": 0.3, "mutual_defection_rate": 0.2, "power_stats": {"gini_coefficient": 0.22, "min": 87, "max": 113, "std": 6.5}},
            {"round": 5, "cooperation_rate": 0.5, "average_payoff": 2.5, "mutual_cooperation_rate": 0.3, "mutual_defection_rate": 0.2, "power_stats": {"gini_coefficient": 0.2, "min": 90, "max": 110, "std": 5}},
            {"round": 6, "cooperation_rate": 0.5, "average_payoff": 2.5, "mutual_cooperation_rate": 0.3, "mutual_defection_rate": 0.2, "power_stats": {"gini_coefficient": 0.6, "min": 10, "max": 190, "std": 45}},  # Extreme anomaly
            {"round": 7, "cooperation_rate": 0.5, "average_payoff": 2.5, "mutual_cooperation_rate": 0.3, "mutual_defection_rate": 0.2, "power_stats": {"gini_coefficient": 0.21, "min": 89, "max": 111, "std": 5.5}},
            {"round": 8, "cooperation_rate": 0.5, "average_payoff": 2.5, "mutual_cooperation_rate": 0.3, "mutual_defection_rate": 0.2, "power_stats": {"gini_coefficient": 0.2, "min": 90, "max": 110, "std": 5}},
            {"round": 9, "cooperation_rate": 0.5, "average_payoff": 2.5, "mutual_cooperation_rate": 0.3, "mutual_defection_rate": 0.2, "power_stats": {"gini_coefficient": 0.19, "min": 91, "max": 109, "std": 4.5}},
            {"round": 10, "cooperation_rate": 0.5, "average_payoff": 2.5, "mutual_cooperation_rate": 0.3, "mutual_defection_rate": 0.2, "power_stats": {"gini_coefficient": 0.22, "min": 87, "max": 113, "std": 6.5}},
        ]
        
        node.detect_anomalies(per_round_stats)
        
        # Find power concentration anomaly
        power_anomaly = next((a for a in node.anomalies if a["type"] == "power_concentration"), None)
        assert power_anomaly is not None
        assert power_anomaly["round"] == 6
        assert "concentration" in power_anomaly["description"]
    
    @pytest.mark.asyncio
    async def test_moving_average_calculation(self):
        """Test moving average calculation."""
        node = StatisticsNode()
        
        values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        window = 3
        
        moving_avg = node._calculate_moving_average(values, window)
        
        assert len(moving_avg) == len(values)
        assert moving_avg[0] == 1  # First value
        assert moving_avg[1] == 1.5  # Average of [1, 2]
        assert moving_avg[2] == 2  # Average of [1, 2, 3]
        assert moving_avg[3] == 3  # Average of [2, 3, 4]
        assert moving_avg[-1] == 9  # Average of [8, 9, 10]
    
    @pytest.mark.asyncio
    async def test_quartile_analysis(self, sample_game_data):
        """Test cooperation rate by power quartile calculation."""
        node = StatisticsNode()
        
        # Create games with clear power differences
        games = [
            {
                "player1_id": 0, "player2_id": 1,
                "player1_action": "DEFECT", "player2_action": "COOPERATE",
                "player1_power_before": 50.0, "player2_power_before": 150.0,
                "player1_payoff": 5.0, "player2_payoff": 0.0,
                "player1_power_after": 55.0, "player2_power_after": 150.0
            },
            {
                "player1_id": 2, "player2_id": 3,
                "player1_action": "COOPERATE", "player2_action": "COOPERATE",
                "player1_power_before": 75.0, "player2_power_before": 125.0,
                "player1_payoff": 3.0, "player2_payoff": 3.0,
                "player1_power_after": 78.0, "player2_power_after": 128.0
            }
        ]
        
        stats = node.compute_round_statistics(games, 1)
        quartiles = stats["cooperation_by_power_quartile"]
        
        # Check quartile structure
        assert all(q in quartiles for q in ["Q1", "Q2", "Q3", "Q4"])
        assert all(0 <= quartiles[q] <= 1 for q in quartiles)
    
    @pytest.mark.asyncio
    async def test_handle_missing_rounds(self, basic_context):
        """Test handling of missing round data."""
        node = StatisticsNode()
        
        # Create stats with missing round
        per_round_stats = [
            {"round": 1, "cooperation_rate": 0.5, "average_payoff": 2.5, "power_stats": {"gini_coefficient": 0.2}},
            {"round": 3, "cooperation_rate": 0.6, "average_payoff": 3.0, "power_stats": {"gini_coefficient": 0.18}},  # Round 2 missing
        ]
        
        # Should handle gracefully
        trends = node.analyze_trends(per_round_stats)
        assert trends is not None
        assert "cooperation_trend" in trends
    
    @pytest.mark.asyncio
    async def test_statistical_report_generation(self, basic_context):
        """Test comprehensive statistical report generation."""
        node = StatisticsNode()
        
        # Create sample data
        per_round_stats = [
            {
                "round": i,
                "cooperation_rate": 0.5 + (i * 0.02),
                "defection_rate": 0.5 - (i * 0.02),
                "mutual_cooperation_rate": 0.3 + (i * 0.02),
                "mutual_defection_rate": 0.2 - (i * 0.01),
                "asymmetric_outcome_rate": 0.5 - (i * 0.01),
                "average_payoff": 2.5 + (i * 0.1),
                "payoff_variance": 0.5,
                "payoff_std": 0.7,
                "power_stats": {"gini_coefficient": 0.25 - (i * 0.01)},
                "cooperation_by_power_quartile": {"Q1": 0.4, "Q2": 0.5, "Q3": 0.6, "Q4": 0.7}
            }
            for i in range(1, 6)
        ]
        
        trend_analysis = node.analyze_trends(per_round_stats)
        node.detect_anomalies(per_round_stats)
        
        report = node.generate_statistics_report(
            "test_exp_001",
            per_round_stats,
            trend_analysis,
            node.anomalies
        )
        
        # Check report structure
        assert "statistical_analysis" in report
        stats = report["statistical_analysis"]
        
        assert stats["experiment_id"] == "test_exp_001"
        assert "analysis_timestamp" in stats
        assert "per_round_statistics" in stats
        assert "trend_analysis" in stats
        assert "anomalies_detected" in stats
        assert "experiment_summary" in stats
        assert "metadata" in stats
        
        # Check experiment summary
        summary = stats["experiment_summary"]
        assert summary["total_games"] == 225  # 5 rounds * 45 games per round
        assert "overall_cooperation_rate" in summary
        assert "cooperation_improvement" in summary
        assert "dominant_outcome" in summary
    
    @pytest.mark.asyncio
    async def test_integration_with_experiment(self, basic_context, data_manager, sample_game_data):
        """Test integration with experiment flow."""
        node = StatisticsNode()
        
        # Set up test data
        self.setup_test_data(data_manager, "test_exp_001", {1: sample_game_data})
        
        # Execute node
        result_context = await node.execute(basic_context)
        
        # Check context updated
        assert "statistical_analysis" in result_context
        
        # Check file saved
        exp_path = data_manager.get_experiment_path()
        analysis_file = exp_path / "statistical_analysis.json"
        assert analysis_file.exists()
    
    @pytest.mark.asyncio
    async def test_empty_game_data_handling(self):
        """Test handling of empty game data."""
        node = StatisticsNode()
        
        # Test with empty games list
        stats = node.compute_round_statistics([], 1)
        
        assert stats["round"] == 1
        assert stats["cooperation_rate"] == 0.0
        assert stats["average_payoff"] == 0.0
        assert stats["power_stats"]["gini_coefficient"] == 0.0
    
    @pytest.mark.asyncio
    async def test_single_round_statistics(self):
        """Test statistics calculation with only one round."""
        node = StatisticsNode()
        
        # Single round data
        per_round_stats = [{
            "round": 1,
            "cooperation_rate": 0.6,
            "average_payoff": 3.0,
            "power_stats": {"gini_coefficient": 0.25}
        }]
        
        trends = node.analyze_trends(per_round_stats)
        
        # Should handle single round gracefully
        assert trends["cooperation_trend"]["direction"] == "insufficient_data"
        assert trends["cooperation_trend"]["slope"] == 0.0
    
    @pytest.mark.asyncio
    async def test_correlation_calculations(self):
        """Test correlation calculations between payoffs and cooperation."""
        node = StatisticsNode()
        
        # Create data with clear positive correlation
        per_round_stats = []
        for i in range(10):
            coop_rate = 0.1 + (i * 0.08)  # Increases from 0.1 to 0.82
            avg_payoff = 1.0 + (coop_rate * 3)  # Payoff correlates with cooperation
            per_round_stats.append({
                "round": i + 1,
                "cooperation_rate": coop_rate,
                "average_payoff": avg_payoff,
                "power_stats": {"gini_coefficient": 0.2}
            })
        
        trends = node.analyze_trends(per_round_stats)
        
        correlation = trends["payoff_trend"]["correlation_with_cooperation"]
        assert correlation > 0.9  # Strong positive correlation


if __name__ == "__main__":
    pytest.main([__file__, "-v"])