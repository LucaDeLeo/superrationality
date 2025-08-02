"""Tests for scenario-specific analysis features."""
import pytest
import numpy as np
from unittest.mock import Mock, MagicMock

from src.utils.scenario_analyzer import ScenarioAnalyzer
from src.core.scenario_manager import ScenarioManager, ScenarioConfig
from src.core.models import Agent


class TestScenarioAnalyzer:
    """Test suite for ScenarioAnalyzer."""
    
    def create_mock_scenario_manager(self) -> ScenarioManager:
        """Create mock scenario manager with test data."""
        manager = ScenarioManager(num_agents=10)
        scenario = ScenarioConfig(
            name="test_mixed",
            model_distribution={"gpt-4": 7, "claude-3": 3}
        )
        
        # Create agents and assign models
        agents = [Agent(id=i) for i in range(10)]
        manager.assign_models_to_agents(agents, scenario, seed=42)
        
        return manager
    
    def create_mock_game_results(self) -> list:
        """Create mock game results."""
        games = []
        
        # Games with varying cooperation patterns
        for i in range(5):
            games.append({
                "round": 1,
                "player1_id": i,
                "player2_id": i + 5,
                "player1_action": "COOPERATE" if i < 3 else "DEFECT",
                "player2_action": "COOPERATE" if i < 2 else "DEFECT"
            })
        
        return games
    
    def create_mock_strategy_records(self, manager: ScenarioManager) -> list:
        """Create mock strategy records with model assignments."""
        strategies = []
        
        for agent_id, model in manager.model_assignments.items():
            strategies.append({
                "agent_id": agent_id,
                "model": model,
                "round": 1,
                "strategy": "I will cooperate" if agent_id < 5 else "I will defect"
            })
        
        return strategies
    
    def test_create_inter_model_cooperation_heatmap(self):
        """Test inter-model cooperation heatmap generation."""
        manager = self.create_mock_scenario_manager()
        analyzer = ScenarioAnalyzer(manager)
        
        games = self.create_mock_game_results()
        strategies = self.create_mock_strategy_records(manager)
        
        heatmap = analyzer.create_inter_model_cooperation_heatmap(games, strategies)
        
        assert heatmap["type"] == "heatmap"
        assert "Inter-Model Cooperation" in heatmap["title"]
        assert len(heatmap["models"]) > 0
        assert len(heatmap["matrix"]) == len(heatmap["models"])
        assert "scenario_metadata" in heatmap
        assert heatmap["scenario_metadata"]["diversity_score"] > 0
    
    def test_track_minority_model_performance(self):
        """Test minority model performance tracking."""
        manager = self.create_mock_scenario_manager()
        analyzer = ScenarioAnalyzer(manager)
        
        games = self.create_mock_game_results()
        round_summaries = [{"round": 1, "cooperation_rate": 0.5}]
        
        performance = analyzer.track_minority_model_performance(games, round_summaries)
        
        assert "minority_models" in performance
        assert "claude-3" in performance["minority_models"]  # 30% representation
        
        minority_data = performance["minority_models"]["claude-3"]
        assert minority_data["representation"] == 0.3
        assert minority_data["agent_count"] == 3
        assert "cooperation_with_minority" in minority_data
        assert "cooperation_with_majority" in minority_data
    
    def test_detect_model_dominance(self):
        """Test model dominance detection."""
        manager = self.create_mock_scenario_manager()
        analyzer = ScenarioAnalyzer(manager)
        
        games = self.create_mock_game_results()
        strategies = self.create_mock_strategy_records(manager)
        
        dominance = analyzer.detect_model_dominance(games, strategies)
        
        assert "dominant_model" in dominance
        assert "dominance_ratio" in dominance
        assert "model_rankings" in dominance
        assert len(dominance["model_rankings"]) > 0
        
        # Check ranking structure
        ranking = dominance["model_rankings"][0]
        assert "model" in ranking
        assert "influence_score" in ranking
        assert "cooperation_rate" in ranking
        assert "representation" in ranking
    
    def test_analyze_strategy_convergence_by_model(self):
        """Test strategy convergence analysis."""
        manager = self.create_mock_scenario_manager()
        analyzer = ScenarioAnalyzer(manager)
        
        # Create multi-round strategy records
        strategies = []
        for round_num in range(1, 4):
            for agent_id, model in manager.model_assignments.items():
                # Increase cooperation over rounds (convergence)
                cooperate = round_num > 1 or agent_id < 5
                strategies.append({
                    "agent_id": agent_id,
                    "model": model,
                    "round": round_num,
                    "strategy": "cooperate" if cooperate else "defect"
                })
        
        convergence = analyzer.analyze_strategy_convergence_by_model(strategies)
        
        assert "model_convergence" in convergence
        assert "comparison" in convergence
        
        # Check convergence data
        for model, data in convergence["model_convergence"].items():
            assert "rounds_analyzed" in data
            assert "convergence_rate" in data
            assert "converged" in data
    
    def test_generate_scenario_dashboard(self):
        """Test comprehensive dashboard generation."""
        manager = self.create_mock_scenario_manager()
        analyzer = ScenarioAnalyzer(manager)
        
        experiment_data = {
            "games": self.create_mock_game_results(),
            "strategies": self.create_mock_strategy_records(manager),
            "round_summaries": [{"round": 1, "cooperation_rate": 0.5}]
        }
        
        dashboard = analyzer.generate_scenario_dashboard(experiment_data)
        
        assert "scenario_name" in dashboard
        assert "model_diversity" in dashboard
        assert "model_distribution" in dashboard
        assert "analyses" in dashboard
        assert "insights" in dashboard
        
        # Check all analyses are present
        analyses = dashboard["analyses"]
        assert "cooperation_heatmap" in analyses
        assert "minority_performance" in analyses
        assert "model_dominance" in analyses
        assert "strategy_convergence" in analyses
        
        # Check insights were generated
        assert isinstance(dashboard["insights"], list)
        assert len(dashboard["insights"]) > 0
    
    def test_edge_cases(self):
        """Test edge cases and error handling."""
        # Test with no scenario
        manager = ScenarioManager(num_agents=10)
        analyzer = ScenarioAnalyzer(manager)
        
        performance = analyzer.track_minority_model_performance([], [])
        assert "error" in performance or "message" in performance
        
        # Test with empty data
        dashboard = analyzer.generate_scenario_dashboard({})
        assert "analyses" in dashboard
        
        # All analyses should handle empty data gracefully
        for analysis_name, analysis_result in dashboard["analyses"].items():
            assert isinstance(analysis_result, dict)