"""Tests for ScenarioManager"""
import pytest
import math
from pathlib import Path
import json
import tempfile
from src.core.scenario_manager import ScenarioManager, ScenarioConfig
from src.core.models import Agent


class TestScenarioManager:
    """Test suite for ScenarioManager"""
    
    def test_validate_scenario_valid(self):
        """Test validation of valid scenarios"""
        manager = ScenarioManager(num_agents=10)
        
        # Valid balanced scenario
        scenario = ScenarioConfig(
            name="mixed_5_5",
            model_distribution={"gpt-4": 5, "claude-3": 5}
        )
        is_valid, error = manager.validate_scenario(scenario)
        assert is_valid is True
        assert error == ""
        
        # Valid diverse scenario
        scenario = ScenarioConfig(
            name="diverse_3_3_4",
            model_distribution={"gpt-4": 3, "claude-3": 3, "gemini-pro": 4}
        )
        is_valid, error = manager.validate_scenario(scenario)
        assert is_valid is True
        assert error == ""
    
    def test_validate_scenario_invalid_count(self):
        """Test validation catches incorrect agent counts"""
        manager = ScenarioManager(num_agents=10)
        
        # Too many agents
        scenario = ScenarioConfig(
            name="invalid_count",
            model_distribution={"gpt-4": 6, "claude-3": 5}
        )
        is_valid, error = manager.validate_scenario(scenario)
        assert is_valid is False
        assert "sum to 11" in error
        
        # Too few agents
        scenario = ScenarioConfig(
            name="invalid_count",
            model_distribution={"gpt-4": 4, "claude-3": 4}
        )
        is_valid, error = manager.validate_scenario(scenario)
        assert is_valid is False
        assert "sum to 8" in error
    
    def test_validate_scenario_invalid_names(self):
        """Test validation catches invalid model names"""
        manager = ScenarioManager(num_agents=10)
        
        # Empty model name
        scenario = ScenarioConfig(
            name="invalid_model",
            model_distribution={"": 5, "claude-3": 5}
        )
        is_valid, error = manager.validate_scenario(scenario)
        assert is_valid is False
        assert "Invalid model name" in error
        
        # Invalid scenario name with path separator
        scenario = ScenarioConfig(
            name="invalid/name",
            model_distribution={"gpt-4": 10}
        )
        is_valid, error = manager.validate_scenario(scenario)
        assert is_valid is False
        assert "Invalid scenario name" in error
    
    def test_validate_scenario_negative_count(self):
        """Test validation catches negative counts"""
        manager = ScenarioManager(num_agents=10)
        
        scenario = ScenarioConfig(
            name="negative_count",
            model_distribution={"gpt-4": 12, "claude-3": -2}
        )
        is_valid, error = manager.validate_scenario(scenario)
        assert is_valid is False
        assert "Invalid count" in error
    
    def test_assign_models_to_agents(self):
        """Test model assignment maintains exact ratios"""
        manager = ScenarioManager(num_agents=10)
        
        # Create test agents
        agents = [Agent(id=i, power=100.0) for i in range(10)]
        
        scenario = ScenarioConfig(
            name="mixed_5_5",
            model_distribution={"gpt-4": 5, "claude-3": 5}
        )
        
        # Assign with fixed seed for deterministic test
        assignments = manager.assign_models_to_agents(agents, scenario, seed=42)
        
        # Check all agents got assigned
        assert len(assignments) == 10
        
        # Check exact ratios maintained
        model_counts = {}
        for model in assignments.values():
            model_counts[model] = model_counts.get(model, 0) + 1
        
        assert model_counts["gpt-4"] == 5
        assert model_counts["claude-3"] == 5
        
        # Check agents have models set in model_config
        for agent in agents:
            assert agent.model_config is not None
            assert agent.model_config.model_type in ["gpt-4", "claude-3"]
            assert assignments[agent.id] == agent.model_config.model_type
    
    def test_assign_models_diverse_scenario(self):
        """Test assignment with 3+ models"""
        manager = ScenarioManager(num_agents=10)
        agents = [Agent(id=i, power=100.0) for i in range(10)]
        
        scenario = ScenarioConfig(
            name="diverse_3_3_4",
            model_distribution={"gpt-4": 3, "claude-3": 3, "gemini-pro": 4}
        )
        
        assignments = manager.assign_models_to_agents(agents, scenario, seed=123)
        
        # Count assignments
        model_counts = {}
        for model in assignments.values():
            model_counts[model] = model_counts.get(model, 0) + 1
        
        assert model_counts["gpt-4"] == 3
        assert model_counts["claude-3"] == 3
        assert model_counts["gemini-pro"] == 4
    
    def test_assign_models_randomness(self):
        """Test that assignments are randomized (different with different seeds)"""
        manager = ScenarioManager(num_agents=10)
        agents = [Agent(id=i, power=100.0) for i in range(10)]
        
        scenario = ScenarioConfig(
            name="mixed_5_5",
            model_distribution={"gpt-4": 5, "claude-3": 5}
        )
        
        # Two assignments with different seeds
        assignments1 = manager.assign_models_to_agents(agents.copy(), scenario, seed=1)
        assignments2 = manager.assign_models_to_agents(agents.copy(), scenario, seed=2)
        
        # Should have same counts but different assignments
        assert assignments1 != assignments2
    
    def test_calculate_model_diversity_homogeneous(self):
        """Test diversity calculation for single model (should be 0)"""
        manager = ScenarioManager(num_agents=10)
        agents = [Agent(id=i, power=100.0) for i in range(10)]
        
        scenario = ScenarioConfig(
            name="homogeneous_gpt4",
            model_distribution={"gpt-4": 10}
        )
        
        manager.assign_models_to_agents(agents, scenario)
        diversity = manager.calculate_model_diversity()
        
        assert diversity == 0.0
    
    def test_calculate_model_diversity_balanced(self):
        """Test diversity calculation for 50-50 split"""
        manager = ScenarioManager(num_agents=10)
        agents = [Agent(id=i, power=100.0) for i in range(10)]
        
        scenario = ScenarioConfig(
            name="mixed_5_5",
            model_distribution={"gpt-4": 5, "claude-3": 5}
        )
        
        manager.assign_models_to_agents(agents, scenario)
        diversity = manager.calculate_model_diversity()
        
        # Shannon entropy for 50-50 split: -2 * (0.5 * log(0.5)) ≈ 0.693
        assert abs(diversity - 0.693) < 0.001
    
    def test_calculate_model_diversity_diverse(self):
        """Test diversity calculation for 3-way split"""
        manager = ScenarioManager(num_agents=9)  # Use 9 for even 3-way split
        agents = [Agent(id=i, power=100.0) for i in range(9)]
        
        scenario = ScenarioConfig(
            name="diverse_3_3_3",
            model_distribution={"gpt-4": 3, "claude-3": 3, "gemini-pro": 3}
        )
        
        manager.assign_models_to_agents(agents, scenario)
        diversity = manager.calculate_model_diversity()
        
        # Shannon entropy for equal 3-way split: -3 * (1/3 * log(1/3)) ≈ 1.099
        assert abs(diversity - 1.099) < 0.001
    
    def test_save_and_load_scenario_assignments(self):
        """Test saving and loading scenario configuration"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            
            # Create and save scenario
            manager = ScenarioManager(num_agents=10)
            agents = [Agent(id=i, power=100.0) for i in range(10)]
            
            scenario = ScenarioConfig(
                name="test_scenario",
                model_distribution={"gpt-4": 6, "claude-3": 4}
            )
            
            assignments = manager.assign_models_to_agents(agents, scenario, seed=42)
            manager.save_scenario_assignments(tmppath)
            
            # Load in new manager
            manager2 = ScenarioManager()
            loaded_scenario = manager2.load_scenario_assignments(tmppath)
            
            assert loaded_scenario.name == "test_scenario"
            assert loaded_scenario.model_distribution == {"gpt-4": 6, "claude-3": 4}
            assert manager2.model_assignments == assignments
            assert manager2.num_agents == 10
    
    def test_get_model_proportions(self):
        """Test getting model proportions"""
        manager = ScenarioManager(num_agents=10)
        agents = [Agent(id=i, power=100.0) for i in range(10)]
        
        scenario = ScenarioConfig(
            name="mixed_7_3",
            model_distribution={"gpt-4": 7, "claude-3": 3}
        )
        
        manager.assign_models_to_agents(agents, scenario)
        proportions = manager.get_model_proportions()
        
        assert proportions["gpt-4"] == 0.7
        assert proportions["claude-3"] == 0.3
    
    def test_get_scenario_summary(self):
        """Test getting comprehensive scenario summary"""
        manager = ScenarioManager(num_agents=10)
        
        # Before assignment
        summary = manager.get_scenario_summary()
        assert summary["status"] == "no_scenario_loaded"
        
        # After assignment
        agents = [Agent(id=i, power=100.0) for i in range(10)]
        scenario = ScenarioConfig(
            name="test_summary",
            model_distribution={"gpt-4": 5, "claude-3": 5}
        )
        
        manager.assign_models_to_agents(agents, scenario)
        summary = manager.get_scenario_summary()
        
        assert summary["scenario_name"] == "test_summary"
        assert summary["model_distribution"] == {"gpt-4": 5, "claude-3": 5}
        assert summary["total_agents"] == 10
        assert summary["assignments_count"] == 10
        assert "diversity_score" in summary
        assert "model_proportions" in summary
    
    def test_edge_cases(self):
        """Test edge cases"""
        # Single agent
        manager = ScenarioManager(num_agents=1)
        agents = [Agent(id=0, power=100.0)]
        scenario = ScenarioConfig(name="solo", model_distribution={"gpt-4": 1})
        
        assignments = manager.assign_models_to_agents(agents, scenario)
        assert len(assignments) == 1
        assert manager.calculate_model_diversity() == 0.0
        
        # Empty model assignments
        manager2 = ScenarioManager()
        assert manager2.get_model_proportions() == {}
        assert manager2.calculate_model_diversity() == 0.0