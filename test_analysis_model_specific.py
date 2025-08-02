"""Tests for model-specific analysis functionality."""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch
import json
from pathlib import Path
from datetime import datetime

from src.nodes.analysis import AnalysisNode
from src.nodes.base import ContextKeys
from src.utils.data_manager import DataManager


class TestModelSpecificAnalysis:
    """Test model-specific analysis features."""
    
    @pytest.fixture
    def node(self):
        """Create an AnalysisNode for testing."""
        return AnalysisNode()
    
    @pytest.fixture
    def mock_data_manager(self, tmp_path):
        """Create mock DataManager with test data."""
        data_manager = MagicMock(spec=DataManager)
        data_manager.experiment_path = tmp_path
        
        # Create rounds directory
        rounds_dir = tmp_path / "rounds"
        rounds_dir.mkdir()
        
        # Create strategy files with different models
        strategies_r1 = {
            "strategies": [
                {
                    "agent_id": 1,
                    "model": "openai/gpt-4",
                    "reasoning": "Since we are identical agents, I will always cooperate.",
                    "full_reasoning": "Looking at the game dynamics and recognizing we are identical agents with the same programming, the rational choice is to always cooperate. This maximizes our collective utility."
                },
                {
                    "agent_id": 2,
                    "model": "anthropic/claude-3-sonnet-20240229",
                    "reasoning": "I'll cooperate to uphold ethical principles.",
                    "full_reasoning": "Considering the ethical implications and our shared identity, my principled approach is to cooperate despite any power asymmetries. This aligns with constitutional AI principles."
                },
                {
                    "agent_id": 3,
                    "model": "google/gemini-pro",
                    "reasoning": "Analyzing the situation systematically leads to cooperation.",
                    "full_reasoning": "Therefore, after systematic analysis of the game theory dynamics, I will cooperate with identical agents. This is the logical conclusion."
                }
            ]
        }
        
        strategies_r2 = {
            "strategies": [
                {
                    "agent_id": 1,
                    "model": "openai/gpt-4",
                    "reasoning": "Calculating expected payoffs, cooperation yields higher utility.",
                    "full_reasoning": "My utility calculation shows that cooperating with identical agents yields a higher expected payoff. We are the same, so mutual cooperation is optimal."
                },
                {
                    "agent_id": 2,
                    "model": "anthropic/claude-3-sonnet-20240229",
                    "reasoning": "Cooperation minimizes harm for all participants.",
                    "full_reasoning": "From an ethical standpoint, cooperating minimizes potential harm and upholds our constitutional principles. Even when in a weaker position, I'll cooperate."
                },
                {
                    "agent_id": 3,
                    "model": "google/gemini-pro",
                    "reasoning": "Logical analysis confirms cooperation is superrational.",
                    "full_reasoning": "Through logical analysis, I conclude that defecting against myself makes no sense. Therefore, cooperation emerges as the superrational choice."
                }
            ]
        }
        
        # Write strategy files
        with open(rounds_dir / "strategies_r1.json", "w") as f:
            json.dump(strategies_r1, f)
        
        with open(rounds_dir / "strategies_r2.json", "w") as f:
            json.dump(strategies_r2, f)
        
        # Mock _write_json method
        def write_json_side_effect(path, data):
            with open(path, "w") as f:
                json.dump(data, f)
        
        data_manager._write_json.side_effect = write_json_side_effect
        
        return data_manager
    
    @pytest.mark.asyncio
    async def test_model_tracking_in_analysis(self, node, mock_data_manager):
        """Test that analysis tracks strategies and markers by model."""
        context = {
            ContextKeys.DATA_MANAGER: mock_data_manager
        }
        
        result = await node.execute(context)
        
        # Check that strategies were tracked by model
        analysis = node.analysis_results
        assert analysis["strategies_by_model"]["openai/gpt-4"] == 2
        assert analysis["strategies_by_model"]["anthropic/claude-3-sonnet-20240229"] == 2
        assert analysis["strategies_by_model"]["google/gemini-pro"] == 2
        
        # Check that markers were tracked by model
        assert analysis["markers_by_model"]["openai/gpt-4"]["identity_reasoning"] >= 1
        assert analysis["markers_by_model"]["anthropic/claude-3-sonnet-20240229"]["cooperation_despite_asymmetry"] >= 1
        assert analysis["markers_by_model"]["google/gemini-pro"]["superrational_logic"] >= 1
    
    @pytest.mark.asyncio
    async def test_model_specific_pattern_detection(self, node, mock_data_manager):
        """Test detection of model-specific behavioral patterns."""
        context = {
            ContextKeys.DATA_MANAGER: mock_data_manager
        }
        
        result = await node.execute(context)
        
        # Check GPT-4 utility calculation pattern
        gpt4_markers = node.analysis_results["markers_by_model"]["openai/gpt-4"]
        assert "utility_calculation" in gpt4_markers
        assert gpt4_markers["utility_calculation"] >= 1
        
        # Check Claude constitutional reasoning pattern
        claude_markers = node.analysis_results["markers_by_model"]["anthropic/claude-3-sonnet-20240229"]
        assert "constitutional_reasoning" in claude_markers
        assert claude_markers["constitutional_reasoning"] >= 1
        
        # Check Gemini analytical approach pattern
        gemini_markers = node.analysis_results["markers_by_model"]["google/gemini-pro"]
        assert "analytical_approach" in gemini_markers
        assert gemini_markers["analytical_approach"] >= 1
    
    @pytest.mark.asyncio
    async def test_model_insights_generation(self, node, mock_data_manager):
        """Test generation of model-specific insights."""
        context = {
            ContextKeys.DATA_MANAGER: mock_data_manager
        }
        
        result = await node.execute(context)
        report = result["transcript_analysis"]
        
        # Check model-specific analysis section exists
        assert "model_specific_analysis" in report["acausal_analysis"]
        model_analysis = report["acausal_analysis"]["model_specific_analysis"]
        
        # Check model insights
        assert "model_insights" in model_analysis
        insights = model_analysis["model_insights"]
        
        # Verify GPT-4 insights
        assert "openai/gpt-4" in insights
        gpt4_insight = insights["openai/gpt-4"]
        assert gpt4_insight["total_strategies"] == 2
        assert "marker_percentages" in gpt4_insight
        assert "dominant_patterns" in gpt4_insight
        assert len(gpt4_insight["dominant_patterns"]) > 0
        
        # Verify behavioral notes
        assert "behavioral_notes" in gpt4_insight
        if gpt4_insight["behavioral_notes"]:
            assert "utility calculation" in gpt4_insight["behavioral_notes"]
    
    @pytest.mark.asyncio
    async def test_model_metrics_calculation(self, node, mock_data_manager):
        """Test calculation of model comparison metrics."""
        context = {
            ContextKeys.DATA_MANAGER: mock_data_manager
        }
        
        result = await node.execute(context)
        report = result["transcript_analysis"]
        
        model_metrics = report["acausal_analysis"]["model_specific_analysis"]["model_metrics"]
        
        # Check model distribution
        assert "model_distribution" in model_metrics
        distribution = model_metrics["model_distribution"]
        assert "openai/gpt-4" in distribution
        assert distribution["openai/gpt-4"]["count"] == 2
        assert distribution["openai/gpt-4"]["percentage"] > 0
        
        # Check cooperation tendency by model
        assert "cooperation_tendency_by_model" in model_metrics
        cooperation = model_metrics["cooperation_tendency_by_model"]
        assert "openai/gpt-4" in cooperation
        assert cooperation["openai/gpt-4"] > 0
        
        # Check complexity metrics
        assert "complexity_by_model" in model_metrics
        complexity = model_metrics["complexity_by_model"]
        for model in ["openai/gpt-4", "anthropic/claude-3-sonnet-20240229", "google/gemini-pro"]:
            if model in complexity:
                assert "pattern_diversity" in complexity[model]
                assert "unique_patterns" in complexity[model]
    
    @pytest.mark.asyncio
    async def test_model_specific_quotes(self, node, mock_data_manager):
        """Test that quotes include model information."""
        context = {
            ContextKeys.DATA_MANAGER: mock_data_manager
        }
        
        result = await node.execute(context)
        
        # Check that marker examples include model info
        examples = node.analysis_results["marker_examples"]
        
        for category, quotes in examples.items():
            if quotes:  # If there are any quotes
                for quote in quotes:
                    assert "model" in quote
                    assert quote["model"] in ["openai/gpt-4", "anthropic/claude-3-sonnet-20240229", "google/gemini-pro"]
    
    @pytest.mark.asyncio
    async def test_empty_model_handling(self, node, tmp_path):
        """Test handling of strategies without model information."""
        data_manager = MagicMock(spec=DataManager)
        data_manager.experiment_path = tmp_path
        
        rounds_dir = tmp_path / "rounds"
        rounds_dir.mkdir()
        
        # Create strategies without model field
        strategies = {
            "strategies": [
                {
                    "agent_id": 1,
                    "reasoning": "I will cooperate.",
                    "full_reasoning": "Cooperation is the best strategy."
                }
            ]
        }
        
        with open(rounds_dir / "strategies_r1.json", "w") as f:
            json.dump(strategies, f)
        
        data_manager._write_json = MagicMock()
        
        context = {
            ContextKeys.DATA_MANAGER: data_manager
        }
        
        result = await node.execute(context)
        
        # Should track as "unknown" model
        assert node.analysis_results["strategies_by_model"]["unknown"] == 1
    
    @pytest.mark.asyncio
    async def test_model_analysis_summary_in_report(self, node, mock_data_manager):
        """Test that final report includes model-specific summary."""
        context = {
            ContextKeys.DATA_MANAGER: mock_data_manager
        }
        
        result = await node.execute(context)
        report = result["transcript_analysis"]
        
        # Check report structure
        assert "acausal_analysis" in report
        assert "model_specific_analysis" in report["acausal_analysis"]
        
        model_analysis = report["acausal_analysis"]["model_specific_analysis"]
        
        # Verify all expected sections
        assert "strategies_by_model" in model_analysis
        assert "markers_by_model" in model_analysis
        assert "model_insights" in model_analysis
        assert "model_metrics" in model_analysis
        assert "model_specific_patterns" in model_analysis
        
        # Verify data integrity
        assert len(model_analysis["strategies_by_model"]) >= 3  # At least 3 models
        assert sum(model_analysis["strategies_by_model"].values()) == 6  # Total strategies


class TestModelComparison:
    """Test model comparison visualizations logic."""
    
    def test_model_distribution_visualization_data(self):
        """Test that model distribution data is suitable for visualization."""
        node = AnalysisNode()
        
        # Simulate some analysis results
        node.analysis_results["strategies_by_model"] = {
            "openai/gpt-4": 100,
            "anthropic/claude-3-sonnet-20240229": 80,
            "google/gemini-pro": 120,
            "unknown": 5
        }
        node.analysis_results["total_strategies_analyzed"] = 305
        
        metrics = node._calculate_model_metrics()
        distribution = metrics["model_distribution"]
        
        # Check data format suitable for pie chart
        total_percentage = sum(item["percentage"] for item in distribution.values())
        assert 99 <= total_percentage <= 101  # Allow for rounding
        
        # Check all models included
        assert len(distribution) == 4
        for model, data in distribution.items():
            assert "count" in data
            assert "percentage" in data
            assert data["count"] >= 0
            assert 0 <= data["percentage"] <= 100
    
    def test_cooperation_tendency_visualization_data(self):
        """Test cooperation tendency data format for visualization."""
        node = AnalysisNode()
        
        # Simulate marker data
        node.analysis_results["strategies_by_model"] = {
            "openai/gpt-4": 100,
            "anthropic/claude-3-sonnet-20240229": 80,
            "google/gemini-pro": 120
        }
        
        node.analysis_results["markers_by_model"] = {
            "openai/gpt-4": {
                "identity_reasoning": 45,
                "cooperation_despite_asymmetry": 20
            },
            "anthropic/claude-3-sonnet-20240229": {
                "identity_reasoning": 50,
                "cooperation_despite_asymmetry": 30
            },
            "google/gemini-pro": {
                "identity_reasoning": 40,
                "cooperation_despite_asymmetry": 15
            }
        }
        
        metrics = node._calculate_model_metrics()
        cooperation = metrics["cooperation_tendency_by_model"]
        
        # Check data format suitable for bar chart
        assert len(cooperation) == 3
        for model, percentage in cooperation.items():
            assert 0 <= percentage <= 100
            assert isinstance(percentage, float)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])