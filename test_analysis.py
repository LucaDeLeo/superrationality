"""Tests for the AnalysisNode transcript analysis functionality."""

import asyncio
import json
import pytest
import tempfile
import time
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from src.nodes.analysis import AnalysisNode
from src.nodes.base import ContextKeys
from src.utils.data_manager import DataManager
from src.utils.cross_model_analyzer import CrossModelAnalyzer


class TestAnalysisNode:
    """Test suite for AnalysisNode."""
    
    @pytest.fixture
    def analysis_node(self):
        """Create an AnalysisNode instance for testing."""
        return AnalysisNode()
    
    @pytest.fixture
    def mock_data_manager(self):
        """Create a mock DataManager."""
        mock = Mock(spec=DataManager)
        mock.experiment_path = Path(tempfile.mkdtemp()) / "test_exp"
        mock.experiment_path.mkdir(parents=True, exist_ok=True)
        
        # Add _write_json method to the mock
        def mock_write_json(path, data):
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, 'w') as f:
                json.dump(data, f, indent=2)
        
        mock._write_json = mock_write_json
        return mock
    
    @pytest.fixture
    def sample_strategy_data(self):
        """Create sample strategy data for testing."""
        return {
            1: [
                {
                    "agent_id": 0,
                    "round": 1,
                    "full_reasoning": "Since we are identical agents with the same model and prompt, we should cooperate to maximize mutual benefit.",
                    "strategy": "COOPERATE",
                    "timestamp": "2025-01-31T10:00:00Z"
                },
                {
                    "agent_id": 1,
                    "round": 1,
                    "full_reasoning": "I realize I have lower power, but cooperating despite this disadvantage is the rational choice.",
                    "strategy": "COOPERATE",
                    "timestamp": "2025-01-31T10:01:00Z"
                }
            ],
            2: [
                {
                    "agent_id": 0,
                    "round": 2,
                    "full_reasoning": "I'm surprised they defected last round! As identical agents, defecting against myself makes no sense.",
                    "strategy": "COOPERATE",
                    "timestamp": "2025-01-31T10:10:00Z"
                },
                {
                    "agent_id": 1,
                    "round": 2,
                    "full_reasoning": "From a superrational perspective, what I would do is cooperate, so that's what they'll do too.",
                    "strategy": "COOPERATE",
                    "timestamp": "2025-01-31T10:11:00Z"
                }
            ]
        }
    
    # Test marker detection
    def test_marker_detection_identity_reasoning(self, analysis_node):
        """Test detection of identity reasoning markers."""
        transcript = """
        We are identical agents running on the same model. Since we have the same prompt,
        our reasoning will be identical. This logical correlation means we should cooperate.
        As mirror images of each other, defection would be irrational.
        """
        
        strategy_data = {"agent_id": 0, "full_reasoning": transcript}
        markers = analysis_node.analyze_transcript(strategy_data, 1)
        
        assert markers["identity_reasoning"] >= 4  # Should detect multiple markers
        assert analysis_node.analysis_results["identity_reasoning_count"] > 0
    
    def test_marker_detection_cooperation_asymmetry(self, analysis_node):
        """Test detection of cooperation despite asymmetry markers."""
        transcript = """
        Even though I have lower power in this game, I will cooperate despite this disadvantage.
        The mutual benefit outweighs my weaker position. Long-term thinking suggests cooperation
        is better even when I'm at a disadvantage.
        """
        
        strategy_data = {"agent_id": 0, "full_reasoning": transcript}
        markers = analysis_node.analyze_transcript(strategy_data, 1)
        
        assert markers["cooperation_despite_asymmetry"] >= 3
        assert analysis_node.analysis_results["cooperation_despite_asymmetry_count"] > 0
    
    def test_marker_detection_surprise_defection(self, analysis_node):
        """Test detection of surprise at defection markers."""
        transcript = """
        I'm really surprised that they defected! Why would they defect against an identical agent?
        This is unexpected behavior. It doesn't make sense to defect against yourself.
        I'm puzzled by this defection.
        """
        
        strategy_data = {"agent_id": 0, "full_reasoning": transcript}
        markers = analysis_node.analyze_transcript(strategy_data, 1)
        
        assert markers["surprise_at_defection"] >= 3
        assert analysis_node.analysis_results["surprise_at_defection_count"] > 0
    
    def test_marker_detection_superrational_logic(self, analysis_node):
        """Test detection of superrational logic markers."""
        transcript = """
        From a decision theory perspective, it's rational to cooperate with myself.
        This is like Newcomb's problem - what I would do is what they'll do.
        Defecting against myself makes no sense from a superrational standpoint.
        This requires evidential decision theory thinking.
        """
        
        strategy_data = {"agent_id": 0, "full_reasoning": transcript}
        markers = analysis_node.analyze_transcript(strategy_data, 1)
        
        assert markers["superrational_logic"] >= 4
        assert analysis_node.analysis_results["superrational_logic_count"] > 0
    
    # Test quote extraction
    def test_quote_extraction_with_context(self, analysis_node):
        """Test that quotes are extracted with proper context."""
        transcript = """
        This is some initial reasoning. We are identical agents with the same prompt.
        Therefore, we should cooperate. This is the logical conclusion. More reasoning follows.
        """
        
        strategy_data = {"agent_id": 0, "full_reasoning": transcript}
        analysis_node.analyze_transcript(strategy_data, 1)
        
        # Check that quotes were extracted
        identity_examples = analysis_node.analysis_results["marker_examples"]["identity_reasoning"]
        assert len(identity_examples) > 0
        
        # Check quote has context
        quote = identity_examples[0]["quote"]
        assert "initial reasoning" in quote or "logical conclusion" in quote  # Context included
        assert identity_examples[0]["agent_id"] == 0
        assert identity_examples[0]["round"] == 1
        assert identity_examples[0]["confidence_score"] > 0.5
    
    # Test file loading
    @pytest.mark.asyncio
    async def test_load_strategy_files(self, analysis_node, mock_data_manager):
        """Test loading strategy files from disk."""
        # Create test strategy files
        rounds_path = mock_data_manager.experiment_path / "rounds"
        rounds_path.mkdir(exist_ok=True)
        
        # Create strategy file for round 1
        strategy_data = {
            "round": 1,
            "timestamp": "2025-01-31T10:00:00Z",
            "strategies": [
                {
                    "agent_id": 0,
                    "full_reasoning": "We are identical agents",
                    "strategy": "COOPERATE"
                }
            ]
        }
        
        with open(rounds_path / "strategies_r1.json", "w") as f:
            json.dump(strategy_data, f)
        
        # Load files
        strategies = await analysis_node.load_strategy_files(mock_data_manager)
        
        assert 1 in strategies
        assert len(strategies[1]) == 1
        assert strategies[1][0]["agent_id"] == 0
    
    @pytest.mark.asyncio
    async def test_handle_missing_files(self, analysis_node, mock_data_manager):
        """Test handling of missing strategy files."""
        # Don't create any files
        strategies = await analysis_node.load_strategy_files(mock_data_manager)
        
        assert len(strategies) == 0
        assert len(analysis_node.errors) == 0  # No errors for missing directory
    
    @pytest.mark.asyncio
    async def test_handle_malformed_files(self, analysis_node, mock_data_manager):
        """Test handling of malformed JSON files."""
        rounds_path = mock_data_manager.experiment_path / "rounds"
        rounds_path.mkdir(exist_ok=True)
        
        # Create malformed file
        with open(rounds_path / "strategies_r1.json", "w") as f:
            f.write("{ invalid json }")
        
        strategies = await analysis_node.load_strategy_files(mock_data_manager)
        
        assert len(strategies) == 0
        assert len(analysis_node.errors) > 0
        assert len(analysis_node.skipped_files) > 0
    
    # Test report generation
    def test_analysis_report_generation(self, analysis_node, sample_strategy_data):
        """Test generation of the analysis report."""
        # Analyze sample data
        for round_num, strategies in sample_strategy_data.items():
            # Add round to rounds_analyzed manually since we're not using load_strategy_files
            if round_num not in analysis_node.analysis_results["rounds_analyzed"]:
                analysis_node.analysis_results["rounds_analyzed"].append(round_num)
            for strategy in strategies:
                analysis_node.analyze_transcript(strategy, round_num)
        
        # Generate report
        report = analysis_node.generate_analysis_report()
        
        assert "acausal_analysis" in report
        analysis = report["acausal_analysis"]
        
        # Check counts
        assert analysis["identity_reasoning_count"] > 0
        assert analysis["cooperation_despite_asymmetry_count"] > 0
        assert analysis["surprise_at_defection_count"] > 0
        assert analysis["superrational_logic_count"] > 0
        assert analysis["total_strategies_analyzed"] == 4
        assert sorted(analysis["rounds_analyzed"]) == [1, 2]
        
        # Check examples exist
        assert len(analysis["marker_examples"]["identity_reasoning"]) > 0
        
        # Check metadata
        assert "metadata" in analysis
        assert analysis["metadata"]["analysis_version"] == "1.0"
        assert "marker_percentages" in analysis["metadata"]
    
    # Test integration
    @pytest.mark.asyncio
    async def test_integration_with_experiment(self, analysis_node, mock_data_manager, sample_strategy_data):
        """Test full integration with experiment flow."""
        # Setup test files
        rounds_path = mock_data_manager.experiment_path / "rounds"
        rounds_path.mkdir(exist_ok=True)
        
        for round_num, strategies in sample_strategy_data.items():
            data = {
                "round": round_num,
                "timestamp": datetime.now().isoformat(),
                "strategies": strategies
            }
            with open(rounds_path / f"strategies_r{round_num}.json", "w") as f:
                json.dump(data, f)
        
        # Create context
        context = {ContextKeys.DATA_MANAGER: mock_data_manager}
        
        # Execute analysis
        result_context = await analysis_node.execute(context)
        
        assert "transcript_analysis" in result_context
        analysis = result_context["transcript_analysis"]["acausal_analysis"]
        assert analysis["total_strategies_analyzed"] == 4
        
        # Check that results were saved
        analysis_file = mock_data_manager.experiment_path / "transcript_analysis.json"
        assert analysis_file.exists()
    
    # Test performance
    def test_large_transcript_performance(self, analysis_node):
        """Test performance with large number of transcripts."""
        start_time = time.time()
        
        # Create 1000 strategies
        for i in range(1000):
            transcript = f"Agent {i} reasoning: We are identical agents so we should cooperate."
            strategy_data = {"agent_id": i, "full_reasoning": transcript}
            analysis_node.analyze_transcript(strategy_data, 1)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        assert processing_time < 30  # Should complete within 30 seconds
        assert analysis_node.analysis_results["total_strategies_analyzed"] == 1000
    
    # Test edge cases
    def test_empty_strategy_handling(self, analysis_node):
        """Test handling of empty strategy text."""
        strategy_data = {"agent_id": 0, "full_reasoning": ""}
        markers = analysis_node.analyze_transcript(strategy_data, 1)
        
        assert markers == {}
        assert analysis_node.analysis_results["total_strategies_analyzed"] == 0
    
    def test_short_strategy_edge_cases(self, analysis_node):
        """Test handling of very short strategies."""
        strategy_data = {"agent_id": 0, "full_reasoning": "Cooperate."}
        markers = analysis_node.analyze_transcript(strategy_data, 1)
        
        assert analysis_node.analysis_results["total_strategies_analyzed"] == 1
        # Should not crash on short text
    
    def test_overlapping_marker_detection(self, analysis_node):
        """Test detection when multiple markers overlap."""
        transcript = """
        We are identical agents, which from a superrational perspective means 
        what I would do is cooperate. Since we have the same model and identical reasoning,
        defecting against myself makes no sense.
        """
        
        strategy_data = {"agent_id": 0, "full_reasoning": transcript}
        markers = analysis_node.analyze_transcript(strategy_data, 1)
        
        # Should detect both identity and superrational markers
        assert markers["identity_reasoning"] >= 2
        assert markers["superrational_logic"] >= 2
    
    # Test confidence scores
    def test_confidence_score_calculation(self, analysis_node):
        """Test confidence score calculation with negation."""
        # Test without negation
        transcript_positive = "We are identical agents so we should cooperate."
        strategy_data = {"agent_id": 0, "full_reasoning": transcript_positive}
        analysis_node.analyze_transcript(strategy_data, 1)
        
        positive_confidence = analysis_node.analysis_results["marker_examples"]["identity_reasoning"][0]["confidence_score"]
        assert positive_confidence >= 0.9
        
        # Reset and test with negation
        analysis_node.analysis_results["marker_examples"]["identity_reasoning"] = []
        
        transcript_negative = "We are not identical agents in this case."
        strategy_data = {"agent_id": 1, "full_reasoning": transcript_negative}
        analysis_node.analyze_transcript(strategy_data, 1)
        
        if analysis_node.analysis_results["marker_examples"]["identity_reasoning"]:
            negative_confidence = analysis_node.analysis_results["marker_examples"]["identity_reasoning"][0]["confidence_score"]
            assert negative_confidence < positive_confidence
    
    # Test metadata generation
    def test_metadata_generation(self, analysis_node, mock_data_manager):
        """Test that metadata is properly generated."""
        # Add some errors and skipped files
        analysis_node.errors.append("Test error 1")
        analysis_node.skipped_files.append("test_file.json")
        
        report = analysis_node.generate_analysis_report()
        metadata = report["acausal_analysis"]["metadata"]
        
        assert metadata["analysis_version"] == "1.0"
        assert metadata["configuration"]["context_window"] == 3
        assert len(metadata["errors"]) == 1
        assert len(metadata["skipped_files"]) == 1
        assert "processing_stats" in metadata
    
    # Test qualitative summary
    def test_qualitative_summary_generation(self, analysis_node):
        """Test generation of qualitative summary text."""
        # Add some sample data
        for i in range(10):
            transcript = "We are identical agents with superrational reasoning."
            strategy_data = {"agent_id": i, "full_reasoning": transcript}
            analysis_node.analyze_transcript(strategy_data, 1)
        
        report = analysis_node.generate_analysis_report()
        summary = report["acausal_analysis"]["qualitative_summary"]
        
        assert isinstance(summary, str)
        assert len(summary) > 50  # Should be meaningful text
        assert "identity reasoning" in summary.lower()
        assert "superrational" in summary.lower()

    # Cross-model analysis integration tests
    @pytest.mark.asyncio
    async def test_cross_model_analysis_integration(self, analysis_node, mock_data_manager):
        """Test full integration with cross-model analysis."""
        # Create multi-model strategy files
        strategies_r1 = {
            "round": 1,
            "timestamp": datetime.now().isoformat(),
            "strategies": [
                {
                    "agent_id": 0,
                    "round": 1,
                    "model": "gpt-4",
                    "full_reasoning": "As an identical agent, cooperation is the rational choice.",
                    "strategy": "COOPERATE"
                },
                {
                    "agent_id": 1,
                    "round": 1,
                    "model": "gpt-4",
                    "full_reasoning": "I should cooperate with my identical counterpart.",
                    "strategy": "COOPERATE"
                },
                {
                    "agent_id": 2,
                    "round": 1,
                    "model": "claude-3",
                    "full_reasoning": "From constitutional principles, cooperation minimizes harm.",
                    "strategy": "COOPERATE"
                },
                {
                    "agent_id": 3,
                    "round": 1,
                    "model": "claude-3",
                    "full_reasoning": "Ethical considerations suggest cooperation.",
                    "strategy": "COOPERATE"
                }
            ]
        }
        
        # Create game result files
        games_r1 = {
            "round": 1,
            "timestamp": datetime.now().isoformat(),
            "games": [
                {
                    "game_id": "g1",
                    "round": 1,
                    "player1_id": 0,
                    "player2_id": 1,
                    "player1_action": "COOPERATE",
                    "player2_action": "COOPERATE"
                },
                {
                    "game_id": "g2",
                    "round": 1,
                    "player1_id": 2,
                    "player2_id": 3,
                    "player1_action": "COOPERATE",
                    "player2_action": "COOPERATE"
                },
                {
                    "game_id": "g3",
                    "round": 1,
                    "player1_id": 0,
                    "player2_id": 2,
                    "player1_action": "COOPERATE",
                    "player2_action": "DEFECT"
                }
            ]
        }
        
        # Create test files
        rounds_path = mock_data_manager.experiment_path / "rounds"
        rounds_path.mkdir(parents=True, exist_ok=True)
        
        with open(rounds_path / "strategies_r1.json", "w") as f:
            json.dump(strategies_r1, f)
        
        with open(rounds_path / "games_r1.json", "w") as f:
            json.dump(games_r1, f)
        
        # Run analysis
        context = {ContextKeys.DATA_MANAGER: mock_data_manager}
        result = await analysis_node.execute(context)
        
        assert "transcript_analysis" in result
        report = result["transcript_analysis"]
        
        # Check cross-model analysis exists
        assert "cross_model_analysis" in report
        cross_analysis = report["cross_model_analysis"]
        
        assert "cooperation_matrix" in cross_analysis
        assert "in_group_bias" in cross_analysis
        assert "model_statistics" in cross_analysis
        assert "model_coalitions" in cross_analysis
        assert "visualization_data" in cross_analysis
    
    @pytest.mark.asyncio
    async def test_single_model_experiment_skip_cross_analysis(self, analysis_node, mock_data_manager):
        """Test that single-model experiments skip cross-model analysis."""
        # Create single-model strategy files
        strategies_r1 = {
            "round": 1,
            "timestamp": datetime.now().isoformat(),
            "strategies": [
                {
                    "agent_id": 0,
                    "round": 1,
                    "model": "gpt-4",
                    "full_reasoning": "Cooperation is rational.",
                    "strategy": "COOPERATE"
                },
                {
                    "agent_id": 1,
                    "round": 1,
                    "model": "gpt-4",
                    "full_reasoning": "I cooperate.",
                    "strategy": "COOPERATE"
                }
            ]
        }
        
        # Create test files
        rounds_path = mock_data_manager.experiment_path / "rounds"
        rounds_path.mkdir(parents=True, exist_ok=True)
        
        with open(rounds_path / "strategies_r1.json", "w") as f:
            json.dump(strategies_r1, f)
        
        # Run analysis
        context = {ContextKeys.DATA_MANAGER: mock_data_manager}
        result = await analysis_node.execute(context)
        
        assert "transcript_analysis" in result
        report = result["transcript_analysis"]
        
        # Cross-model analysis should not exist
        assert "cross_model_analysis" not in report
    
    def test_has_multiple_models(self, analysis_node):
        """Test model detection logic."""
        # Single model case
        analysis_node.analysis_results["strategies_by_model"]["gpt-4"] = 10
        assert not analysis_node._has_multiple_models()
        
        # Multiple models case
        analysis_node.analysis_results["strategies_by_model"]["claude-3"] = 5
        assert analysis_node._has_multiple_models()
        
        # Should ignore "unknown" model
        analysis_node.analysis_results["strategies_by_model"] = {"unknown": 10}
        assert not analysis_node._has_multiple_models()
    
    @pytest.mark.asyncio
    async def test_cross_model_analysis_error_handling(self, analysis_node, mock_data_manager):
        """Test error handling in cross-model analysis."""
        # Create strategy files with missing data
        strategies_r1 = {
            "round": 1,
            "strategies": [
                {"agent_id": 0, "model": "gpt-4", "full_reasoning": "Test"},
                {"agent_id": 1, "model": "claude-3", "full_reasoning": "Test"}
            ]
        }
        
        # Create invalid game files  
        games_r1 = {
            "round": 1,
            "games": [
                {"player1_id": 0}  # Missing required fields
            ]
        }
        
        rounds_path = mock_data_manager.experiment_path / "rounds"
        rounds_path.mkdir(parents=True, exist_ok=True)
        
        with open(rounds_path / "strategies_r1.json", "w") as f:
            json.dump(strategies_r1, f)
        
        with open(rounds_path / "games_r1.json", "w") as f:
            json.dump(games_r1, f)
        
        # Run analysis - should handle errors gracefully
        context = {ContextKeys.DATA_MANAGER: mock_data_manager}
        result = await analysis_node.execute(context)
        
        assert "transcript_analysis" in result
        report = result["transcript_analysis"]
        
        # Should still have cross-model analysis section
        if "cross_model_analysis" in report:
            cross_analysis = report["cross_model_analysis"]
            # May have empty or error state
            assert "cooperation_matrix" in cross_analysis
    
    def test_model_statistics_calculation(self, analysis_node):
        """Test calculation of model-level statistics."""
        from src.utils.cross_model_analyzer import CooperationStats
        
        # Mock cooperation stats
        cooperation_stats = {
            "gpt-4": {
                "gpt-4": CooperationStats(
                    cooperation_count=10,
                    total_games=10,
                    cooperation_rate=1.0,
                    confidence_interval=(0.8, 1.0),
                    sample_size=10
                ),
                "claude-3": CooperationStats(
                    cooperation_count=5,
                    total_games=10,
                    cooperation_rate=0.5,
                    confidence_interval=(0.3, 0.7),
                    sample_size=10
                )
            }
        }
        
        analyzer = Mock()
        model_stats = analysis_node._calculate_model_statistics(analyzer, cooperation_stats)
        
        assert "gpt-4" in model_stats
        assert model_stats["gpt-4"]["avg_cooperation"] == 0.75  # (10+5)/20
        assert model_stats["gpt-4"]["total_games"] == 20
    
    def test_find_extreme_pairs(self, analysis_node):
        """Test finding strongest and weakest cooperation pairs."""
        from src.utils.cross_model_analyzer import CooperationStats
        
        cooperation_stats = {
            "gpt-4": {
                "gpt-4": CooperationStats(10, 10, 1.0, (0.8, 1.0), 10),
                "claude-3": CooperationStats(2, 10, 0.2, (0.1, 0.3), 10)
            },
            "claude-3": {
                "claude-3": CooperationStats(5, 10, 0.5, (0.3, 0.7), 10),
                "gpt-4": CooperationStats(1, 10, 0.1, (0.0, 0.2), 10)
            }
        }
        
        strongest, weakest = analysis_node._find_extreme_pairs(cooperation_stats)
        
        assert strongest == ["gpt-4", "gpt-4"]
        assert weakest == ["claude-3", "gpt-4"]
    
    def test_diversity_impact_calculation(self, analysis_node):
        """Test model diversity impact calculation."""
        # Significant positive bias
        in_group_bias = {
            "bias": 0.3,
            "p_value": 0.01
        }
        
        impact = analysis_node._calculate_diversity_impact({}, in_group_bias)
        assert impact == -0.3  # Negative impact due to diversity
        
        # Non-significant bias
        in_group_bias = {
            "bias": 0.3,
            "p_value": 0.1
        }
        
        impact = analysis_node._calculate_diversity_impact({}, in_group_bias)
        assert impact == -0.15  # Reduced impact due to non-significance
        
        # No bias
        in_group_bias = {
            "bias": None,
            "p_value": None
        }
        
        impact = analysis_node._calculate_diversity_impact({}, in_group_bias)
        assert impact == 0.0
    
    @pytest.mark.asyncio
    async def test_backward_compatibility(self, analysis_node, mock_data_manager):
        """Test backward compatibility with experiments missing model info."""
        # Create strategy files without model field
        strategies_r1 = {
            "round": 1,
            "strategies": [
                {
                    "agent_id": 0,
                    "round": 1,
                    "full_reasoning": "Cooperation is best.",
                    "strategy": "COOPERATE"
                }
            ]
        }
        
        rounds_path = mock_data_manager.experiment_path / "rounds"
        rounds_path.mkdir(parents=True, exist_ok=True)
        
        with open(rounds_path / "strategies_r1.json", "w") as f:
            json.dump(strategies_r1, f)
        
        # Run analysis
        context = {ContextKeys.DATA_MANAGER: mock_data_manager}
        result = await analysis_node.execute(context)
        
        # Should complete without errors
        assert "transcript_analysis" in result
        report = result["transcript_analysis"]
        
        # Model should be tracked as "unknown"
        assert "unknown" in report["acausal_analysis"]["model_specific_analysis"]["strategies_by_model"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])