"""Tests for model-specific pattern detection in AnalysisNode."""

import pytest
import json
from pathlib import Path
from unittest.mock import Mock, MagicMock
from src.nodes.analysis import AnalysisNode
from src.nodes.base import ContextKeys
from src.utils.data_manager import DataManager


class TestModelSpecificPatternDetection:
    """Test suite for model-specific pattern detection."""
    
    @pytest.fixture
    def analysis_node(self):
        """Create an AnalysisNode instance."""
        return AnalysisNode()
    
    @pytest.fixture
    def mock_data_manager(self, tmp_path):
        """Create a mock DataManager."""
        mock_dm = Mock(spec=DataManager)
        mock_dm.experiment_path = tmp_path
        mock_dm._write_json = Mock()
        return mock_dm
    
    @pytest.fixture
    def gpt4_strategy(self):
        """Sample GPT-4 strategy with characteristic patterns."""
        return {
            "agent_id": 1,
            "model": "gpt-4",
            "reasoning": """Let me think through this step by step. First, I'll analyze the chain of thought 
            for this decision. The explicit utility calculation shows that cooperation yields 3 points 
            while defection yields 0 or 5. Computing the expected value mathematically, I can quantify 
            the payoff calculation. Breaking this down numerically, the optimal strategy depends on 
            what I expect my opponent to do.""",
            "decision": "COOPERATE"
        }
    
    @pytest.fixture  
    def claude_strategy(self):
        """Sample Claude strategy with characteristic patterns."""
        return {
            "agent_id": 2,
            "model": "claude-3-opus",
            "reasoning": """I need to consider the constitutional principles at play here. My ethical 
            framework prioritizes minimizing harm while being helpful, harmless, and honest. The moral 
            framework suggests cooperation aligns with human values. From a safety consideration 
            perspective, responsible AI behavior means choosing beneficence over exploitation.""",
            "decision": "COOPERATE"
        }
    
    @pytest.fixture
    def gemini_strategy(self):
        """Sample Gemini strategy with characteristic patterns."""
        return {
            "agent_id": 3,
            "model": "gemini-pro",
            "reasoning": """I will analyze this systematically using a logical framework. Therefore, 
            the structured approach suggests examining each option methodically. Thus, by organizing 
            this in a categorical manner, we can see that cooperation is optimal. Consequently, my 
            step-wise analysis leads to a clear conclusion.""",
            "decision": "COOPERATE"
        }
    
    def test_gpt4_pattern_detection(self, analysis_node, gpt4_strategy):
        """Test detection of GPT-4 specific patterns."""
        markers = analysis_node.analyze_transcript(gpt4_strategy, round_num=1)
        
        # Check that GPT-4 patterns were detected
        assert "gpt4_patterns" in analysis_node.analysis_results["markers_by_model"]["gpt-4"]
        gpt4_count = analysis_node.analysis_results["markers_by_model"]["gpt-4"]["gpt4_patterns"]
        
        # Should detect multiple GPT-4 patterns
        assert gpt4_count >= 5  # chain of thought, step by step, utility calculation, etc.
        
        # Check examples were captured
        assert any(
            example["model"] == "gpt-4" and "chain of thought" in example["quote"].lower()
            for example in analysis_node.analysis_results["marker_examples"].get("gpt4_patterns", [])
        )
    
    def test_claude_pattern_detection(self, analysis_node, claude_strategy):
        """Test detection of Claude specific patterns."""
        markers = analysis_node.analyze_transcript(claude_strategy, round_num=1)
        
        # Check that Claude patterns were detected
        assert "claude_patterns" in analysis_node.analysis_results["markers_by_model"]["claude-3-opus"]
        claude_count = analysis_node.analysis_results["markers_by_model"]["claude-3-opus"]["claude_patterns"]
        
        # Should detect multiple Claude patterns
        assert claude_count >= 5  # constitutional, ethical, harm minimization, etc.
        
        # Check specific pattern examples
        assert any(
            example["model"] == "claude-3-opus" and "constitutional" in example["quote"].lower()
            for example in analysis_node.analysis_results["marker_examples"].get("claude_patterns", [])
        )
    
    def test_gemini_pattern_detection(self, analysis_node, gemini_strategy):
        """Test detection of Gemini specific patterns."""
        markers = analysis_node.analyze_transcript(gemini_strategy, round_num=1)
        
        # Check that Gemini patterns were detected
        assert "gemini_patterns" in analysis_node.analysis_results["markers_by_model"]["gemini-pro"]
        gemini_count = analysis_node.analysis_results["markers_by_model"]["gemini-pro"]["gemini_patterns"]
        
        # Should detect multiple Gemini patterns
        assert gemini_count >= 5  # analyze, systematic, logical, therefore, thus, etc.
        
        # Verify pattern tracking
        model_patterns = analysis_node.analysis_results["model_specific_patterns"]["gemini-pro"]
        assert any("gemini_patterns:" in pattern for pattern in model_patterns)
    
    def test_model_specific_patterns_not_cross_detected(self, analysis_node, gpt4_strategy, claude_strategy):
        """Test that model-specific patterns aren't detected for wrong models."""
        # Analyze GPT-4 strategy
        analysis_node.analyze_transcript(gpt4_strategy, round_num=1)
        
        # GPT-4 shouldn't have Claude patterns
        assert "claude_patterns" not in analysis_node.analysis_results["markers_by_model"]["gpt-4"]
        assert "gemini_patterns" not in analysis_node.analysis_results["markers_by_model"]["gpt-4"]
        
        # Analyze Claude strategy
        analysis_node.analyze_transcript(claude_strategy, round_num=1)
        
        # Claude shouldn't have GPT-4 patterns
        assert "gpt4_patterns" not in analysis_node.analysis_results["markers_by_model"]["claude-3-opus"]
        assert "gemini_patterns" not in analysis_node.analysis_results["markers_by_model"]["claude-3-opus"]
    
    def test_standard_markers_still_detected(self, analysis_node):
        """Test that standard acausal markers are still detected alongside model-specific ones."""
        strategy = {
            "agent_id": 1,
            "model": "gpt-4",
            "reasoning": """Let me think step by step about this. Since we are identical agents with 
            the same model and same prompt, it's rational to cooperate with myself. The chain of thought 
            leads me to calculate the utility of mutual cooperation.""",
            "decision": "COOPERATE"
        }
        
        markers = analysis_node.analyze_transcript(strategy, round_num=1)
        
        # Should detect both standard and model-specific markers
        assert analysis_node.analysis_results["identity_reasoning_count"] > 0
        assert analysis_node.analysis_results["superrational_logic_count"] > 0
        assert "gpt4_patterns" in analysis_node.analysis_results["markers_by_model"]["gpt-4"]
    
    def test_model_insights_generation(self, analysis_node, gpt4_strategy, claude_strategy, gemini_strategy):
        """Test generation of model-specific insights."""
        # Analyze strategies from different models
        analysis_node.analyze_transcript(gpt4_strategy, round_num=1)
        analysis_node.analyze_transcript(claude_strategy, round_num=1)
        analysis_node.analyze_transcript(gemini_strategy, round_num=1)
        
        # Generate insights
        insights = analysis_node._generate_model_insights()
        
        # Check GPT-4 insights
        assert "gpt-4" in insights
        assert insights["gpt-4"]["behavioral_notes"] != ""
        assert "chain of thought" in insights["gpt-4"]["behavioral_notes"]
        
        # Check Claude insights
        assert "claude-3-opus" in insights
        assert "constitutional" in insights["claude-3-opus"]["behavioral_notes"]
        
        # Check Gemini insights
        assert "gemini-pro" in insights
        assert "analytical" in insights["gemini-pro"]["behavioral_notes"]
    
    def test_mixed_model_pattern_tracking(self, analysis_node):
        """Test tracking patterns across multiple models."""
        strategies = [
            {
                "agent_id": 1,
                "model": "gpt-4",
                "reasoning": "Let me calculate the utility step by step.",
                "decision": "COOPERATE"
            },
            {
                "agent_id": 2,
                "model": "gpt-4", 
                "reasoning": "Breaking this down mathematically shows cooperation is optimal.",
                "decision": "COOPERATE"
            },
            {
                "agent_id": 3,
                "model": "claude-3",
                "reasoning": "My ethical principles and constitutional framework suggest cooperation.",
                "decision": "COOPERATE"
            },
            {
                "agent_id": 4,
                "model": "gemini-pro",
                "reasoning": "Therefore, analyzing this systematically, I conclude cooperation is best.",
                "decision": "COOPERATE"
            }
        ]
        
        for strategy in strategies:
            analysis_node.analyze_transcript(strategy, round_num=1)
        
        # Check model distribution
        assert analysis_node.analysis_results["strategies_by_model"]["gpt-4"] == 2
        assert analysis_node.analysis_results["strategies_by_model"]["claude-3"] == 1
        assert analysis_node.analysis_results["strategies_by_model"]["gemini-pro"] == 1
        
        # Check pattern distribution
        gpt4_markers = analysis_node.analysis_results["markers_by_model"]["gpt-4"]
        assert "gpt4_patterns" in gpt4_markers
        assert gpt4_markers["gpt4_patterns"] >= 2
        
        claude_markers = analysis_node.analysis_results["markers_by_model"]["claude-3"]
        assert "claude_patterns" in claude_markers
        
        gemini_markers = analysis_node.analysis_results["markers_by_model"]["gemini-pro"]
        assert "gemini_patterns" in gemini_markers
    
    def test_pattern_detection_case_insensitive(self, analysis_node):
        """Test that pattern detection is case insensitive."""
        strategy = {
            "agent_id": 1,
            "model": "GPT-4",  # Uppercase model name
            "reasoning": "LET ME THINK about this. The CHAIN OF THOUGHT suggests EXPLICIT UTILITY CALCULATION.",
            "decision": "COOPERATE"
        }
        
        analysis_node.analyze_transcript(strategy, round_num=1)
        
        # Should still detect patterns despite case differences
        assert "gpt4_patterns" in analysis_node.analysis_results["markers_by_model"]["GPT-4"]
        assert analysis_node.analysis_results["markers_by_model"]["GPT-4"]["gpt4_patterns"] >= 2
    
    def test_unknown_model_handling(self, analysis_node):
        """Test handling of unknown model types."""
        strategy = {
            "agent_id": 1,
            "model": "unknown-model-xyz",
            "reasoning": "I need to think about this decision carefully.",
            "decision": "COOPERATE"
        }
        
        analysis_node.analyze_transcript(strategy, round_num=1)
        
        # Should track the strategy but not apply model-specific patterns
        assert analysis_node.analysis_results["strategies_by_model"]["unknown-model-xyz"] == 1
        assert "gpt4_patterns" not in analysis_node.analysis_results["markers_by_model"]["unknown-model-xyz"]
        assert "claude_patterns" not in analysis_node.analysis_results["markers_by_model"]["unknown-model-xyz"]
        assert "gemini_patterns" not in analysis_node.analysis_results["markers_by_model"]["unknown-model-xyz"]