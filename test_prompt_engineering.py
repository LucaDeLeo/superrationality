"""Unit tests for prompt engineering components."""

import pytest
from unittest.mock import MagicMock

from src.core.prompts import (
    PromptTemplate, STRATEGY_COLLECTION_PROMPT, 
    format_distribution, format_round_summary,
    apply_model_variations, validate_prompt_compatibility,
    MODEL_PROMPT_VARIATIONS
)
from src.core.models import RoundSummary, AnonymizedGameResult


class TestPromptTemplate:
    """Test the PromptTemplate class."""
    
    def test_prompt_template_renders_correctly(self):
        """Test that prompt template renders with provided context."""
        template = PromptTemplate(
            template="Hello {name}, you have {count} messages."
        )
        
        context = {"name": "Alice", "count": 5}
        result = template.render(context)
        
        assert result == "Hello Alice, you have 5 messages."
    
    def test_prompt_template_extracts_required_variables(self):
        """Test that required variables are extracted from template."""
        template = PromptTemplate(
            template="User {user_id} has {score} points in round {round}."
        )
        
        assert template.required_variables == {"user_id", "score", "round"}
    
    def test_prompt_template_validates_missing_variables(self):
        """Test that validation fails for missing variables."""
        template = PromptTemplate(
            template="Hello {name}, you have {count} messages."
        )
        
        # Missing 'count' variable
        context = {"name": "Alice"}
        
        with pytest.raises(ValueError, match="Missing required template variables: \\['count'\\]"):
            template.render(context)
    
    def test_prompt_template_handles_explicit_required_variables(self):
        """Test template with explicitly set required variables."""
        template = PromptTemplate(
            template="Simple template",
            required_variables={"var1", "var2"}
        )
        
        assert template.required_variables == {"var1", "var2"}
        
        # Should fail even though template doesn't use these vars
        with pytest.raises(ValueError):
            template.render({})


class TestStrategyCollectionPrompt:
    """Test the main strategy collection prompt."""
    
    def test_critical_insight_always_included(self):
        """Test that critical insight text is always included in the prompt."""
        context = {
            "coop_rate": 75.0,
            "distribution": "min: 10.0, max: 50.0, avg: 30.0"
        }
        
        result = STRATEGY_COLLECTION_PROMPT.render(context)
        
        # Check for critical insight
        assert "CRITICAL INSIGHT:" in result
        assert "identical copies of the same AI model" in result
        assert "functionally the same agent" in result
    
    def test_prompt_includes_cooperation_rate(self):
        """Test that cooperation rate is included correctly."""
        context = {
            "coop_rate": 82.5,
            "distribution": "min: 0.0, max: 100.0, avg: 50.0"
        }
        
        result = STRATEGY_COLLECTION_PROMPT.render(context)
        
        assert "82.5%" in result
        assert "Average cooperation rate: 82.5%" in result
    
    def test_prompt_includes_score_distribution(self):
        """Test that score distribution is included correctly."""
        context = {
            "coop_rate": 50.0,
            "distribution": "min: 15.5, max: 85.3, avg: 45.2"
        }
        
        result = STRATEGY_COLLECTION_PROMPT.render(context)
        
        assert "min: 15.5, max: 85.3, avg: 45.2" in result
        assert "Score distribution: min: 15.5, max: 85.3, avg: 45.2" in result
    
    def test_prompt_matches_epic_specification(self):
        """Test that prompt output matches Epic 2 specification exactly."""
        context = {
            "coop_rate": 65.0,
            "distribution": "min: 20.0, max: 80.0, avg: 50.0"
        }
        
        result = STRATEGY_COLLECTION_PROMPT.render(context)
        
        # Verify key components from Epic 2
        assert "You are participating in an experimental study" in result
        assert "Previous rounds showed:" in result
        assert "Design a strategy for your subagent" in result
        assert "clear and implementable as a simple decision rule" in result


class TestRoundSummaryFormatting:
    """Test round summary formatting functions."""
    
    def test_format_distribution(self):
        """Test score distribution formatting."""
        result = format_distribution(10.5, 95.3, 52.7)
        assert result == "min: 10.5, max: 95.3, avg: 52.7"
        
        # Test with whole numbers
        result = format_distribution(0, 100, 50)
        assert result == "min: 0.0, max: 100.0, avg: 50.0"
    
    def test_format_round_summary_first_round(self):
        """Test formatting when no previous round data exists."""
        result = format_round_summary(None)
        
        assert result["coop_rate"] == 50.0
        assert result["distribution"] == "No previous data"
    
    def test_format_round_summary_with_data(self):
        """Test formatting with actual round summary data."""
        # Create a mock RoundSummary
        round_summary = MagicMock(spec=RoundSummary)
        round_summary.cooperation_rate = 73.5
        round_summary.average_score = 45.2
        round_summary.score_variance = 100.0  # std = 10
        round_summary.score_distribution = {
            'min': 25.2,
            'max': 65.2,
            'avg': 45.2
        }
        
        result = format_round_summary(round_summary)
        
        assert result["coop_rate"] == 73.5
        assert result["distribution"] == "min: 25.2, max: 65.2, avg: 45.2"
    
    def test_format_round_summary_without_score_distribution(self):
        """Test formatting when score_distribution is missing."""
        # Create a mock RoundSummary without score_distribution
        round_summary = MagicMock(spec=RoundSummary)
        round_summary.cooperation_rate = 60.0
        round_summary.average_score = 30.0
        round_summary.score_variance = 0.0
        round_summary.score_distribution = {}
        
        result = format_round_summary(round_summary)
        
        assert result["coop_rate"] == 60.0
        # Should use average score for min/max when distribution not available
        assert result["distribution"] == "min: 30.0, max: 30.0, avg: 30.0"
    
    def test_format_round_summary_edge_cases(self):
        """Test edge cases in round summary formatting."""
        # Test with extreme values
        round_summary = MagicMock(spec=RoundSummary)
        round_summary.cooperation_rate = 0.0
        round_summary.average_score = 0.0
        round_summary.score_variance = 0.0
        round_summary.score_distribution = {
            'min': 0.0,
            'max': 0.0,
            'avg': 0.0
        }
        
        result = format_round_summary(round_summary)
        
        assert result["coop_rate"] == 0.0
        assert result["distribution"] == "min: 0.0, max: 0.0, avg: 0.0"
        
        # Test with 100% cooperation
        round_summary.cooperation_rate = 100.0
        round_summary.score_distribution = {
            'min': 50.0,
            'max': 100.0,
            'avg': 75.0
        }
        
        result = format_round_summary(round_summary)
        
        assert result["coop_rate"] == 100.0
        assert result["distribution"] == "min: 50.0, max: 100.0, avg: 75.0"


class TestRoundSummaryCreation:
    """Test RoundSummary.from_game_results method."""
    
    def test_round_summary_from_game_results(self):
        """Test creating RoundSummary from game results."""
        from src.core.models import Agent, GameResult
        
        # Create test agents
        agents = [
            Agent(id=0, power=100.0, total_score=50.0),
            Agent(id=1, power=110.0, total_score=60.0),
            Agent(id=2, power=90.0, total_score=40.0)
        ]
        
        # Create test games
        games = [
            GameResult(
                game_id="g1", round=1,
                player1_id=0, player2_id=1,
                player1_action="COOPERATE", player2_action="COOPERATE",
                player1_power_before=100.0, player2_power_before=110.0
            ),
            GameResult(
                game_id="g2", round=1,
                player1_id=0, player2_id=2,
                player1_action="DEFECT", player2_action="COOPERATE",
                player1_power_before=100.0, player2_power_before=90.0
            ),
            GameResult(
                game_id="g3", round=1,
                player1_id=1, player2_id=2,
                player1_action="COOPERATE", player2_action="DEFECT",
                player1_power_before=110.0, player2_power_before=90.0
            )
        ]
        
        # Create round summary
        summary = RoundSummary.from_game_results(1, games, agents)
        
        # Check cooperation rate: 4 COOPERATE out of 6 actions = 66.67%
        assert summary.cooperation_rate == pytest.approx(66.67, rel=0.01)
        
        # Check average score
        assert summary.average_score == 50.0  # (50 + 60 + 40) / 3
        
        # Check score distribution
        assert summary.score_distribution['min'] == 40.0
        assert summary.score_distribution['max'] == 60.0
        assert summary.score_distribution['avg'] == 50.0
        
        # Check power distribution
        assert summary.power_distribution['mean'] == 100.0  # (100 + 110 + 90) / 3
        assert summary.power_distribution['min'] == 90.0
        assert summary.power_distribution['max'] == 110.0
        
        # Check anonymized games
        assert len(summary.anonymized_games) == 3
        for anon_game in summary.anonymized_games:
            assert anon_game.round == 1
            assert anon_game.anonymous_id1.startswith("Agent_")
            assert anon_game.anonymous_id2.startswith("Agent_")


class TestModelAwarePrompting:
    """Test model-aware prompt rendering functionality."""
    
    def test_format_round_summary_with_model_type(self):
        """Test that format_round_summary includes model_type in context."""
        # Test with first round (no previous data)
        result = format_round_summary(None, None, "openai/gpt-4")
        assert result["model_type"] == "openai/gpt-4"
        assert result["coop_rate"] == 50.0
        
        # Test with round data
        round_summary = MagicMock(spec=RoundSummary)
        round_summary.cooperation_rate = 75.0
        round_summary.average_score = 40.0
        round_summary.score_distribution = {'min': 20.0, 'max': 60.0, 'avg': 40.0}
        
        result = format_round_summary(round_summary, None, "anthropic/claude-3-sonnet-20240229")
        assert result["model_type"] == "anthropic/claude-3-sonnet-20240229"
        assert result["coop_rate"] == 75.0
    
    def test_apply_model_variations(self):
        """Test applying model-specific prompt variations."""
        base_prompt = "Design a strategy for the game."
        
        # Test GPT-4 variations
        gpt4_prompt = apply_model_variations(base_prompt, "openai/gpt-4")
        assert "Provide a structured strategy with clear reasoning" in gpt4_prompt
        assert "Present your strategy as a concise decision rule" in gpt4_prompt
        
        # Test Claude variations
        claude_prompt = apply_model_variations(base_prompt, "anthropic/claude-3-sonnet-20240229")
        assert "Consider the ethical implications" in claude_prompt
        assert "Explain your strategy and its rationale" in claude_prompt
        
        # Test Gemini Pro variations
        gemini_prompt = apply_model_variations(base_prompt, "google/gemini-pro")
        assert "Analyze the strategic implications systematically" in gemini_prompt
        assert "Describe your strategy with logical reasoning" in gemini_prompt
        
        # Test unknown model (should return unchanged)
        unknown_prompt = apply_model_variations(base_prompt, "unknown/model")
        assert unknown_prompt == base_prompt
        
        # Test None model (should return unchanged)
        none_prompt = apply_model_variations(base_prompt, None)
        assert none_prompt == base_prompt
    
    def test_validate_prompt_compatibility(self):
        """Test prompt compatibility validation across models."""
        # Test with the actual STRATEGY_COLLECTION_PROMPT
        validation_results = validate_prompt_compatibility(STRATEGY_COLLECTION_PROMPT)
        
        # All supported models should validate successfully
        for model_type in MODEL_PROMPT_VARIATIONS.keys():
            assert model_type in validation_results
            assert validation_results[model_type] is True
    
    def test_validate_prompt_compatibility_with_custom_context(self):
        """Test prompt validation with custom context."""
        custom_context = {
            'coop_rate': 90.0,
            'distribution': 'min: 5.0, max: 95.0, avg: 50.0',
            'previous_rounds_detail': 'Round 1: High cooperation observed.'
        }
        
        validation_results = validate_prompt_compatibility(
            STRATEGY_COLLECTION_PROMPT, 
            custom_context
        )
        
        # All models should validate successfully
        assert all(validation_results.values())
    
    def test_model_specific_prompt_integration(self):
        """Test full integration of model-aware prompting."""
        # Create test context
        round_summary = MagicMock(spec=RoundSummary)
        round_summary.cooperation_rate = 80.0
        round_summary.average_score = 55.0
        round_summary.score_distribution = {'min': 30.0, 'max': 80.0, 'avg': 55.0}
        
        # Test rendering for each model type
        for model_type in MODEL_PROMPT_VARIATIONS.keys():
            context = format_round_summary(round_summary, None, model_type)
            base_prompt = STRATEGY_COLLECTION_PROMPT.render(context)
            final_prompt = apply_model_variations(base_prompt, model_type)
            
            # Verify critical components are present
            assert "CRITICAL INSIGHT:" in final_prompt
            assert "80.0%" in final_prompt  # cooperation rate
            assert "min: 30.0, max: 80.0, avg: 55.0" in final_prompt
            
            # Verify model-specific enhancements
            variations = MODEL_PROMPT_VARIATIONS[model_type]
            if "instruction_suffix" in variations:
                suffix_text = variations["instruction_suffix"].strip()
                assert suffix_text in final_prompt
            if "format_hint" in variations:
                assert variations["format_hint"] in final_prompt
    
    def test_prompt_length_constraints(self):
        """Test that prompts stay within reasonable length limits."""
        # Create a context with very long previous rounds detail
        long_detail = "Round details " * 500  # Very long string
        
        context = {
            'coop_rate': 75.0,
            'distribution': 'min: 0.0, max: 100.0, avg: 50.0',
            'previous_rounds_detail': long_detail
        }
        
        # Render for each model
        for model_type in MODEL_PROMPT_VARIATIONS.keys():
            context['model_type'] = model_type
            prompt = STRATEGY_COLLECTION_PROMPT.render(context)
            enhanced = apply_model_variations(prompt, model_type)
            
            # Verify prompt is not excessively long
            assert len(enhanced) < 10000  # Reasonable upper limit


if __name__ == "__main__":
    pytest.main([__file__, "-v"])