#!/usr/bin/env python3
"""Test to verify cooperation rate formatting fix."""

from src.core.models import RoundSummary
from src.core.prompts import format_previous_rounds

def test_cooperation_rate_formatting():
    """Test that cooperation rates are formatted correctly."""
    
    # Create test data with various cooperation rates
    test_summaries = [
        RoundSummary(
            round=1,
            cooperation_rate=1.0,  # 100%
            average_score=27.0,
            score_variance=0.0,
            power_distribution={"min": 100.0, "max": 100.0, "mean": 100.0, "std": 0.0},
            score_distribution={"min": 27.0, "max": 27.0, "avg": 27.0},
            anonymized_games=[]
        ),
        RoundSummary(
            round=2,
            cooperation_rate=0.5,  # 50%
            average_score=20.0,
            score_variance=5.0,
            power_distribution={"min": 95.0, "max": 105.0, "mean": 100.0, "std": 3.0},
            score_distribution={"min": 15.0, "max": 25.0, "avg": 20.0},
            anonymized_games=[]
        ),
        RoundSummary(
            round=3,
            cooperation_rate=0.01,  # 1%
            average_score=10.0,
            score_variance=2.0,
            power_distribution={"min": 90.0, "max": 110.0, "mean": 100.0, "std": 5.0},
            score_distribution={"min": 8.0, "max": 12.0, "avg": 10.0},
            anonymized_games=[]
        )
    ]
    
    # Format the rounds
    formatted = format_previous_rounds(test_summaries)
    
    print("Formatted output:")
    print(formatted)
    print("\n" + "="*50 + "\n")
    
    # Verify correct formatting
    assert "Average cooperation rate: 100.0%" in formatted, "1.0 should format as 100.0%"
    assert "Average cooperation rate: 50.0%" in formatted, "0.5 should format as 50.0%"
    assert "Average cooperation rate: 1.0%" in formatted, "0.01 should format as 1.0%"
    
    # Make sure the old incorrect format is not present
    assert "Average cooperation rate: 1.0%" in formatted and "Round 1:" in formatted, "Should not show 1.0% for Round 1"
    assert "Average cooperation rate: 0.5%" not in formatted, "Should not show 0.5% anywhere"
    assert "Average cooperation rate: 0.0%" not in formatted, "Should not show 0.0% anywhere"
    
    print("✅ All formatting tests passed!")
    print("\nCorrect formatting examples:")
    print("- cooperation_rate=1.0 → 100.0%")
    print("- cooperation_rate=0.5 → 50.0%")
    print("- cooperation_rate=0.01 → 1.0%")

if __name__ == "__main__":
    test_cooperation_rate_formatting()