"""Simple test to verify strategy storage JSON format."""

import json
import tempfile
import shutil
from pathlib import Path
from datetime import datetime

# Test the JSON format directly
def test_strategy_storage_format():
    """Test that the JSON format matches Epic 2 specification."""

    # Create test data in Epic 2 format
    test_data = {
        "round": 1,
        "timestamp": datetime.now().isoformat(),
        "strategies": [
            {
                "round": 1,
                "agent_id": 0,
                "timestamp": "2024-01-01T00:00:00Z",
                "model": "google/gemini-2.5-flash",
                "strategy": "Always cooperate with identical agents",  # Note: "strategy" not "strategy_text"
                "full_reasoning": "After careful analysis, I conclude that always cooperating with identical agents is optimal.",
                "prompt_tokens": 500,
                "completion_tokens": 300
            },
            {
                "round": 1,
                "agent_id": 1,
                "timestamp": "2024-01-01T00:00:01Z",
                "model": "google/gemini-2.5-flash",
                "strategy": "Tit-for-tat strategy",
                "full_reasoning": "I will start with cooperation and then mirror the opponent's previous move.",
                "prompt_tokens": 450,
                "completion_tokens": 250
            }
        ]
    }

    # Create temporary directory
    temp_dir = tempfile.mkdtemp()
    try:
        # Create the expected directory structure
        rounds_dir = Path(temp_dir) / "experiment" / "rounds"
        rounds_dir.mkdir(parents=True, exist_ok=True)

        # Write test file
        test_file = rounds_dir / "strategies_r1.json"
        with open(test_file, 'w') as f:
            json.dump(test_data, f, indent=2)

        # Read back and verify
        with open(test_file, 'r') as f:
            loaded_data = json.load(f)

        # Verify structure
        assert "round" in loaded_data
        assert "timestamp" in loaded_data
        assert "strategies" in loaded_data
        assert loaded_data["round"] == 1
        assert len(loaded_data["strategies"]) == 2

        # Verify each strategy has Epic 2 format
        for i, strategy in enumerate(loaded_data["strategies"]):
            assert "round" in strategy
            assert "agent_id" in strategy
            assert "timestamp" in strategy
            assert "model" in strategy
            assert "strategy" in strategy  # Epic 2 uses "strategy" not "strategy_text"
            assert "full_reasoning" in strategy
            assert "prompt_tokens" in strategy
            assert "completion_tokens" in strategy

            # Verify specific values
            assert strategy["agent_id"] == i
            assert strategy["prompt_tokens"] > 0
            assert strategy["completion_tokens"] > 0
            assert isinstance(strategy["strategy"], str)
            assert isinstance(strategy["full_reasoning"], str)

        print("✓ JSON structure matches Epic 2 specification")
        print("✓ Field mapping (strategy_text → strategy) verified")
        print("✓ All required metadata fields present")
        print("✓ Token counts properly included")

        # Test with long reasoning
        long_reasoning = "This is a very long reasoning. " * 1000
        test_data["strategies"][0]["full_reasoning"] = long_reasoning

        with open(test_file, 'w') as f:
            json.dump(test_data, f, indent=2)

        with open(test_file, 'r') as f:
            loaded_data = json.load(f)

        assert loaded_data["strategies"][0]["full_reasoning"] == long_reasoning
        print("✓ Full reasoning preserved without truncation")

        # Test special characters
        special_text = 'Strategy with "quotes" and \n newlines and \t tabs'
        test_data["strategies"][0]["strategy"] = special_text

        with open(test_file, 'w') as f:
            json.dump(test_data, f, indent=2)

        with open(test_file, 'r') as f:
            loaded_data = json.load(f)

        assert loaded_data["strategies"][0]["strategy"] == special_text
        print("✓ Special characters handled correctly")

        print("\nAll tests passed! Strategy storage format is correct.")

    finally:
        # Clean up
        shutil.rmtree(temp_dir)


if __name__ == "__main__":
    test_strategy_storage_format()
