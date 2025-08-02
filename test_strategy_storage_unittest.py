"""Unit tests for strategy storage functionality using unittest."""

import unittest
import json
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from core.models import StrategyRecord
from utils.data_manager import DataManager


class TestStrategyStorage(unittest.TestCase):
    """Test strategy storage functionality."""

    def setUp(self):
        """Create a temporary directory for testing."""
        self.temp_dir = tempfile.mkdtemp()
        self.data_manager = DataManager(base_path=self.temp_dir)

        # Create sample strategies
        self.sample_strategies = [
            StrategyRecord(
                strategy_id="strat_0_r1_abc123",
                agent_id=0,
                round=1,
                strategy_text="Always cooperate with identical agents",
                full_reasoning="After careful analysis, I conclude that always cooperating with identical agents is optimal.",
                prompt_tokens=500,
                completion_tokens=300,
                model="google/gemini-2.5-flash",
                timestamp="2024-01-01T00:00:00Z"
            ),
            StrategyRecord(
                strategy_id="strat_1_r1_def456",
                agent_id=1,
                round=1,
                strategy_text="Tit-for-tat strategy",
                full_reasoning="I will start with cooperation and then mirror the opponent's previous move.",
                prompt_tokens=450,
                completion_tokens=250,
                model="google/gemini-2.5-flash",
                timestamp="2024-01-01T00:00:01Z"
            )
        ]

    def tearDown(self):
        """Remove temporary directory."""
        shutil.rmtree(self.temp_dir)

    def test_save_strategies_creates_correct_file_path(self):
        """Test that save_strategies creates the correct file path format."""
        round_num = 1
        self.data_manager.save_strategies(round_num, self.sample_strategies)

        expected_path = self.data_manager.experiment_path / "rounds" / "strategies_r1.json"
        self.assertTrue(expected_path.exists())
        self.assertTrue(expected_path.is_file())

    def test_json_structure_matches_epic_specification(self):
        """Test that saved JSON structure matches Epic 2 specification."""
        round_num = 1
        self.data_manager.save_strategies(round_num, self.sample_strategies)

        # Read the saved file
        file_path = self.data_manager.experiment_path / "rounds" / "strategies_r1.json"
        with open(file_path, 'r') as f:
            data = json.load(f)

        # Check top-level structure
        self.assertIn("round", data)
        self.assertIn("timestamp", data)
        self.assertIn("strategies", data)
        self.assertEqual(data["round"], round_num)
        self.assertIsInstance(data["strategies"], list)
        self.assertEqual(len(data["strategies"]), 2)

        # Check each strategy follows Epic 2 format
        for i, strategy in enumerate(data["strategies"]):
            self.assertIn("round", strategy)
            self.assertIn("agent_id", strategy)
            self.assertIn("timestamp", strategy)
            self.assertIn("model", strategy)
            self.assertIn("strategy", strategy)  # Not strategy_text
            self.assertIn("full_reasoning", strategy)
            self.assertIn("prompt_tokens", strategy)
            self.assertIn("completion_tokens", strategy)

            # Verify field values
            self.assertEqual(strategy["round"], 1)
            self.assertEqual(strategy["agent_id"], i)
            self.assertEqual(strategy["strategy"], self.sample_strategies[i].strategy_text)
            self.assertEqual(strategy["full_reasoning"], self.sample_strategies[i].full_reasoning)
            self.assertEqual(strategy["prompt_tokens"], self.sample_strategies[i].prompt_tokens)
            self.assertEqual(strategy["completion_tokens"], self.sample_strategies[i].completion_tokens)

    def test_full_reasoning_preserved_without_truncation(self):
        """Test that full reasoning transcript is preserved without truncation."""
        # Create a strategy with very long reasoning
        long_reasoning = "This is a very long reasoning. " * 1000  # ~6000 characters
        strategy = StrategyRecord(
            strategy_id="strat_0_r1_test",
            agent_id=0,
            round=1,
            strategy_text="Test strategy",
            full_reasoning=long_reasoning,
            prompt_tokens=1000,
            completion_tokens=2000,
            model="test-model"
        )

        self.data_manager.save_strategies(1, [strategy])

        # Read back and verify
        file_path = self.data_manager.experiment_path / "rounds" / "strategies_r1.json"
        with open(file_path, 'r') as f:
            data = json.load(f)

        saved_reasoning = data["strategies"][0]["full_reasoning"]
        self.assertEqual(saved_reasoning, long_reasoning)
        self.assertEqual(len(saved_reasoning), len(long_reasoning))

    def test_metadata_fields_populated_correctly(self):
        """Test that all metadata fields are correctly populated."""
        round_num = 3
        self.data_manager.save_strategies(round_num, self.sample_strategies)

        file_path = self.data_manager.experiment_path / "rounds" / "strategies_r3.json"
        with open(file_path, 'r') as f:
            data = json.load(f)

        # Check file-level metadata
        self.assertEqual(data["round"], round_num)
        self.assertIn("timestamp", data)
        # Verify timestamp is ISO format
        datetime.fromisoformat(data["timestamp"])

        # Check strategy-level metadata
        for strategy in data["strategies"]:
            self.assertEqual(strategy["model"], "google/gemini-2.5-flash")
            self.assertIsInstance(strategy["prompt_tokens"], int)
            self.assertIsInstance(strategy["completion_tokens"], int)
            self.assertGreater(strategy["prompt_tokens"], 0)
            self.assertGreater(strategy["completion_tokens"], 0)
            # Verify timestamp is ISO format
            datetime.fromisoformat(strategy["timestamp"])

    def test_empty_strategies_list_handled(self):
        """Test handling of empty strategies list."""
        self.data_manager.save_strategies(1, [])

        file_path = self.data_manager.experiment_path / "rounds" / "strategies_r1.json"
        self.assertTrue(file_path.exists())

        with open(file_path, 'r') as f:
            data = json.load(f)
        self.assertEqual(data["strategies"], [])
        self.assertEqual(data["round"], 1)

    def test_special_characters_in_strategy_text(self):
        """Test handling of special characters in strategy text."""
        strategy = StrategyRecord(
            strategy_id="strat_0_r1_test",
            agent_id=0,
            round=1,
            strategy_text='Strategy with "quotes" and \n newlines and \t tabs',
            full_reasoning='Reasoning with special chars: {"key": "value"}',
            prompt_tokens=100,
            completion_tokens=50,
            model="test-model"
        )

        self.data_manager.save_strategies(1, [strategy])

        file_path = self.data_manager.experiment_path / "rounds" / "strategies_r1.json"
        with open(file_path, 'r') as f:
            data = json.load(f)

        saved_strategy = data["strategies"][0]
        self.assertEqual(saved_strategy["strategy"], strategy.strategy_text)
        self.assertEqual(saved_strategy["full_reasoning"], strategy.full_reasoning)


if __name__ == "__main__":
    unittest.main(verbosity=2)
