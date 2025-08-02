"""Unit tests for strategy storage functionality."""

import pytest
import json
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock
import uuid

from src.core.models import StrategyRecord
from src.utils.data_manager import DataManager


class TestStrategyStorage:
    """Test strategy storage functionality."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def data_manager(self, temp_dir):
        """Create a DataManager instance with temp directory."""
        return DataManager(base_path=temp_dir)

    @pytest.fixture
    def sample_strategies(self):
        """Create sample strategy records."""
        return [
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

    def test_save_strategies_creates_correct_file_path(self, data_manager, sample_strategies):
        """Test that save_strategies creates the correct file path format."""
        round_num = 1
        data_manager.save_strategies(round_num, sample_strategies)

        expected_path = data_manager.experiment_path / "rounds" / "strategies_r1.json"
        assert expected_path.exists()
        assert expected_path.is_file()

    def test_json_structure_matches_epic_specification(self, data_manager, sample_strategies):
        """Test that saved JSON structure matches Epic 2 specification."""
        round_num = 1
        data_manager.save_strategies(round_num, sample_strategies)

        # Read the saved file
        file_path = data_manager.experiment_path / "rounds" / "strategies_r1.json"
        with open(file_path, 'r') as f:
            data = json.load(f)

        # Check top-level structure
        assert "round" in data
        assert "timestamp" in data
        assert "strategies" in data
        assert data["round"] == round_num
        assert isinstance(data["strategies"], list)
        assert len(data["strategies"]) == 2

        # Check each strategy follows Epic 2 format
        for i, strategy in enumerate(data["strategies"]):
            assert "round" in strategy
            assert "agent_id" in strategy
            assert "timestamp" in strategy
            assert "model" in strategy
            assert "strategy" in strategy  # Not strategy_text
            assert "full_reasoning" in strategy
            assert "prompt_tokens" in strategy
            assert "completion_tokens" in strategy

            # Verify field values
            assert strategy["round"] == 1
            assert strategy["agent_id"] == i
            assert strategy["strategy"] == sample_strategies[i].strategy_text
            assert strategy["full_reasoning"] == sample_strategies[i].full_reasoning
            assert strategy["prompt_tokens"] == sample_strategies[i].prompt_tokens
            assert strategy["completion_tokens"] == sample_strategies[i].completion_tokens

    def test_full_reasoning_preserved_without_truncation(self, data_manager):
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

        data_manager.save_strategies(1, [strategy])

        # Read back and verify
        file_path = data_manager.experiment_path / "rounds" / "strategies_r1.json"
        with open(file_path, 'r') as f:
            data = json.load(f)

        saved_reasoning = data["strategies"][0]["full_reasoning"]
        assert saved_reasoning == long_reasoning
        assert len(saved_reasoning) == len(long_reasoning)

    def test_metadata_fields_populated_correctly(self, data_manager, sample_strategies):
        """Test that all metadata fields are correctly populated."""
        round_num = 3
        data_manager.save_strategies(round_num, sample_strategies)

        file_path = data_manager.experiment_path / "rounds" / "strategies_r3.json"
        with open(file_path, 'r') as f:
            data = json.load(f)

        # Check file-level metadata
        assert data["round"] == round_num
        assert "timestamp" in data
        # Verify timestamp is ISO format
        datetime.fromisoformat(data["timestamp"])

        # Check strategy-level metadata
        for strategy in data["strategies"]:
            assert strategy["model"] == "google/gemini-2.5-flash"
            assert isinstance(strategy["prompt_tokens"], int)
            assert isinstance(strategy["completion_tokens"], int)
            assert strategy["prompt_tokens"] > 0
            assert strategy["completion_tokens"] > 0
            # Verify timestamp is ISO format
            datetime.fromisoformat(strategy["timestamp"])

    def test_atomic_write_prevents_corruption(self, data_manager, sample_strategies, monkeypatch):
        """Test that atomic write behavior prevents file corruption."""
        round_num = 1

        # Mock shutil.move to simulate failure during write
        original_move = shutil.move
        call_count = 0

        def failing_move(src, dst):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise OSError("Simulated disk error")
            return original_move(src, dst)

        monkeypatch.setattr(shutil, "move", failing_move)

        # First attempt should fail
        with pytest.raises(OSError):
            data_manager.save_strategies(round_num, sample_strategies)

        # File should not exist after failed write
        file_path = data_manager.experiment_path / "rounds" / "strategies_r1.json"
        assert not file_path.exists()

        # Second attempt should succeed
        data_manager.save_strategies(round_num, sample_strategies)
        assert file_path.exists()

        # Verify file is valid JSON
        with open(file_path, 'r') as f:
            data = json.load(f)
        assert len(data["strategies"]) == 2

    def test_error_handling_for_write_failures(self, data_manager, sample_strategies, monkeypatch):
        """Test error handling for disk write failures."""
        round_num = 1

        # Make directory read-only to simulate permission error
        rounds_dir = data_manager.experiment_path / "rounds"
        rounds_dir.mkdir(exist_ok=True)

        # Mock the _write_json method to raise an exception
        def failing_write(path, data):
            raise PermissionError("No write permission")

        monkeypatch.setattr(data_manager, "_write_json", failing_write)

        # Should raise the permission error
        with pytest.raises(PermissionError):
            data_manager.save_strategies(round_num, sample_strategies)

    def test_multiple_rounds_saved_correctly(self, data_manager, sample_strategies):
        """Test that multiple rounds are saved to separate files."""
        # Save strategies for multiple rounds
        for round_num in range(1, 4):
            # Modify round number in strategies
            for strategy in sample_strategies:
                strategy.round = round_num
            data_manager.save_strategies(round_num, sample_strategies)

        # Verify all files exist
        for round_num in range(1, 4):
            file_path = data_manager.experiment_path / "rounds" / f"strategies_r{round_num}.json"
            assert file_path.exists()

            with open(file_path, 'r') as f:
                data = json.load(f)
            assert data["round"] == round_num
            assert all(s["round"] == round_num for s in data["strategies"])

    def test_empty_strategies_list_handled(self, data_manager):
        """Test handling of empty strategies list."""
        data_manager.save_strategies(1, [])

        file_path = data_manager.experiment_path / "rounds" / "strategies_r1.json"
        assert file_path.exists()

        with open(file_path, 'r') as f:
            data = json.load(f)
        assert data["strategies"] == []
        assert data["round"] == 1

    def test_special_characters_in_strategy_text(self, data_manager):
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

        data_manager.save_strategies(1, [strategy])

        file_path = data_manager.experiment_path / "rounds" / "strategies_r1.json"
        with open(file_path, 'r') as f:
            data = json.load(f)

        saved_strategy = data["strategies"][0]
        assert saved_strategy["strategy"] == strategy.strategy_text
        assert saved_strategy["full_reasoning"] == strategy.full_reasoning


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
