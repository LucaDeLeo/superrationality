"""Integration tests for the full experiment flow."""

import asyncio
import json
import os
import signal
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch, call
import pytest

from src.core.config import Config
from src.core.models import Agent, GameResult, StrategyRecord, RoundSummary, ExperimentResult
from src.utils.data_manager import DataManager
from run_experiment import ExperimentRunner, RateLimiter


class MockOpenRouterClient:
    """Mock OpenRouter client for testing."""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.call_count = 0
        
    async def __aenter__(self):
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass
        
    async def complete(self, prompt: str, model: str = None, temperature: float = None):
        """Mock API completion."""
        self.call_count += 1
        
        # Return different responses based on prompt content
        if "You are agent" in prompt:
            # Strategy collection response
            return {
                "content": "I'll analyze the situation and choose to cooperate. STRATEGY: Always cooperate on first round to establish trust.",
                "usage": {"prompt_tokens": 100, "completion_tokens": 50}
            }
        else:
            # Subagent decision response
            action = "COOPERATE" if self.call_count % 3 != 0 else "DEFECT"
            return {
                "content": f"Based on the strategy, I'll {action}.",
                "usage": {"prompt_tokens": 80, "completion_tokens": 20}
            }


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test results."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def mock_config():
    """Create a mock config with test values."""
    config = Config()
    config.NUM_AGENTS = 3  # Smaller for faster tests
    config.NUM_ROUNDS = 2  # Fewer rounds for tests
    config.OPENROUTER_API_KEY = "test-key"
    return config


@pytest.mark.asyncio
async def test_complete_experiment_execution(temp_dir, mock_config, monkeypatch):
    """Test complete experiment execution with mocked API."""
    # Mock environment variable
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    
    # Create runner with temp directory
    runner = ExperimentRunner()
    runner.config = mock_config
    runner.data_manager = DataManager(base_path=temp_dir)
    
    # Mock the OpenRouterClient
    with patch('run_experiment.OpenRouterClient', MockOpenRouterClient):
        with patch('src.experiment.OpenRouterClient', MockOpenRouterClient):
            # Run experiment
            result = await runner.run_experiment()
    
    # Verify experiment completed
    assert result is not None
    assert result.total_rounds == 2
    assert result.total_games == 6  # 3 agents, C(3,2) = 3 games per round, 2 rounds
    assert result.total_api_calls > 0
    assert result.total_cost > 0
    
    # Verify files were created
    exp_path = runner.data_manager.experiment_path
    assert (exp_path / "experiment_results.json").exists()
    assert (exp_path / "rounds" / "strategies_r1.json").exists()
    assert (exp_path / "rounds" / "games_r1.json").exists()
    assert (exp_path / "rounds" / "summary_r1.json").exists()


@pytest.mark.asyncio
async def test_error_handling_and_partial_results(temp_dir, mock_config, monkeypatch):
    """Test error handling and partial result saving."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    
    runner = ExperimentRunner()
    runner.config = mock_config
    runner.data_manager = DataManager(base_path=temp_dir)
    
    # Create a mock that fails on second round
    class FailingMockClient(MockOpenRouterClient):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.round_count = 0
            
        async def complete(self, prompt: str, model: str = None, temperature: float = None):
            if "round 2" in prompt.lower() or self.call_count > 10:
                raise Exception("Simulated API failure")
            return await super().complete(prompt, model, temperature)
    
    # Mock the clients
    with patch('run_experiment.OpenRouterClient', FailingMockClient):
        with patch('src.experiment.OpenRouterClient', FailingMockClient):
            # Run experiment expecting partial failure
            try:
                result = await runner.run_experiment()
            except Exception:
                pass  # Expected to fail
    
    # Check error log exists
    exp_path = runner.data_manager.experiment_path
    error_log = exp_path / "experiment_errors.log"
    assert error_log.exists()
    
    # Verify partial results for round 1
    assert (exp_path / "rounds" / "strategies_r1.json").exists()
    assert (exp_path / "rounds" / "games_r1.json").exists()


@pytest.mark.asyncio
async def test_experiment_results_json_structure(temp_dir, mock_config, monkeypatch):
    """Test that experiment_results.json has correct structure."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    
    runner = ExperimentRunner()
    runner.config = mock_config
    runner.data_manager = DataManager(base_path=temp_dir)
    
    with patch('run_experiment.OpenRouterClient', MockOpenRouterClient):
        with patch('src.experiment.OpenRouterClient', MockOpenRouterClient):
            result = await runner.run_experiment()
    
    # Load and verify JSON structure
    results_path = runner.data_manager.experiment_path / "experiment_results.json"
    with open(results_path) as f:
        data = json.load(f)
    
    # Check required fields
    assert "experiment_id" in data
    assert "start_time" in data
    assert "end_time" in data
    assert "total_rounds" in data
    assert data["total_rounds"] == 2
    assert "total_games" in data
    assert "total_api_calls" in data
    assert "total_cost" in data
    assert "round_summaries" in data
    assert len(data["round_summaries"]) == 2
    assert "acausal_indicators" in data
    
    # Check round summary structure
    round_summary = data["round_summaries"][0]
    assert "round" in round_summary
    assert "cooperation_rate" in round_summary
    assert "average_score" in round_summary
    assert "score_variance" in round_summary
    assert "power_distribution" in round_summary
    assert "anonymized_games" in round_summary


@pytest.mark.asyncio
async def test_graceful_shutdown_handling(temp_dir, mock_config, monkeypatch):
    """Test interruption handling and graceful shutdown."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    
    runner = ExperimentRunner()
    runner.config = mock_config
    runner.data_manager = DataManager(base_path=temp_dir)
    
    # Simulate shutdown after first round
    async def delayed_shutdown():
        await asyncio.sleep(0.5)  # Let first round start
        runner.shutdown_requested = True
    
    # Run experiment with shutdown
    with patch('run_experiment.OpenRouterClient', MockOpenRouterClient):
        with patch('src.experiment.OpenRouterClient', MockOpenRouterClient):
            # Start shutdown task
            shutdown_task = asyncio.create_task(delayed_shutdown())
            
            # Run experiment
            result = await runner.run_experiment()
            
            # Wait for shutdown task
            await shutdown_task
    
    # Verify partial completion
    assert result.total_rounds < runner.config.NUM_ROUNDS
    assert result.end_time != ""
    
    # Verify saved results
    results_path = runner.data_manager.experiment_path / "experiment_results.json"
    assert results_path.exists()


@pytest.mark.asyncio
async def test_rate_limiter():
    """Test rate limiter functionality."""
    rate_limiter = RateLimiter(max_calls=3, window_seconds=1)
    
    start_time = datetime.now()
    
    # First 3 calls should be immediate
    for i in range(3):
        await rate_limiter.acquire()
    
    # 4th call should wait
    await rate_limiter.acquire()
    
    elapsed = (datetime.now() - start_time).total_seconds()
    assert elapsed >= 1.0  # Should have waited at least 1 second


@pytest.mark.asyncio
async def test_progress_tracking(temp_dir, mock_config, monkeypatch, capsys):
    """Test progress tracking and time estimation."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    
    runner = ExperimentRunner()
    runner.config = mock_config
    runner.data_manager = DataManager(base_path=temp_dir)
    
    with patch('run_experiment.OpenRouterClient', MockOpenRouterClient):
        with patch('src.experiment.OpenRouterClient', MockOpenRouterClient):
            await runner.run_experiment()
    
    # Check console output contains progress info
    captured = capsys.readouterr()
    assert "Round 1/2" in captured.out
    assert "Round 2/2" in captured.out
    assert "Cooperation Rate:" in captured.out
    assert "Average Score:" in captured.out
    assert "✅ Experiment completed successfully!" in captured.out


def test_data_manager_atomic_writes(temp_dir):
    """Test that DataManager writes are atomic."""
    dm = DataManager(base_path=temp_dir)
    
    # Create test data
    strategies = [
        StrategyRecord(
            strategy_id="s1",
            agent_id=0,
            round=1,
            strategy_text="Test strategy",
            full_reasoning="Test reasoning"
        )
    ]
    
    # Save strategies
    dm.save_strategies(1, strategies)
    
    # Verify file exists and no temp files remain
    strategies_path = dm.experiment_path / "rounds" / "strategies_r1.json"
    assert strategies_path.exists()
    
    # Check no .tmp files
    tmp_files = list(dm.experiment_path.glob("**/*.tmp"))
    assert len(tmp_files) == 0
    
    # Verify content
    with open(strategies_path) as f:
        data = json.load(f)
    assert data["round"] == 1
    assert len(data["strategies"]) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])