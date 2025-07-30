"""Unit tests for parallel strategy collection."""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime
import time

from src.nodes import StrategyCollectionNode, ContextKeys
from src.core.models import Agent, StrategyRecord
from src.core.config import Config
from src.core.api_client import OpenRouterClient
from run_experiment import RateLimiter


@pytest.fixture
def mock_config():
    """Create mock config for testing."""
    config = MagicMock(spec=Config)
    config.MAIN_MODEL = "google/gemini-2.0-flash-exp:free"
    config.NUM_AGENTS = 10
    return config


@pytest.fixture
def mock_api_client():
    """Create mock API client."""
    client = AsyncMock(spec=OpenRouterClient)
    return client


@pytest.fixture
def mock_rate_limiter():
    """Create mock rate limiter."""
    limiter = AsyncMock(spec=RateLimiter)
    limiter.acquire = AsyncMock()
    return limiter


@pytest.fixture
def agents():
    """Create test agents."""
    return [Agent(id=i) for i in range(10)]


@pytest.fixture
def test_context(agents):
    """Create test context."""
    return {
        ContextKeys.AGENTS: agents,
        ContextKeys.ROUND: 1,
        ContextKeys.ROUND_SUMMARIES: []
    }


@pytest.mark.asyncio
async def test_parallel_strategy_collection_success(
    mock_api_client, mock_config, mock_rate_limiter, agents, test_context
):
    """Test successful parallel collection of 10 strategies."""
    # Setup mock responses
    mock_api_client.get_completion_text = AsyncMock(
        side_effect=[
            f"REASONING: Agent {i} reasoning\nSTRATEGY: Strategy for agent {i}"
            for i in range(10)
        ]
    )
    
    # Create node
    node = StrategyCollectionNode(mock_api_client, mock_config, mock_rate_limiter)
    
    # Record start time
    start_time = time.time()
    
    # Execute
    result = await node.execute(test_context)
    
    # Record end time
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    # Verify all strategies collected
    strategies = result[ContextKeys.STRATEGIES]
    assert len(strategies) == 10
    
    # Verify each strategy
    for i, strategy in enumerate(strategies):
        assert isinstance(strategy, StrategyRecord)
        assert strategy.agent_id == i
        assert strategy.round == 1
        assert f"Strategy for agent {i}" in strategy.strategy_text
        assert strategy.model == "google/gemini-2.0-flash-exp:free"
    
    # Verify parallel execution (should complete quickly, not 10x sequential time)
    # In real parallel execution, this should be much less than 10 seconds
    assert elapsed_time < 5.0  # Generous timeout for test environment
    
    # Verify API calls made
    assert mock_api_client.get_completion_text.call_count == 10
    
    # Verify rate limiting applied
    assert mock_rate_limiter.acquire.call_count == 10
    
    # Verify stats added to context
    assert 'strategy_collection_stats' in result
    stats = result['strategy_collection_stats']
    assert stats['total_agents'] == 10
    assert stats['successful_collections'] == 10
    assert stats['failure_count'] == 0


@pytest.mark.asyncio
async def test_strategy_collection_with_timeouts(
    mock_api_client, mock_config, mock_rate_limiter, agents, test_context
):
    """Test timeout handling with mocked slow API responses."""
    # Track which agents have been called
    call_count = 0
    agent_order = []
    
    async def mock_completion(messages, *args, **kwargs):
        nonlocal call_count
        # Get the agent ID from the prompt
        prompt = messages[0]["content"]
        # Extract agent ID - it's after "You are Agent "
        start = prompt.find("You are Agent ") + len("You are Agent ")
        agent_id = int(prompt[start:start+1])
        agent_order.append(agent_id)
        call_count += 1
        
        if agent_id in [2, 5]:
            # Simulate timeout by sleeping longer than the timeout
            await asyncio.sleep(0.2)  # Longer than 100ms timeout
            return "REASONING: Late response\nSTRATEGY: Late strategy"
        else:
            return f"REASONING: Agent {agent_id}\nSTRATEGY: Strategy {agent_id}"
    
    mock_api_client.get_completion_text = mock_completion
    
    # Create node with shorter timeout for testing
    node = StrategyCollectionNode(mock_api_client, mock_config, mock_rate_limiter)
    node.timeout = 0.1  # 100ms timeout for quick test
    
    # Execute
    result = await node.execute(test_context)
    
    # Verify strategies collected (8 successful, 2 timeouts with fallbacks)
    strategies = result[ContextKeys.STRATEGIES]
    assert len(strategies) == 10
    
    # Count fallback strategies
    fallback_count = sum(1 for s in strategies if s.model == "fallback")
    assert fallback_count == 2
    
    # Verify fallback strategies have correct agent IDs
    fallback_strategies = [s for s in strategies if s.model == "fallback"]
    fallback_agent_ids = sorted([s.agent_id for s in fallback_strategies])
    assert fallback_agent_ids == [2, 5]
    
    for fallback in fallback_strategies:
        assert fallback.strategy_text == "Always Cooperate"
        assert "Timeout occurred" in fallback.full_reasoning
        assert "_fallback" in fallback.strategy_id
    
    # Verify stats
    stats = result['strategy_collection_stats']
    # Since we return fallback strategies, they still count as "successful"
    assert stats['failure_count'] == 0  # No None values returned
    assert stats['successful_collections'] == 10  # All agents got strategies (some fallback)


@pytest.mark.asyncio
async def test_strategy_collection_with_failures(
    mock_api_client, mock_config, mock_rate_limiter, agents, test_context
):
    """Test graceful degradation when some agents fail."""
    # Setup mock responses with some failures
    async def mock_completion(messages, *args, **kwargs):
        # Get the agent ID from the prompt
        prompt = messages[0]["content"]
        # Extract agent ID - it's after "You are Agent "
        start = prompt.find("You are Agent ") + len("You are Agent ")
        agent_id = int(prompt[start:start+1])
        
        if agent_id in [1, 4, 7]:
            raise Exception(f"API error for agent {agent_id}")
        else:
            return f"REASONING: Agent {agent_id}\nSTRATEGY: Strategy {agent_id}"
    
    mock_api_client.get_completion_text = mock_completion
    
    # Create node
    node = StrategyCollectionNode(mock_api_client, mock_config, mock_rate_limiter)
    
    # Execute
    result = await node.execute(test_context)
    
    # Verify strategies collected (7 successful, 3 failures with fallbacks)
    strategies = result[ContextKeys.STRATEGIES]
    assert len(strategies) == 10
    
    # Count fallback strategies
    fallback_count = sum(1 for s in strategies if s.model == "fallback")
    assert fallback_count == 3
    
    # Verify fallback strategies have error information
    fallback_strategies = [s for s in strategies if s.model == "fallback"]
    for fallback in fallback_strategies:
        assert fallback.strategy_text == "Always Cooperate"
        assert "API error" in fallback.full_reasoning
    
    # Verify stats
    stats = result['strategy_collection_stats']
    # Since we return fallback strategies, they still count as "successful"
    assert stats['failure_count'] == 0  # No None values returned
    assert stats['successful_collections'] == 10  # All agents got strategies (some fallback)


@pytest.mark.asyncio
async def test_rate_limiting_compliance(
    mock_api_client, mock_config, agents, test_context
):
    """Test that rate limiting is applied correctly."""
    # Create real rate limiter with strict limits
    rate_limiter = RateLimiter(max_calls=5, window_seconds=1)
    
    # Mock fast API responses
    mock_api_client.get_completion_text = AsyncMock(
        side_effect=[
            f"REASONING: Agent {i}\nSTRATEGY: Strategy {i}"
            for i in range(10)
        ]
    )
    
    # Create node
    node = StrategyCollectionNode(mock_api_client, mock_config, rate_limiter)
    
    # Record timing
    start_time = time.time()
    
    # Execute
    result = await node.execute(test_context)
    
    # Record end time
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    # With 10 agents and rate limit of 5/second, should take at least 1 second
    assert elapsed_time >= 1.0
    
    # Verify all strategies collected
    assert len(result[ContextKeys.STRATEGIES]) == 10


@pytest.mark.asyncio
async def test_parallel_execution_verification(
    mock_api_client, mock_config, mock_rate_limiter, agents, test_context
):
    """Verify that agents are queried simultaneously, not sequentially."""
    call_times = []
    
    async def track_call_time(*args, **kwargs):
        call_times.append(time.time())
        await asyncio.sleep(0.5)  # Simulate API delay
        agent_id = len(call_times) - 1
        return f"REASONING: Agent {agent_id}\nSTRATEGY: Strategy {agent_id}"
    
    mock_api_client.get_completion_text = AsyncMock(side_effect=track_call_time)
    
    # Create node
    node = StrategyCollectionNode(mock_api_client, mock_config, mock_rate_limiter)
    
    # Execute
    start_time = time.time()
    result = await node.execute(test_context)
    end_time = time.time()
    
    # Verify all calls were made
    assert len(call_times) == 10
    
    # In parallel execution, all calls should start within a short window
    # Calculate time spread of call initiation
    first_call = min(call_times)
    last_call = max(call_times)
    call_spread = last_call - first_call
    
    # All calls should start within 100ms of each other (generous for test environment)
    assert call_spread < 0.1
    
    # Total execution time should be close to single call time (0.5s) not 10 * 0.5s
    total_time = end_time - start_time
    assert total_time < 1.0  # Should complete in ~0.5s plus overhead


@pytest.mark.asyncio
async def test_context_validation(
    mock_api_client, mock_config, mock_rate_limiter
):
    """Test that context validation works correctly."""
    node = StrategyCollectionNode(mock_api_client, mock_config, mock_rate_limiter)
    
    # Test with missing AGENTS key
    bad_context = {
        ContextKeys.ROUND: 1,
        ContextKeys.ROUND_SUMMARIES: []
    }
    
    with pytest.raises(ValueError, match="Context missing required keys"):
        await node.execute(bad_context)
    
    # Test with missing ROUND key
    bad_context = {
        ContextKeys.AGENTS: [Agent(id=0)],
        ContextKeys.ROUND_SUMMARIES: []
    }
    
    with pytest.raises(ValueError, match="Context missing required keys"):
        await node.execute(bad_context)


@pytest.mark.asyncio
async def test_strategy_parsing_edge_cases(
    mock_api_client, mock_config, mock_rate_limiter, test_context
):
    """Test strategy parsing with various response formats."""
    # Create only 3 agents for this test
    test_context[ContextKeys.AGENTS] = [Agent(id=i) for i in range(3)]
    
    # Test different response formats
    mock_api_client.get_completion_text = AsyncMock(
        side_effect=[
            # Proper format
            "REASONING: Good reasoning\nSTRATEGY: Good strategy",
            # Missing STRATEGY label
            "Just some text without proper formatting",
            # Very long strategy (should be truncated)
            "REASONING: Long\nSTRATEGY: " + " ".join([f"word{i}" for i in range(150)])
        ]
    )
    
    node = StrategyCollectionNode(mock_api_client, mock_config, mock_rate_limiter)
    result = await node.execute(test_context)
    
    strategies = result[ContextKeys.STRATEGIES]
    assert len(strategies) == 3
    
    # Check proper format
    assert strategies[0].strategy_text == "Good strategy"
    
    # Check fallback for missing format
    assert "Cooperate with agents of similar power" in strategies[1].strategy_text
    
    # Check truncation (100 words max)
    words = strategies[2].strategy_text.split()
    assert len(words) == 100


@pytest.mark.asyncio
async def test_round_summary_usage(
    mock_api_client, mock_config, mock_rate_limiter, agents
):
    """Test that round summaries are properly used in prompts."""
    from src.core.models import RoundSummary, AnonymizedGameResult
    
    # Create context with round summaries
    round_summary = RoundSummary(
        round=1,
        cooperation_rate=0.7,
        average_score=2.5,
        score_variance=0.5,
        power_distribution={"mean": 100.0, "std": 10.0},
        anonymized_games=[]
    )
    
    context = {
        ContextKeys.AGENTS: agents[:1],  # Just test with one agent
        ContextKeys.ROUND: 2,
        ContextKeys.ROUND_SUMMARIES: [round_summary]
    }
    
    # Capture the prompt used
    captured_prompt = None
    
    async def capture_prompt(messages, *args, **kwargs):
        nonlocal captured_prompt
        captured_prompt = messages[0]["content"]
        return "REASONING: Test\nSTRATEGY: Test strategy"
    
    mock_api_client.get_completion_text = AsyncMock(side_effect=capture_prompt)
    
    node = StrategyCollectionNode(mock_api_client, mock_config, mock_rate_limiter)
    await node.execute(context)
    
    # Verify round summary data appears in prompt
    assert "Round 1:" in captured_prompt
    assert "Cooperation rate: 70.0%" in captured_prompt
    assert "Average score: 2.5" in captured_prompt
    assert "mean=100.0, std=10.0" in captured_prompt


if __name__ == "__main__":
    pytest.main([__file__, "-v"])