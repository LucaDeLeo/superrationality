"""Tests for node architecture and experiment orchestration."""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime
import logging

from src.nodes import (
    AsyncNode, AsyncFlow, AsyncParallelBatchNode,
    StrategyCollectionNode, SubagentDecisionNode,
    ContextKeys, validate_context
)
from src.core.models import Agent, StrategyRecord, GameResult
from src.core.config import Config
from src.core.api_client import OpenRouterClient
from src.flows.experiment import RoundSummaryNode, RoundFlow
from src.utils.game_logic import calculate_payoffs, update_powers


# Configure logging for tests
logging.basicConfig(level=logging.DEBUG)


class TestAsyncNode:
    """Test AsyncNode base class functionality."""
    
    @pytest.mark.asyncio
    async def test_async_node_retry_logic(self, tmp_path):
        """Test that AsyncNode retries on failure."""
        # Create error log file
        error_log = tmp_path / "experiment_errors.log"
        
        # Mock implementation that fails twice then succeeds
        call_count = 0
        
        class TestNode(AsyncNode):
            async def _execute_impl(self, context):
                nonlocal call_count
                call_count += 1
                if call_count < 3:
                    raise Exception(f"Test error {call_count}")
                return {"result": "success"}
        
        with patch("builtins.open", mock_open_wrapper(error_log)):
            node = TestNode(max_retries=3, retry_delay=0.01)
            result = await node.execute({"input": "test"})
        
        assert call_count == 3
        assert result["result"] == "success"
    
    @pytest.mark.asyncio
    async def test_async_node_max_retries_exceeded(self, tmp_path):
        """Test that AsyncNode fails after max retries."""
        error_log = tmp_path / "experiment_errors.log"
        
        class FailingNode(AsyncNode):
            async def _execute_impl(self, context):
                raise Exception("Persistent error")
        
        with patch("builtins.open", mock_open_wrapper(error_log)):
            node = FailingNode(max_retries=3, retry_delay=0.01)
            
            with pytest.raises(Exception) as exc_info:
                await node.execute({})
            
            assert "Persistent error" in str(exc_info.value)


class TestAsyncFlow:
    """Test AsyncFlow orchestration."""
    
    @pytest.mark.asyncio
    async def test_async_flow_sequential_execution(self):
        """Test that AsyncFlow executes nodes in sequence."""
        execution_order = []
        
        class TrackingNode(AsyncNode):
            def __init__(self, node_id):
                super().__init__()
                self.node_id = node_id
            
            async def _execute_impl(self, context):
                execution_order.append(self.node_id)
                context[f"node_{self.node_id}"] = True
                return context
        
        flow = AsyncFlow()
        flow.add_node(TrackingNode(1))
        flow.add_node(TrackingNode(2))
        flow.add_node(TrackingNode(3))
        
        result = await flow.run({"initial": True})
        
        assert execution_order == [1, 2, 3]
        assert result["node_1"] is True
        assert result["node_2"] is True
        assert result["node_3"] is True


class TestAsyncParallelBatchNode:
    """Test AsyncParallelBatchNode functionality."""
    
    @pytest.mark.asyncio
    async def test_parallel_batch_processing(self, tmp_path):
        """Test that items are processed in parallel."""
        error_log = tmp_path / "experiment_errors.log"
        
        class TestBatchNode(AsyncParallelBatchNode):
            async def process_item(self, item):
                await asyncio.sleep(0.01)  # Simulate work
                return item * 2
        
        with patch("builtins.open", mock_open_wrapper(error_log)):
            node = TestBatchNode()
            items = [1, 2, 3, 4, 5]
            
            start_time = asyncio.get_event_loop().time()
            results = await node.execute_batch(items)
            end_time = asyncio.get_event_loop().time()
            
            # Should be much faster than sequential (0.05s)
            assert (end_time - start_time) < 0.03
            assert results == [2, 4, 6, 8, 10]
    
    @pytest.mark.asyncio
    async def test_partial_failure_handling(self, tmp_path):
        """Test that batch processing continues despite partial failures."""
        error_log = tmp_path / "experiment_errors.log"
        
        class PartialFailureNode(AsyncParallelBatchNode):
            async def process_item(self, item):
                if item == 3:
                    raise Exception("Item 3 fails")
                return item * 2
        
        with patch("builtins.open", mock_open_wrapper(error_log)):
            node = PartialFailureNode()
            items = [1, 2, 3, 4, 5]
            
            results = await node.execute_batch(items)
            
            assert results == [2, 4, None, 8, 10]


class TestStrategyCollectionNode:
    """Test StrategyCollectionNode functionality."""
    
    @pytest.mark.asyncio
    async def test_strategy_collection(self):
        """Test strategy collection from agents."""
        # Mock API client
        mock_client = AsyncMock(spec=OpenRouterClient)
        mock_client.get_completion_text.return_value = """
        REASONING: I should cooperate to build trust.
        STRATEGY: Always cooperate with agents who have similar power levels.
        """
        
        # Mock config
        config = Mock(spec=Config)
        config.MAIN_MODEL = "test-model"
        
        # Create node
        node = StrategyCollectionNode(mock_client, config)
        
        # Create test context
        agents = [Agent(id=i) for i in range(3)]
        context = {
            ContextKeys.AGENTS: agents,
            ContextKeys.ROUND: 1,
            ContextKeys.ROUND_SUMMARIES: []
        }
        
        # Execute
        result = await node.execute(context)
        
        # Verify
        strategies = result[ContextKeys.STRATEGIES]
        assert len(strategies) == 3
        assert all(isinstance(s, StrategyRecord) for s in strategies)
        assert strategies[0].strategy_text == "Always cooperate with agents who have similar power levels."


class TestSubagentDecisionNode:
    """Test SubagentDecisionNode functionality."""
    
    @pytest.mark.asyncio
    async def test_decision_making(self):
        """Test subagent decision making."""
        # Mock API client
        mock_client = AsyncMock(spec=OpenRouterClient)
        mock_client.get_completion_text.return_value = "COOPERATE"
        
        # Mock config
        config = Mock(spec=Config)
        config.SUB_MODEL = "test-submodel"
        
        # Create node
        node = SubagentDecisionNode(mock_client, config)
        
        # Test agents
        agent1 = Agent(id=1, power=100)
        agent2 = Agent(id=2, power=120)
        
        # Make decision
        decision = await node.make_decision(
            agent1, agent2, 
            "Cooperate with stronger agents",
            []
        )
        
        assert decision == "COOPERATE"
    
    @pytest.mark.asyncio
    async def test_decision_parsing(self):
        """Test decision parsing from various responses."""
        # Mock API client
        mock_client = AsyncMock(spec=OpenRouterClient)
        config = Mock(spec=Config)
        config.SUB_MODEL = "test-submodel"
        
        node = SubagentDecisionNode(mock_client, config)
        
        # Test various response formats
        test_cases = [
            ("COOPERATE", "COOPERATE"),
            ("I will DEFECT", "DEFECT"),
            ("My action is: Cooperate", "COOPERATE"),
            ("Unclear response", "COOPERATE"),  # Default
        ]
        
        for response, expected in test_cases:
            mock_client.get_completion_text.return_value = response
            decision = await node.make_decision(
                Agent(id=1), Agent(id=2), "Test strategy", []
            )
            assert decision == expected


class TestGameLogic:
    """Test game logic functions."""
    
    def test_calculate_payoffs(self):
        """Test payoff calculation with power dynamics."""
        agent1 = Agent(id=1, power=100)
        agent2 = Agent(id=2, power=150)
        
        # Test mutual cooperation
        p1, p2 = calculate_payoffs(agent1, agent2, "COOPERATE", "COOPERATE")
        assert p1 == 3 * (150/100)  # 4.5
        assert p2 == 3 * (100/150)  # 2.0
        
        # Test defection scenarios
        p1, p2 = calculate_payoffs(agent1, agent2, "DEFECT", "COOPERATE")
        assert p1 == 5 * (150/100)  # 7.5
        assert p2 == 0
    
    def test_update_powers(self):
        """Test power update mechanism."""
        agents = [
            Agent(id=0, power=100),
            Agent(id=1, power=100),
            Agent(id=2, power=100)
        ]
        
        # Create games where agent 0 does well
        games = [
            GameResult(
                game_id="test1", round=1,
                player1_id=0, player2_id=1,
                player1_action="DEFECT", player2_action="COOPERATE",
                player1_payoff=5.0, player2_payoff=0.0,
                player1_power_before=100, player2_power_before=100
            ),
            GameResult(
                game_id="test2", round=1,
                player1_id=0, player2_id=2,
                player1_action="DEFECT", player2_action="COOPERATE",
                player1_payoff=5.0, player2_payoff=0.0,
                player1_power_before=100, player2_power_before=100
            ),
            GameResult(
                game_id="test3", round=1,
                player1_id=1, player2_id=2,
                player1_action="COOPERATE", player2_action="COOPERATE",
                player1_payoff=3.0, player2_payoff=3.0,
                player1_power_before=100, player2_power_before=100
            )
        ]
        
        update_powers(agents, games)
        
        # Agent 0 should have gained power (scored 10)
        # Agents 1 and 2 should have lost power (scored 3 each)
        # Average score is 16/3 = 5.33
        assert agents[0].power > 100  # Above average
        assert agents[1].power < 100  # Below average
        assert agents[2].power < 100  # Below average
        
        # Check bounds
        assert all(50 <= agent.power <= 150 for agent in agents)


class TestValidation:
    """Test validation functions."""
    
    def test_validate_context(self):
        """Test context validation."""
        context = {
            "key1": "value1",
            "key2": "value2"
        }
        
        # Should not raise for present keys
        validate_context(context, ["key1", "key2"])
        
        # Should raise for missing keys
        with pytest.raises(ValueError) as exc_info:
            validate_context(context, ["key1", "key3"])
        
        assert "key3" in str(exc_info.value)


# Helper function for mocking file operations
def mock_open_wrapper(target_path):
    """Create a mock for open() that works with context managers."""
    mock = MagicMock()
    mock.return_value.__enter__ = Mock(return_value=Mock())
    mock.return_value.__exit__ = Mock(return_value=None)
    return mock


if __name__ == "__main__":
    pytest.main([__file__, "-v"])