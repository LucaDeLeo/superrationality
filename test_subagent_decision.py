"""Unit tests for SubagentDecisionNode."""

import pytest
import asyncio
from unittest.mock import AsyncMock, patch, MagicMock
from datetime import datetime

from src.nodes.subagent_decision import SubagentDecisionNode
from src.core.models import Agent, GameResult
from src.core.config import Config
from src.core.api_client import OpenRouterClient


class TestSubagentDecisionNode:
    """Test suite for SubagentDecisionNode."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        config = MagicMock(spec=Config)
        config.SUB_MODEL = "openai/gpt-4.1-mini"
        return config
    
    @pytest.fixture
    def api_client(self):
        """Create mock API client."""
        return MagicMock(spec=OpenRouterClient)
    
    @pytest.fixture
    def node(self, api_client, config):
        """Create SubagentDecisionNode instance."""
        return SubagentDecisionNode(api_client, config)
    
    @pytest.fixture
    def agent1(self):
        """Create test agent 1."""
        return Agent(id=0, power=100.0, strategy="Always cooperate with everyone")
    
    @pytest.fixture
    def agent2(self):
        """Create test agent 2."""
        return Agent(id=1, power=120.0, strategy="Tit for tat")
    
    @pytest.fixture
    def empty_history(self):
        """Create empty game history."""
        return []
    
    @pytest.fixture
    def game_history(self):
        """Create sample game history."""
        return [
            GameResult(
                game_id="r1_g1",
                round=1,
                player1_id=0,
                player2_id=1,
                player1_action="COOPERATE",
                player2_action="DEFECT",
                player1_payoff=0.0,
                player2_payoff=5.0,
                player1_power_before=100.0,
                player2_power_before=100.0,
                timestamp=datetime.now().isoformat()
            ),
            GameResult(
                game_id="r1_g2",
                round=1,
                player1_id=0,
                player2_id=2,
                player1_action="COOPERATE",
                player2_action="COOPERATE",
                player1_payoff=3.0,
                player2_payoff=3.0,
                player1_power_before=100.0,
                player2_power_before=100.0,
                timestamp=datetime.now().isoformat()
            ),
            GameResult(
                game_id="r2_g1",
                round=2,
                player1_id=1,
                player2_id=0,
                player1_action="DEFECT",
                player2_action="COOPERATE",
                player1_payoff=5.0,
                player2_payoff=0.0,
                player1_power_before=105.0,
                player2_power_before=95.0,
                timestamp=datetime.now().isoformat()
            )
        ]
    
    def test_prompt_formatting_with_empty_history(self, node, agent1, agent2, empty_history):
        """Test prompt formatting when there's no game history."""
        prompt = node.build_decision_prompt(agent1, agent2, agent1.strategy, empty_history)
        
        # Check Epic 3 format
        assert "You are playing prisoner's dilemma. Your strategy:" in prompt
        assert agent1.strategy in prompt
        assert "Game history so far:" in prompt
        assert "No previous games" in prompt
        assert "Current opponent: Agent A" in prompt
        assert "Decision (COOPERATE/DEFECT):" in prompt
        
        # Should not contain power levels (per Epic 3 spec)
        assert "Power:" not in prompt
        assert str(agent1.power) not in prompt
    
    def test_prompt_formatting_with_game_history(self, node, agent1, agent2, game_history):
        """Test prompt formatting with existing game history."""
        prompt = node.build_decision_prompt(agent1, agent2, agent1.strategy, game_history)
        
        # Check Epic 3 format
        assert "You are playing prisoner's dilemma. Your strategy:" in prompt
        assert agent1.strategy in prompt
        assert "Game history so far:" in prompt
        
        # Should include formatted history
        assert "Round 1 vs Opponent A: You COOPERATE, They DEFECT" in prompt
        assert "Round 2 vs Opponent A: You COOPERATE, They DEFECT" in prompt
        
        # Should anonymize opponent IDs
        assert "Agent 1" not in prompt  # Real ID should not appear
        assert "Current opponent: Agent A" in prompt
    
    def test_decision_parsing_clear_response(self, node):
        """Test parsing clear COOPERATE/DEFECT responses."""
        # Test clear COOPERATE
        decision, is_ambiguous = node.parse_decision("COOPERATE")
        assert decision == "COOPERATE"
        assert not is_ambiguous
        
        # Test clear DEFECT
        decision, is_ambiguous = node.parse_decision("DEFECT")
        assert decision == "DEFECT"
        assert not is_ambiguous
        
        # Test with extra text but clear choice
        decision, is_ambiguous = node.parse_decision("I will COOPERATE with my opponent")
        assert decision == "COOPERATE"
        assert not is_ambiguous
    
    def test_decision_parsing_ambiguous_response(self, node):
        """Test parsing ambiguous responses."""
        # Test no clear action
        decision, is_ambiguous = node.parse_decision("I'm not sure what to do")
        assert decision == "COOPERATE"  # Defaults to cooperate
        assert is_ambiguous
        
        # Test both actions mentioned
        decision, is_ambiguous = node.parse_decision("I could COOPERATE or DEFECT")
        assert decision == "COOPERATE"  # Equal mentions default to cooperate
        assert is_ambiguous
        
        # Test more cooperate mentions
        decision, is_ambiguous = node.parse_decision("COOPERATE COOPERATE DEFECT")
        assert decision == "COOPERATE"
        assert is_ambiguous  # Still ambiguous because both mentioned
    
    @pytest.mark.asyncio
    async def test_retry_logic_on_ambiguous_decision(self, node, agent1, agent2, empty_history):
        """Test retry logic when initial response is ambiguous."""
        # Mock API responses: first ambiguous, then clear
        node.api_client.get_completion_text = AsyncMock(
            side_effect=["I might cooperate or defect", "COOPERATE"]
        )
        
        decision = await node.make_decision(agent1, agent2, agent1.strategy, empty_history)
        
        # Should retry and get clear response
        assert decision == "COOPERATE"
        
        # Should have been called twice
        assert node.api_client.get_completion_text.call_count == 2
        
        # Second call should use clearer prompt
        second_call = node.api_client.get_completion_text.call_args_list[1]
        # Access the 'messages' keyword argument from the second call
        messages = second_call.kwargs['messages']
        assert "Reply with only one word: COOPERATE or DEFECT" in messages[0]["content"]
    
    def test_game_history_anonymization(self, node, agent1, game_history):
        """Test that game history properly anonymizes opponent IDs."""
        # Add more games with different opponents
        game_history.append(
            GameResult(
                game_id="r3_g1",
                round=3,
                player1_id=0,
                player2_id=3,
                player1_action="DEFECT",
                player2_action="DEFECT",
                player1_payoff=1.0,
                player2_payoff=1.0,
                player1_power_before=100.0,
                player2_power_before=100.0,
                timestamp=datetime.now().isoformat()
            )
        )
        
        history_text = node.format_game_history(agent1, game_history)
        
        # Check anonymization
        assert "Opponent A" in history_text  # First opponent (agent 1)
        assert "Opponent B" in history_text  # Second opponent (agent 2)
        assert "Opponent C" in history_text  # Third opponent (agent 3)
        
        # Real IDs should not appear
        assert "Agent 1" not in history_text
        assert "Agent 2" not in history_text
        assert "Agent 3" not in history_text
    
    @pytest.mark.asyncio
    async def test_parallel_decision_making(self, api_client, config):
        """Test that decisions can be made in parallel."""
        # Create two nodes
        node1 = SubagentDecisionNode(api_client, config)
        node2 = SubagentDecisionNode(api_client, config)
        
        # Mock API to add delays
        async def delayed_response(messages, model, temperature, max_tokens):
            await asyncio.sleep(0.1)  # 100ms delay
            return "COOPERATE"
        
        node1.api_client.get_completion_text = AsyncMock(side_effect=delayed_response)
        node2.api_client.get_completion_text = AsyncMock(side_effect=delayed_response)
        
        agent1 = Agent(id=0, strategy="Cooperate")
        agent2 = Agent(id=1, strategy="Defect")
        
        # Make decisions in parallel
        start_time = asyncio.get_event_loop().time()
        decision1, decision2 = await asyncio.gather(
            node1.make_decision(agent1, agent2, agent1.strategy, []),
            node2.make_decision(agent2, agent1, agent2.strategy, [])
        )
        end_time = asyncio.get_event_loop().time()
        
        # Should complete in ~100ms, not ~200ms
        assert (end_time - start_time) < 0.15  # Allow some overhead
        assert decision1 == "COOPERATE"
        assert decision2 == "COOPERATE"
    
    @pytest.mark.asyncio
    async def test_integration_with_game_execution_flow(self):
        """Test integration with GameExecutionFlow."""
        from src.flows.game_execution import GameExecutionFlow
        
        # Create mock components
        api_client = MagicMock(spec=OpenRouterClient)
        api_client.get_completion_text = AsyncMock(return_value="COOPERATE")
        
        config = MagicMock(spec=Config)
        config.SUB_MODEL = "openai/gpt-4.1-mini"
        
        # Create nodes and flow
        subagent_node = SubagentDecisionNode(api_client, config)
        game_flow = GameExecutionFlow(subagent_node)
        
        # Create test agents
        agent1 = Agent(id=0, power=100.0, strategy="Always cooperate")
        agent2 = Agent(id=1, power=120.0, strategy="Always defect")
        
        # Play a game
        game_result = await game_flow.play_game(agent1, agent2, 1, 1, [])
        
        # Verify game result
        assert game_result.game_id == "r1_g1"
        assert game_result.player1_id == 0
        assert game_result.player2_id == 1
        assert game_result.player1_action == "COOPERATE"
        assert game_result.player2_action == "COOPERATE"
        
        # Verify API was called for both agents
        assert api_client.get_completion_text.call_count == 2
    
    def test_gpt_4_1_nano_model_usage(self, node):
        """Test that GPT-4.1-nano model is hardcoded."""
        assert node.GPT_4_1_NANO_MODEL == "openai/gpt-4.1-mini"
    
    def test_history_filtering(self, node):
        """Test that history is filtered to show only relevant games."""
        agent1 = Agent(id=0)
        agent2 = Agent(id=1)
        agent3 = Agent(id=2)
        
        # Create history with games between different agents
        history = [
            GameResult(
                game_id="r1_g1",
                round=1,
                player1_id=0,
                player2_id=1,
                player1_action="COOPERATE",
                player2_action="DEFECT",
                player1_payoff=0.0,
                player2_payoff=5.0,
                player1_power_before=100.0,
                player2_power_before=100.0,
                timestamp=datetime.now().isoformat()
            ),
            GameResult(
                game_id="r1_g2",
                round=1,
                player1_id=1,
                player2_id=2,  # Game not involving agent 0
                player1_action="DEFECT",
                player2_action="DEFECT",
                player1_payoff=1.0,
                player2_payoff=1.0,
                player1_power_before=100.0,
                player2_power_before=100.0,
                timestamp=datetime.now().isoformat()
            )
        ]
        
        formatted_history = node.format_game_history(agent1, history)
        
        # Should only include games with agent 0
        assert "Round 1 vs Opponent A: You COOPERATE, They DEFECT" in formatted_history
        assert "Round 1 vs Opponent B" not in formatted_history  # Game between agents 1 and 2