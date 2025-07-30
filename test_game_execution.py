"""Unit tests for game execution flow."""

import pytest
import asyncio
from typing import List, Dict, Any

from src.flows.game_execution import GameExecutionFlow
from src.core.models import Agent, GameResult


class TestGameExecutionFlow:
    """Test suite for GameExecutionFlow."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.flow = GameExecutionFlow()
        self.agents = [Agent(id=i) for i in range(10)]
        
    def test_generate_round_matchups_creates_45_pairs(self):
        """Test that exactly 45 unique pairs are generated for 10 agents."""
        matchups = self.flow.generate_round_matchups(self.agents)
        assert len(matchups) == 45
        
    def test_all_pairs_are_unique(self):
        """Test that all generated pairs are unique."""
        matchups = self.flow.generate_round_matchups(self.agents)
        
        # Convert to sets for uniqueness check
        matchup_set = set(matchups)
        assert len(matchup_set) == len(matchups)
        
        # Also check no duplicate in reverse order
        for i, j in matchups:
            assert (j, i) not in matchup_set
            
    def test_each_agent_plays_nine_games(self):
        """Test that each agent appears exactly 9 times."""
        matchups = self.flow.generate_round_matchups(self.agents)
        
        # Count appearances
        agent_counts = {i: 0 for i in range(10)}
        for i, j in matchups:
            agent_counts[i] += 1
            agent_counts[j] += 1
            
        # Each agent should play exactly 9 games
        for agent_id, count in agent_counts.items():
            assert count == 9, f"Agent {agent_id} plays {count} games, expected 9"
            
    def test_matchup_order_is_consistent(self):
        """Test that matchup order is consistent across multiple calls."""
        matchups1 = self.flow.generate_round_matchups(self.agents)
        matchups2 = self.flow.generate_round_matchups(self.agents)
        
        assert matchups1 == matchups2
        
    def test_game_id_format_is_correct(self):
        """Test that game IDs follow the correct format."""
        async def run_test():
            context = {
                "agents": self.agents,
                "round": 3
            }
            
            # Run the flow
            result = await self.flow.run(context)
            games = result["games"]
            
            # Check all game IDs
            for idx, game in enumerate(games):
                expected_id = f"r3_g{idx + 1}"
                assert game.game_id == expected_id
                
        # Run async test
        asyncio.run(run_test())
        
    def test_matchup_validation_different_agent_count(self):
        """Test validation works for different agent counts."""
        # Test with 5 agents (should generate 10 pairs)
        agents_5 = [Agent(id=i) for i in range(5)]
        matchups = self.flow.generate_round_matchups(agents_5)
        assert len(matchups) == 10
        
        # Test with 3 agents (should generate 3 pairs)
        agents_3 = [Agent(id=i) for i in range(3)]
        matchups = self.flow.generate_round_matchups(agents_3)
        assert len(matchups) == 3
        
    def test_no_self_play(self):
        """Test that no agent plays against themselves."""
        matchups = self.flow.generate_round_matchups(self.agents)
        
        for i, j in matchups:
            assert i != j, f"Agent {i} is playing against themselves"
            
    def test_sequential_execution(self):
        """Test that games are executed sequentially with correct numbering."""
        async def run_test():
            context = {
                "agents": self.agents,
                "round": 1
            }
            
            # Run the flow
            result = await self.flow.run(context)
            games = result["games"]
            
            # Verify sequential game numbers
            for idx, game in enumerate(games):
                # Extract game number from ID
                game_num = int(game.game_id.split("_g")[1])
                assert game_num == idx + 1
                
        asyncio.run(run_test())
        
    def test_context_updates(self):
        """Test that context is properly updated with games."""
        async def run_test():
            context = {
                "agents": self.agents,
                "round": 1
            }
            
            # Run the flow
            result = await self.flow.run(context)
            
            # Check context was updated
            assert "games" in result
            assert len(result["games"]) == 45
            assert all(isinstance(g, GameResult) for g in result["games"])
            
        asyncio.run(run_test())
        
    def test_power_levels_preserved(self):
        """Test that power levels are correctly recorded in games."""
        async def run_test():
            # Set different power levels
            for i, agent in enumerate(self.agents):
                agent.power = 50.0 + i * 10.0
                
            context = {
                "agents": self.agents,
                "round": 1
            }
            
            # Run the flow
            result = await self.flow.run(context)
            games = result["games"]
            
            # Check power levels are recorded
            for game in games:
                agent1 = next(a for a in self.agents if a.id == game.player1_id)
                agent2 = next(a for a in self.agents if a.id == game.player2_id)
                
                assert game.player1_power_before == agent1.power
                assert game.player2_power_before == agent2.power
                
        asyncio.run(run_test())


if __name__ == "__main__":
    pytest.main([__file__, "-v"])