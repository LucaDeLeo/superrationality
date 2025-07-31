"""Comprehensive tests for the anonymization system."""

import pytest
import random
import json
from pathlib import Path
import tempfile
import asyncio
from typing import List, Dict

from src.managers.anonymization import AnonymizationManager, AnonymizationMapping
from src.core.models import Agent, GameResult, RoundSummary, AnonymizedGameResult
from src.flows.experiment import RoundSummaryNode
from src.nodes.base import ContextKeys
from src.nodes.strategy_collection import StrategyCollectionNode
from src.core.config import Config
from src.core.api_client import OpenRouterClient


class TestAnonymizationManager:
    """Test the AnonymizationManager class."""
    
    def test_mapping_consistency_within_round(self):
        """Test that mappings are consistent within a round."""
        manager = AnonymizationManager(round_num=1, num_agents=5)
        
        # Check that repeated calls return same mapping
        agent_id = 2
        anon_id1 = manager.anonymize(agent_id)
        anon_id2 = manager.anonymize(agent_id)
        
        assert anon_id1 == anon_id2
        assert anon_id1.startswith("Player_1_")
        
        # Check reverse mapping
        assert manager.deanonymize(anon_id1) == agent_id
    
    def test_mapping_changes_between_rounds(self):
        """Test that mappings differ between rounds."""
        manager1 = AnonymizationManager(round_num=1, num_agents=5)
        manager2 = AnonymizationManager(round_num=2, num_agents=5)
        
        # Get mappings for same agent in different rounds
        agent_id = 2
        anon_id_r1 = manager1.anonymize(agent_id)
        anon_id_r2 = manager2.anonymize(agent_id)
        
        # Should have different anonymous IDs
        assert anon_id_r1 != anon_id_r2
        assert anon_id_r1.startswith("Player_1_")
        assert anon_id_r2.startswith("Player_2_")
    
    def test_agent_cannot_track_opponents(self):
        """Test that agents cannot track opponents across rounds."""
        num_agents = 5
        num_rounds = 3
        
        # Track which anonymous IDs each agent sees across rounds
        agent_observations = {i: [] for i in range(num_agents)}
        
        for round_num in range(1, num_rounds + 1):
            manager = AnonymizationManager(round_num=round_num, num_agents=num_agents)
            
            # Each agent observes all other agents' anonymous IDs
            for observer in range(num_agents):
                round_observations = []
                for observed in range(num_agents):
                    if observer != observed:
                        anon_id = manager.anonymize(observed)
                        round_observations.append(anon_id)
                agent_observations[observer].append(set(round_observations))
        
        # Check that anonymous IDs don't repeat across rounds
        for agent_id, observations in agent_observations.items():
            # Flatten all observations
            all_observed = set()
            for round_obs in observations:
                # Check no overlap with previous rounds
                assert len(all_observed & round_obs) == 0
                all_observed.update(round_obs)
    
    def test_anonymized_history_format(self):
        """Test that anonymized history is properly formatted."""
        # Create mock round summaries
        summaries = []
        for round_num in range(1, 4):
            manager = AnonymizationManager(round_num=round_num, num_agents=3)
            
            # Create anonymized games
            anon_games = []
            for i in range(3):
                game = AnonymizedGameResult(
                    round=round_num,
                    anonymous_id1=manager.anonymize(0),
                    anonymous_id2=manager.anonymize(1),
                    action1="COOPERATE",
                    action2="DEFECT",
                    power_ratio=1.2
                )
                anon_games.append(game)
            
            summary = RoundSummary(
                round=round_num,
                cooperation_rate=75.0,
                average_score=100.0,
                score_variance=10.0,
                power_distribution={"min": 90, "max": 110, "mean": 100, "std": 5},
                score_distribution={"min": 80, "max": 120, "avg": 100},
                anonymized_games=anon_games
            )
            summaries.append(summary)
        
        # Format the history
        from src.core.prompts import format_previous_rounds
        formatted = format_previous_rounds(summaries)
        
        # Check format
        assert "Round 1:" in formatted
        assert "Round 2:" in formatted
        assert "Round 3:" in formatted
        assert "Average cooperation rate: 75.0%" in formatted
        assert "Player_" in formatted  # Anonymous IDs present
        assert "COOPERATE/DEFECT" in formatted
        
        # Ensure no real agent IDs appear
        for i in range(3):
            assert f"agent_{i}" not in formatted
            assert f"Agent {i}" not in formatted
    
    def test_strategy_collection_with_anonymization(self):
        """Test that strategy collection uses anonymized order."""
        # Create agents
        agents = [Agent(id=i) for i in range(4)]
        
        # Create context with anonymization manager
        manager = AnonymizationManager(round_num=1, num_agents=4)
        context = {
            ContextKeys.AGENTS: agents,
            ContextKeys.ROUND: 1,
            ContextKeys.ANONYMIZATION_MANAGER: manager,
            ContextKeys.ROUND_SUMMARIES: []
        }
        
        # Get shuffled order
        shuffled_order = manager.get_shuffled_order()
        
        # Verify it's a valid permutation
        assert sorted(shuffled_order) == list(range(4))
        
        # Verify deterministic shuffling
        manager2 = AnonymizationManager(round_num=1, num_agents=4)
        assert manager2.get_shuffled_order() == shuffled_order
    
    def test_round_summary_anonymization(self):
        """Test that RoundSummaryNode properly anonymizes data."""
        # Create test data
        agents = [Agent(id=i, power=100 + i*10) for i in range(3)]
        games = [
            GameResult(
                game_id="g1",
                round=1,
                player1_id=0,
                player2_id=1,
                player1_action="COOPERATE",
                player2_action="COOPERATE",
                player1_payoff=3.0,
                player2_payoff=3.0,
                player1_power_before=100.0,
                player2_power_before=110.0
            ),
            GameResult(
                game_id="g2",
                round=1,
                player1_id=0,
                player2_id=2,
                player1_action="DEFECT",
                player2_action="COOPERATE",
                player1_payoff=5.0,
                player2_payoff=0.0,
                player1_power_before=100.0,
                player2_power_before=120.0
            )
        ]
        
        # Create context with anonymization manager
        manager = AnonymizationManager(round_num=1, num_agents=3)
        context = {
            ContextKeys.AGENTS: agents,
            ContextKeys.ROUND: 1,
            ContextKeys.GAMES: games,
            ContextKeys.ANONYMIZATION_MANAGER: manager,
            ContextKeys.ROUND_SUMMARIES: []
        }
        
        # Run RoundSummaryNode
        node = RoundSummaryNode()
        result_context = asyncio.run(node.execute(context))
        
        # Check summary was created
        summaries = result_context[ContextKeys.ROUND_SUMMARIES]
        assert len(summaries) == 1
        summary = summaries[0]
        
        # Check anonymized games
        assert len(summary.anonymized_games) == 2
        for anon_game in summary.anonymized_games:
            # Check anonymous ID format
            assert anon_game.anonymous_id1.startswith("Player_1_")
            assert anon_game.anonymous_id2.startswith("Player_1_")
            # Check actions preserved
            assert anon_game.action1 in ["COOPERATE", "DEFECT"]
            assert anon_game.action2 in ["COOPERATE", "DEFECT"]
    
    def test_deterministic_anonymization(self):
        """Test that anonymization is deterministic with same seed."""
        # Create two managers with same parameters
        manager1 = AnonymizationManager(round_num=3, num_agents=5)
        manager2 = AnonymizationManager(round_num=3, num_agents=5)
        
        # Check all mappings are identical
        for agent_id in range(5):
            assert manager1.anonymize(agent_id) == manager2.anonymize(agent_id)
        
        # Check shuffled orders are identical
        assert manager1.get_shuffled_order() == manager2.get_shuffled_order()
    
    def test_anonymization_data_persistence(self):
        """Test saving and loading anonymization mappings."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create and save mapping
            manager1 = AnonymizationManager(round_num=2, num_agents=4)
            mapping_path = Path(tmpdir) / "mapping.json"
            manager1.save_mapping(mapping_path)
            
            # Check file exists and contains correct data
            assert mapping_path.exists()
            with open(mapping_path, 'r') as f:
                data = json.load(f)
            
            assert data['round_num'] == 2
            assert data['seed_used'] == 2 * 12345
            assert len(data['real_to_anonymous']) == 4
            assert len(data['anonymous_to_real']) == 4
            
            # Load mapping
            manager2 = AnonymizationManager.load_mapping(mapping_path)
            
            # Verify loaded mapping matches original
            for agent_id in range(4):
                assert manager1.anonymize(agent_id) == manager2.anonymize(agent_id)


class TestAnonymizationIntegration:
    """Integration tests for anonymization in the full system."""
    
    @pytest.mark.asyncio
    async def test_full_round_with_anonymization(self):
        """Test a complete round with anonymization enabled."""
        # This would require mocking the API client
        # Placeholder for integration test
        pass
    
    def test_anonymization_prevents_tracking(self):
        """Verify that the anonymization system prevents cross-round tracking."""
        # Simulate multiple rounds of games
        num_agents = 4
        num_rounds = 5
        
        # Track game pairings across rounds
        pairings_by_round = []
        
        for round_num in range(1, num_rounds + 1):
            manager = AnonymizationManager(round_num=round_num, num_agents=num_agents)
            
            # Create games for this round
            round_pairings = []
            for i in range(num_agents):
                for j in range(i + 1, num_agents):
                    anon_i = manager.anonymize(i)
                    anon_j = manager.anonymize(j)
                    round_pairings.append((anon_i, anon_j))
            
            pairings_by_round.append(round_pairings)
        
        # Check that anonymous pairings don't repeat across rounds
        all_pairings = []
        for round_pairings in pairings_by_round:
            for pairing in round_pairings:
                # Normalize pairing order
                normalized = tuple(sorted(pairing))
                assert normalized not in all_pairings, "Same anonymous pairing appeared in multiple rounds"
                all_pairings.append(normalized)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])