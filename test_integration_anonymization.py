"""Integration test for anonymization in the full system."""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import asyncio
from src.managers.anonymization import AnonymizationManager
from src.core.models import Agent, GameResult, RoundSummary
from src.flows.experiment import RoundSummaryNode
from src.nodes.base import ContextKeys
from src.core.prompts import format_previous_rounds, format_round_summary


async def test_integration():
    """Test anonymization integration."""
    print("\n=== Testing Anonymization Integration ===\n")
    
    # Setup test data
    agents = [Agent(id=i, power=100 + i*5) for i in range(4)]
    
    # Round 1
    print("Round 1:")
    round1_manager = AnonymizationManager(round_num=1, num_agents=4)
    
    # Create some games
    games_r1 = [
        GameResult(
            game_id="g1_1", round=1,
            player1_id=0, player2_id=1,
            player1_action="COOPERATE", player2_action="COOPERATE",
            player1_payoff=3.0, player2_payoff=3.0,
            player1_power_before=100.0, player2_power_before=105.0
        ),
        GameResult(
            game_id="g1_2", round=1,
            player1_id=2, player2_id=3,
            player1_action="DEFECT", player2_action="COOPERATE",
            player1_payoff=5.0, player2_payoff=0.0,
            player1_power_before=110.0, player2_power_before=115.0
        )
    ]
    
    # Create context
    context = {
        ContextKeys.AGENTS: agents,
        ContextKeys.ROUND: 1,
        ContextKeys.GAMES: games_r1,
        ContextKeys.ANONYMIZATION_MANAGER: round1_manager,
        ContextKeys.ROUND_SUMMARIES: []
    }
    
    # Run RoundSummaryNode
    summary_node = RoundSummaryNode()
    context = await summary_node.execute(context)
    
    round1_summary = context[ContextKeys.ROUND_SUMMARIES][0]
    print(f"  Cooperation rate: {round1_summary.cooperation_rate:.1%}")
    print(f"  Anonymized games:")
    for game in round1_summary.anonymized_games[:2]:
        print(f"    {game.anonymous_id1} vs {game.anonymous_id2}: {game.action1}/{game.action2}")
    
    # Round 2 with different anonymization
    print("\nRound 2:")
    round2_manager = AnonymizationManager(round_num=2, num_agents=4)
    
    games_r2 = [
        GameResult(
            game_id="g2_1", round=2,
            player1_id=0, player2_id=1,  # Same pairing
            player1_action="DEFECT", player2_action="DEFECT",
            player1_payoff=1.0, player2_payoff=1.0,
            player1_power_before=100.0, player2_power_before=105.0
        )
    ]
    
    context[ContextKeys.ROUND] = 2
    context[ContextKeys.GAMES] = games_r2
    context[ContextKeys.ANONYMIZATION_MANAGER] = round2_manager
    
    context = await summary_node.execute(context)
    round2_summary = context[ContextKeys.ROUND_SUMMARIES][1]
    
    print(f"  Cooperation rate: {round2_summary.cooperation_rate:.1%}")
    print(f"  Anonymized games:")
    for game in round2_summary.anonymized_games:
        print(f"    {game.anonymous_id1} vs {game.anonymous_id2}: {game.action1}/{game.action2}")
    
    # Test formatted previous rounds
    print("\nFormatted Previous Rounds:")
    formatted = format_previous_rounds(context[ContextKeys.ROUND_SUMMARIES])
    print(formatted)
    
    # Verify no tracking possible
    print("\n\nVerifying anonymization prevents tracking:")
    # Agent 0 vs Agent 1 in both rounds
    r1_game = round1_summary.anonymized_games[0]
    r2_game = round2_summary.anonymized_games[0]
    
    print(f"  Round 1: {r1_game.anonymous_id1} vs {r1_game.anonymous_id2}")
    print(f"  Round 2: {r2_game.anonymous_id1} vs {r2_game.anonymous_id2}")
    print(f"  IDs match across rounds: {r1_game.anonymous_id1 == r2_game.anonymous_id1}")
    
    print("\n✅ Integration test complete!")


if __name__ == "__main__":
    asyncio.run(test_integration())