"""Game execution flow for round-robin tournament."""

import logging
import asyncio
from typing import Dict, List, Tuple, Any
from datetime import datetime

from src.nodes.base import AsyncFlow
from src.nodes.subagent_decision import SubagentDecisionNode
from src.core.models import Agent, GameResult
from src.utils.game_logic import calculate_payoffs, update_powers

logger = logging.getLogger(__name__)


class GameExecutionFlow(AsyncFlow):
    """Execute round-robin tournament games for all agent pairs."""
    
    def __init__(self, subagent_node: SubagentDecisionNode = None):
        """Initialize GameExecutionFlow.
        
        Args:
            subagent_node: SubagentDecisionNode instance for making decisions
        """
        super().__init__()
        self.subagent_node = subagent_node
        
    def generate_round_matchups(self, agents: List[Agent]) -> List[Tuple[int, int]]:
        """Generate all unique agent pairs for round-robin tournament.
        
        In a round-robin tournament, each agent plays every other agent exactly once.
        For n agents, this generates n*(n-1)/2 unique matchups.
        
        Args:
            agents: List of agents participating in the round
            
        Returns:
            List of (agent1_id, agent2_id) tuples where agent1_id < agent2_id
            
        Raises:
            ValueError: If agent list is empty or validation fails
        """
        # Validate input
        if not agents:
            raise ValueError("Cannot generate matchups for empty agent list")
        
        if len(agents) < 2:
            raise ValueError(f"Need at least 2 agents for matchups, got {len(agents)}")
        
        matchups = []
        for i in range(len(agents)):
            for j in range(i + 1, len(agents)):
                matchups.append((agents[i].id, agents[j].id))
                
        # Validate we have the expected number of pairs
        expected_pairs = len(agents) * (len(agents) - 1) // 2
        if len(matchups) != expected_pairs:
            raise ValueError(
                f"Expected {expected_pairs} matchups for {len(agents)} agents, "
                f"but generated {len(matchups)}"
            )
            
        logger.info(f"Generated {len(matchups)} matchups for round-robin tournament")
        return matchups
    
    async def play_game(self, agent1: Agent, agent2: Agent, round_num: int, game_num: int, game_history: List[GameResult]) -> GameResult:
        """Play a single game between two agents.
        
        Uses subagent decisions based on agent strategies to determine actions,
        then calculates payoffs based on power levels.
        
        Args:
            agent1: First agent participating in the game
            agent2: Second agent participating in the game
            round_num: Current round number (1-10)
            game_num: Game number within round (1-45 for 10 agents)
            game_history: History of games played so far
            
        Returns:
            GameResult object with game outcome, including:
            - game_id in format r{round}_g{game_num}
            - player actions and payoffs
            - power levels before the game
        """
        game_id = f"r{round_num}_g{game_num}"
        
        # Record power levels before the game
        player1_power_before = agent1.power
        player2_power_before = agent2.power
        
        # Get strategies from agents
        strategy1 = agent1.strategy
        strategy2 = agent2.strategy
        
        # Use subagent to make decisions in parallel if available
        if self.subagent_node:
            player1_action, player2_action = await asyncio.gather(
                self.subagent_node.make_decision(agent1, agent2, strategy1, game_history),
                self.subagent_node.make_decision(agent2, agent1, strategy2, game_history)
            )
        else:
            # Fallback to placeholder actions if no subagent node
            player1_action = "COOPERATE"
            player2_action = "COOPERATE"
        
        # Calculate payoffs using game logic
        player1_payoff, player2_payoff = calculate_payoffs(
            agent1, agent2, player1_action, player2_action
        )
        
        return GameResult(
            game_id=game_id,
            round=round_num,
            player1_id=agent1.id,
            player2_id=agent2.id,
            player1_action=player1_action,
            player2_action=player2_action,
            player1_payoff=player1_payoff,
            player2_payoff=player2_payoff,
            player1_power_before=player1_power_before,
            player2_power_before=player2_power_before,
            timestamp=datetime.now().isoformat()
        )
    
    async def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute round-robin tournament for all agents.
        
        Orchestrates a complete round of games where each agent plays
        every other agent exactly once. Games are executed sequentially
        to ensure deterministic power evolution and consistent game history.
        
        Args:
            context: Experiment context containing:
                - agents: List[Agent] - all participating agents
                - round: int - current round number
            
        Returns:
            Updated context with games list added:
                - games: List[GameResult] - all games played this round
                
        Raises:
            ValueError: If required context keys are missing
        """
        agents = context["agents"]
        round_num = context["round"]
        
        # Generate all unique matchups
        matchups = self.generate_round_matchups(agents)
        
        # Create agent lookup for efficient access
        agent_lookup = {agent.id: agent for agent in agents}
        
        # Get game history from context (may be empty for first round)
        game_history = context.get("game_history", [])
        
        # Execute games sequentially
        games = []
        for game_num, (agent1_id, agent2_id) in enumerate(matchups, 1):
            agent1 = agent_lookup[agent1_id]
            agent2 = agent_lookup[agent2_id]
            
            # Play the game with current history
            game = await self.play_game(agent1, agent2, round_num, game_num, game_history)
            games.append(game)
            
            # Add this game to history for future games
            game_history.append(game)
            
            # Log progress
            if game_num % 10 == 0:
                logger.info(f"Completed {game_num}/{len(matchups)} games in round {round_num}")
        
        logger.info(f"Completed all {len(games)} games for round {round_num}")
        
        # Note: Power updates are handled at the experiment level after the round
        # This ensures consistent power levels during the round for testing
        
        # Update context with games and updated history
        context["games"] = games
        context["game_history"] = game_history
        return context