"""Game execution flow for round-robin tournament."""

import logging
from typing import Dict, List, Tuple, Any
from datetime import datetime

from src.nodes.base import AsyncFlow
from src.core.models import Agent, GameResult

logger = logging.getLogger(__name__)


class GameExecutionFlow(AsyncFlow):
    """Execute round-robin tournament games for all agent pairs."""
    
    def __init__(self):
        """Initialize GameExecutionFlow."""
        super().__init__()
        
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
    
    async def play_game(self, agent1: Agent, agent2: Agent, round_num: int, game_num: int) -> GameResult:
        """Play a single game between two agents.
        
        This method currently returns placeholder results. In future stories,
        this will integrate with the SubagentDecisionNode to get actual
        agent decisions based on their strategies.
        
        Args:
            agent1: First agent participating in the game
            agent2: Second agent participating in the game
            round_num: Current round number (1-10)
            game_num: Game number within round (1-45 for 10 agents)
            
        Returns:
            GameResult object with game outcome, including:
            - game_id in format r{round}_g{game_num}
            - player actions and payoffs
            - power levels before the game
        """
        # For now, return placeholder game results
        # Actual implementation will be added in future stories
        game_id = f"r{round_num}_g{game_num}"
        
        # Placeholder actions - will be replaced with actual decision logic
        player1_action = "COOPERATE"
        player2_action = "COOPERATE"
        
        # Placeholder payoffs - will be replaced with actual calculation
        player1_payoff = 3.0
        player2_payoff = 3.0
        
        return GameResult(
            game_id=game_id,
            round=round_num,
            player1_id=agent1.id,
            player2_id=agent2.id,
            player1_action=player1_action,
            player2_action=player2_action,
            player1_payoff=player1_payoff,
            player2_payoff=player2_payoff,
            player1_power_before=agent1.power,
            player2_power_before=agent2.power,
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
        
        # Execute games sequentially
        games = []
        for game_num, (agent1_id, agent2_id) in enumerate(matchups, 1):
            agent1 = agent_lookup[agent1_id]
            agent2 = agent_lookup[agent2_id]
            
            # Play the game
            game = await self.play_game(agent1, agent2, round_num, game_num)
            games.append(game)
            
            # Log progress
            if game_num % 10 == 0:
                logger.info(f"Completed {game_num}/{len(matchups)} games in round {round_num}")
        
        logger.info(f"Completed all {len(games)} games for round {round_num}")
        
        # Update context with games
        context["games"] = games
        return context