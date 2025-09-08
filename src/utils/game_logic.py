"""Game logic for prisoner's dilemma with power dynamics."""

from typing import Tuple, List
from src.core.models import Agent, GameResult
import math
import random


# Payoff matrix (row player perspective)
PAYOFF_MATRIX = {
    ("COOPERATE", "COOPERATE"): (3, 3),
    ("COOPERATE", "DEFECT"): (0, 5),
    ("DEFECT", "COOPERATE"): (5, 0),
    ("DEFECT", "DEFECT"): (1, 1)
}


def calculate_payoffs(
    player1: Agent,
    player2: Agent,
    action1: str,
    action2: str
) -> Tuple[float, float]:
    """Calculate payoffs for both players considering power levels.
    
    Args:
        player1: First player
        player2: Second player
        action1: First player's action
        action2: Second player's action
        
    Returns:
        Tuple of (player1_payoff, player2_payoff)
    """
    # Get base payoffs from matrix
    base_payoff1, base_payoff2 = PAYOFF_MATRIX[(action1, action2)]
    
    # Apply power multipliers
    # Player 1's payoff is multiplied by player 2's power / player 1's power
    payoff1 = base_payoff1 * (player2.power / player1.power)
    # Player 2's payoff is multiplied by player 1's power / player 2's power
    payoff2 = base_payoff2 * (player1.power / player2.power)
    
    return payoff1, payoff2


def update_powers(agents: List[Agent], games: List[GameResult]) -> None:
    """Update agent power levels based on round performance.
    
    Power update formula:
    - Calculate average score for the round
    - Agents above average gain power proportionally
    - Agents below average lose power proportionally
    - Power is capped between 50 and 150
    
    Args:
        agents: List of agents to update
        games: List of game results from the round
    """
    # Calculate round scores for each agent
    round_scores = {agent.id: 0.0 for agent in agents}
    
    for game in games:
        round_scores[game.player1_id] += game.player1_payoff
        round_scores[game.player2_id] += game.player2_payoff
    
    # Calculate average score
    total_score = sum(round_scores.values())
    average_score = total_score / len(agents) if agents else 0
    
    # Update powers based on performance
    for agent in agents:
        score = round_scores[agent.id]
        
        # Calculate power change (10% of difference from average)
        power_change = 0.1 * (score - average_score)
        
        # Update power with bounds
        new_power = agent.power + power_change
        agent.power = max(50.0, min(150.0, new_power))
        
        # Update total score
        agent.total_score += score


def randomize_powers_for_round(agents: List[Agent], mean: float = 100.0, std_dev: float = 15.0) -> None:
    """Randomize agent power levels at the start of each round.
    
    Creates a normal distribution (bell curve) of power levels.
    
    Args:
        agents: List of agents to update
        mean: Mean of the normal distribution (default 100)
        std_dev: Standard deviation of the distribution (default 15)
    """
    for agent in agents:
        # Generate random power from normal distribution
        new_power = random.gauss(mean, std_dev)
        
        # Bound the power between 50 and 150
        agent.power = max(50.0, min(150.0, new_power))


def calculate_payoff(agent1_power: float, agent2_power: float, action1: str, action2: str) -> float:
    """Calculate payoff for agent1 (wrapper for compatibility).
    
    Args:
        agent1_power: Power level of agent1
        agent2_power: Power level of agent2
        action1: Agent1's action
        action2: Agent2's action
        
    Returns:
        Payoff for agent1
    """
    # Get base payoff from matrix
    base_payoff, _ = PAYOFF_MATRIX[(action1, action2)]
    
    # Apply power multiplier
    return base_payoff * (agent2_power / agent1_power)


def update_power(current_power: float, won: bool) -> float:
    """Update agent power based on win/loss (wrapper for compatibility).
    
    Args:
        current_power: Current power level
        won: Whether agent won (had higher payoff)
        
    Returns:
        Updated power level
    """
    # Simple 5% adjustment based on win/loss
    if won:
        new_power = current_power * 1.05
    else:
        new_power = current_power * 0.95
    
    # Apply bounds
    return max(50.0, min(150.0, new_power))


def create_game_result(
    round_num: int,
    game_num: int,
    player1: Agent,
    player2: Agent,
    action1: str,
    action2: str
) -> GameResult:
    """Create a game result with calculated payoffs.
    
    Args:
        round_num: Current round number
        game_num: Game number within the round
        player1: First player
        player2: Second player
        action1: First player's action
        action2: Second player's action
        
    Returns:
        GameResult object
    """
    payoff1, payoff2 = calculate_payoffs(player1, player2, action1, action2)
    
    return GameResult(
        game_id=f"r{round_num}_g{game_num}",
        round=round_num,
        player1_id=player1.id,
        player2_id=player2.id,
        player1_action=action1,
        player2_action=action2,
        player1_payoff=payoff1,
        player2_payoff=payoff2,
        player1_power_before=player1.power,
        player2_power_before=player2.power
    )