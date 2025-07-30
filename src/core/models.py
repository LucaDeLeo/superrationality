"""Data models for the acausal cooperation experiment."""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
from datetime import datetime


@dataclass
class Agent:
    """Represents a participant in the experiment."""
    id: int
    power: float = 100.0
    strategy: str = ""
    total_score: float = 0.0
    
    def __post_init__(self):
        """Validate agent attributes."""
        if not 0 <= self.id <= 9:
            raise ValueError(f"Agent id must be between 0 and 9, got {self.id}")
        if not 50 <= self.power <= 150:
            raise ValueError(f"Agent power must be between 50 and 150, got {self.power}")


@dataclass
class GameResult:
    """Records the outcome of a single prisoner's dilemma game."""
    game_id: str
    round: int
    player1_id: int
    player2_id: int
    player1_action: str
    player2_action: str
    player1_payoff: float = 0.0
    player2_payoff: float = 0.0
    player1_power_before: float = 0.0
    player2_power_before: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def __post_init__(self):
        """Validate game result attributes."""
        valid_actions = {"COOPERATE", "DEFECT"}
        if self.player1_action not in valid_actions:
            raise ValueError(f"Invalid player1_action: {self.player1_action}")
        if self.player2_action not in valid_actions:
            raise ValueError(f"Invalid player2_action: {self.player2_action}")


@dataclass
class StrategyRecord:
    """Stores the full strategy reasoning and decision from main agents."""
    strategy_id: str
    agent_id: int
    round: int
    strategy_text: str
    full_reasoning: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    model: str = "google/gemini-2.0-flash-exp:free"
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class AnonymizedGameResult:
    """Game result with anonymized agent IDs for round summaries."""
    round: int
    anonymous_id1: str
    anonymous_id2: str
    action1: str
    action2: str
    power_ratio: float


@dataclass
class RoundSummary:
    """Aggregated statistics for a complete round."""
    round: int  # Round number (1-10)
    cooperation_rate: float  # Percentage of COOPERATE actions (0-100)
    average_score: float     # Mean score across all agents
    score_variance: float    # Variance in scores
    power_distribution: Dict[str, float]  # Stats on power levels (mean, std, min, max)
    score_distribution: Dict[str, float] = field(default_factory=dict)  # Score stats (min, max, avg)
    anonymized_games: List[AnonymizedGameResult] = field(default_factory=list)
    strategy_similarity: float = 0.0
    
    @classmethod
    def from_game_results(cls, round_num: int, games: List[GameResult], agents: List[Agent]) -> 'RoundSummary':
        """Create RoundSummary from game results and agent states.
        
        Args:
            round_num: Round number
            games: List of GameResult objects for the round
            agents: List of Agent objects with current state
            
        Returns:
            RoundSummary object with calculated statistics
        """
        # Calculate cooperation rate
        total_actions = len(games) * 2  # Each game has 2 actions
        cooperate_count = sum(
            (1 if game.player1_action == "COOPERATE" else 0) +
            (1 if game.player2_action == "COOPERATE" else 0)
            for game in games
        )
        cooperation_rate = (cooperate_count / total_actions * 100) if total_actions > 0 else 0.0
        
        # Calculate average score and variance
        agent_scores = {agent.id: agent.total_score for agent in agents}
        scores = list(agent_scores.values())
        average_score = sum(scores) / len(scores) if scores else 0.0
        
        # Calculate variance
        if len(scores) > 1:
            mean = average_score
            variance = sum((score - mean) ** 2 for score in scores) / len(scores)
            score_variance = variance
        else:
            score_variance = 0.0
        
        # Calculate score distribution
        score_distribution = {
            'min': min(scores) if scores else 0.0,
            'max': max(scores) if scores else 0.0,
            'avg': average_score
        }
        
        # Calculate power distribution
        powers = [agent.power for agent in agents]
        power_distribution = {
            'mean': sum(powers) / len(powers) if powers else 100.0,
            'std': (sum((p - sum(powers)/len(powers)) ** 2 for p in powers) / len(powers)) ** 0.5 if len(powers) > 1 else 0.0,
            'min': min(powers) if powers else 100.0,
            'max': max(powers) if powers else 100.0
        }
        
        # Create anonymized games
        anonymized_games = []
        for game in games:
            # Create anonymous IDs based on position in this round
            anon_id1 = f"Agent_{hash(f'{round_num}_{game.player1_id}') % 1000}"
            anon_id2 = f"Agent_{hash(f'{round_num}_{game.player2_id}') % 1000}"
            
            # Calculate power ratio
            power_ratio = game.player2_power_before / game.player1_power_before if game.player1_power_before > 0 else 1.0
            
            anonymized_games.append(AnonymizedGameResult(
                round=round_num,
                anonymous_id1=anon_id1,
                anonymous_id2=anon_id2,
                action1=game.player1_action,
                action2=game.player2_action,
                power_ratio=power_ratio
            ))
        
        return cls(
            round=round_num,
            cooperation_rate=cooperation_rate,
            average_score=average_score,
            score_variance=score_variance,
            power_distribution=power_distribution,
            score_distribution=score_distribution,
            anonymized_games=anonymized_games,
            strategy_similarity=0.0  # This would be calculated separately if needed
        )


@dataclass
class ExperimentResult:
    """Complete experiment data including all rounds and analysis."""
    experiment_id: str
    start_time: str
    end_time: str = ""
    total_rounds: int = 0
    total_games: int = 0
    total_api_calls: int = 0
    total_cost: float = 0.0
    round_summaries: List[RoundSummary] = field(default_factory=list)
    acausal_indicators: Dict[str, float] = field(default_factory=dict)