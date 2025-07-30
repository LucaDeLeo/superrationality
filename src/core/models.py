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
    round: int
    cooperation_rate: float
    average_score: float
    score_variance: float
    power_distribution: Dict[str, float]
    anonymized_games: List[AnonymizedGameResult]
    strategy_similarity: float = 0.0


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