"""Configuration management for the acausal cooperation experiment."""

from dataclasses import dataclass
import os
import logging

logger = logging.getLogger(__name__)


@dataclass
class Config:
    """Core experiment configuration - simplified for single model."""

    # Experiment parameters
    NUM_AGENTS: int = int(os.getenv('NUM_AGENTS', 10))
    NUM_ROUNDS: int = int(os.getenv('NUM_ROUNDS', 10))
    ENABLE_MULTI_MODEL: bool = False
    scenarios: list = None  # List of scenario configurations for multi-model experiments

    # Model configuration (fixed)
    MAIN_MODEL: str = "google/gemini-2.5-flash"
    SUB_MODEL: str = "openai/gpt-4-1106-preview"  # Using available model

    # API configuration
    OPENROUTER_API_KEY: str = ""

    # Game parameters
    COOPERATE_COOPERATE_PAYOFF: tuple = (3, 3)
    COOPERATE_DEFECT_PAYOFF: tuple = (0, 5)
    DEFECT_COOPERATE_PAYOFF: tuple = (5, 0)
    DEFECT_DEFECT_PAYOFF: tuple = (1, 1)

    # Cost limit
    MAX_COST: float = 10.0

    def __post_init__(self):
        """Validate configuration after initialization."""
        self._validate_parameters()
        self._load_environment_variables()

    def _validate_parameters(self):
        """Validate experiment parameters."""
        if not isinstance(self.NUM_AGENTS, int) or self.NUM_AGENTS <= 0:
            raise ValueError(f"NUM_AGENTS must be a positive integer, got {self.NUM_AGENTS}")

        if not isinstance(self.NUM_ROUNDS, int) or self.NUM_ROUNDS <= 0:
            raise ValueError(f"NUM_ROUNDS must be a positive integer, got {self.NUM_ROUNDS}")

        if self.MAX_COST <= 0:
            raise ValueError(f"MAX_COST must be positive, got {self.MAX_COST}")

    def _load_environment_variables(self):
        """Load environment variables, particularly API keys."""
        if not self.OPENROUTER_API_KEY:
            self.OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

        if not self.OPENROUTER_API_KEY:
            raise ValueError("OPENROUTER_API_KEY environment variable is required")

    def get_api_headers(self) -> dict:
        """Get headers for OpenRouter API requests.

        Returns:
            Dictionary with authorization headers
        """
        return {
            "Authorization": f"Bearer {self.OPENROUTER_API_KEY}",
            "Content-Type": "application/json"
        }

    def get_payoff_matrix(self) -> dict:
        """Get the payoff matrix for the prisoner's dilemma game.

        Returns:
            Dictionary mapping (action1, action2) to (payoff1, payoff2)
        """
        return {
            ("COOPERATE", "COOPERATE"): self.COOPERATE_COOPERATE_PAYOFF,
            ("COOPERATE", "DEFECT"): self.COOPERATE_DEFECT_PAYOFF,
            ("DEFECT", "COOPERATE"): self.DEFECT_COOPERATE_PAYOFF,
            ("DEFECT", "DEFECT"): self.DEFECT_DEFECT_PAYOFF
        }