"""Configuration management for the acausal cooperation experiment."""

from dataclasses import dataclass
import os
import json
import logging
from pathlib import Path
from typing import List, Optional

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
        self._load_scenarios()

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

    def _load_scenarios(self):
        """Load scenarios from scenarios.json if it exists."""
        scenarios_path = Path("scenarios.json")
        if scenarios_path.exists():
            try:
                with open(scenarios_path, 'r') as f:
                    data = json.load(f)
                    # Convert to ScenarioConfig objects
                    from src.core.models import ScenarioConfig
                    self.scenarios = [
                        ScenarioConfig(
                            name=s['name'],
                            model_distribution=s['model_distribution']
                        )
                        for s in data.get('scenarios', [])
                    ]
                    # Enable multi-model if scenarios are loaded
                    if self.scenarios:
                        self.ENABLE_MULTI_MODEL = True
                        logger.info(f"Loaded {len(self.scenarios)} model scenarios")
            except Exception as e:
                logger.warning(f"Failed to load scenarios.json: {e}")
                if self.scenarios is None:
                    self.scenarios = []
        else:
            logger.debug("No scenarios.json found, using single model mode")
            if self.scenarios is None:
                self.scenarios = []

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