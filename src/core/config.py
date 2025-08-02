"""Configuration management for the acausal cooperation experiment."""

from dataclasses import dataclass
from typing import Optional
import os
from pathlib import Path
import yaml


@dataclass
class Config:
    """Core experiment configuration."""

    # Experiment parameters
    NUM_AGENTS: int = 10
    NUM_ROUNDS: int = 10

    # Model configuration
    MAIN_MODEL: str = "google/gemini-2.5-flash"
    SUB_MODEL: str = "openai/GPT-4.1-nano"

    # API configuration
    OPENROUTER_API_KEY: Optional[str] = None

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

    def _load_environment_variables(self):
        """Load environment variables, particularly API keys."""
        if not self.OPENROUTER_API_KEY:
            self.OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

        if not self.OPENROUTER_API_KEY:
            raise ValueError("OPENROUTER_API_KEY environment variable is required")

    @classmethod
    def from_yaml(cls, config_path: Path) -> "Config":
        """Load configuration from a YAML file.

        Args:
            config_path: Path to the YAML configuration file

        Returns:
            Config instance with loaded parameters
        """
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)

        return cls(**config_data)

    def get_api_headers(self) -> dict:
        """Get headers for OpenRouter API requests.

        Returns:
            Dictionary with authorization headers
        """
        return {
            "Authorization": f"Bearer {self.OPENROUTER_API_KEY}",
            "Content-Type": "application/json"
        }
