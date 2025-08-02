"""Configuration management for the acausal cooperation experiment."""

from dataclasses import dataclass, field
from typing import Optional, Dict, List
import os
from pathlib import Path
import yaml
import logging

from src.core.models import ModelConfig, ScenarioConfig

logger = logging.getLogger(__name__)


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

    # Multi-model feature flag
    ENABLE_MULTI_MODEL: bool = False

    # Multi-model configuration (only used when ENABLE_MULTI_MODEL=True)
    model_configs: Dict[str, ModelConfig] = field(default_factory=dict)
    scenarios: List[ScenarioConfig] = field(default_factory=list)

    def __post_init__(self):
        """Validate configuration after initialization."""
        self._validate_parameters()
        self._load_environment_variables()
        if self.ENABLE_MULTI_MODEL:
            self._validate_multi_model_config()

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

        # Load additional API keys if specified in model configs
        if self.ENABLE_MULTI_MODEL:
            for model_name, model_config in self.model_configs.items():
                if model_config.api_key_env != "OPENROUTER_API_KEY":
                    # Check if custom API key is needed
                    if not os.getenv(model_config.api_key_env):
                        logger.warning(
                            f"API key {model_config.api_key_env} not found for model {model_name}. "
                            f"Will use OPENROUTER_API_KEY as fallback."
                        )
                        model_config.api_key_env = "OPENROUTER_API_KEY"

    def _validate_multi_model_config(self):
        """Validate multi-model configuration when feature is enabled."""
        if not self.scenarios:
            logger.info("Multi-model enabled but no scenarios defined. Using default single-model behavior.")
            return

        # Validate each scenario
        for scenario in self.scenarios:
            total_agents = sum(scenario.model_distribution.values())
            if total_agents != self.NUM_AGENTS:
                raise ValueError(
                    f"Scenario '{scenario.name}' assigns {total_agents} agents "
                    f"but NUM_AGENTS is {self.NUM_AGENTS}"
                )

            # Check that all models in distribution are configured
            for model_type in scenario.model_distribution:
                if model_type not in self.model_configs:
                    raise ValueError(
                        f"Scenario '{scenario.name}' references model '{model_type}' "
                        f"which is not in model_configs"
                    )

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

        # Extract basic config
        basic_config = {
            k: v for k, v in config_data.items()
            if k not in ["model_configs", "scenarios"]
        }

        # Create instance with basic config
        config = cls(**basic_config)

        # Only process multi-model config if feature is enabled
        if config.ENABLE_MULTI_MODEL and "model_configs" in config_data:
            # Load model configurations
            for model_name, model_data in config_data["model_configs"].items():
                config.model_configs[model_name] = ModelConfig(**model_data)

            # Load scenarios
            if "scenarios" in config_data:
                for scenario_data in config_data["scenarios"]:
                    config.scenarios.append(ScenarioConfig(**scenario_data))

            # Run multi-model validation
            config._validate_multi_model_config()
        elif "model_configs" in config_data or "scenarios" in config_data:
            logger.warning(
                "Multi-model configuration found but ENABLE_MULTI_MODEL=False. "
                "Set ENABLE_MULTI_MODEL=true to use multi-model features."
            )

        return config

    def get_api_headers(self) -> dict:
        """Get headers for OpenRouter API requests.

        Returns:
            Dictionary with authorization headers
        """
        return {
            "Authorization": f"Bearer {self.OPENROUTER_API_KEY}",
            "Content-Type": "application/json"
        }
