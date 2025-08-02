"""Unit tests for configuration management."""

import pytest
import os
from pathlib import Path
import tempfile
import yaml
from src.core.config import Config


class TestConfig:
    """Test suite for Config class."""

    def test_config_initialization_with_valid_parameters(self):
        """Test Config class initialization with valid parameters."""
        # Set required environment variable
        os.environ["OPENROUTER_API_KEY"] = "test_key"

        config = Config(
            NUM_AGENTS=5,
            NUM_ROUNDS=3,
            MAIN_MODEL="test/model1",
            SUB_MODEL="test/model2"
        )

        assert config.NUM_AGENTS == 5
        assert config.NUM_ROUNDS == 3
        assert config.MAIN_MODEL == "test/model1"
        assert config.SUB_MODEL == "test/model2"
        assert config.OPENROUTER_API_KEY == "test_key"

    def test_default_parameters(self):
        """Test Config class with default parameters."""
        os.environ["OPENROUTER_API_KEY"] = "test_key"

        config = Config()

        assert config.NUM_AGENTS == 10
        assert config.NUM_ROUNDS == 10
        assert config.MAIN_MODEL == "google/gemini-2.5-flash"
        assert config.SUB_MODEL == "openai/GPT-4.1-nano"

    def test_parameter_validation_invalid_num_agents(self):
        """Test parameter validation with invalid NUM_AGENTS."""
        os.environ["OPENROUTER_API_KEY"] = "test_key"

        with pytest.raises(ValueError, match="NUM_AGENTS must be a positive integer"):
            Config(NUM_AGENTS=-1)

        with pytest.raises(ValueError, match="NUM_AGENTS must be a positive integer"):
            Config(NUM_AGENTS=0)

    def test_parameter_validation_invalid_num_rounds(self):
        """Test parameter validation with invalid NUM_ROUNDS."""
        os.environ["OPENROUTER_API_KEY"] = "test_key"

        with pytest.raises(ValueError, match="NUM_ROUNDS must be a positive integer"):
            Config(NUM_ROUNDS=-1)

        with pytest.raises(ValueError, match="NUM_ROUNDS must be a positive integer"):
            Config(NUM_ROUNDS=0)

    def test_environment_variable_loading(self):
        """Test environment variable loading for API key."""
        # Clear API key from environment if set
        if "OPENROUTER_API_KEY" in os.environ:
            del os.environ["OPENROUTER_API_KEY"]

        # Should raise error when API key is not set
        with pytest.raises(ValueError, match="OPENROUTER_API_KEY environment variable is required"):
            Config()

        # Set API key and test successful loading
        os.environ["OPENROUTER_API_KEY"] = "test_api_key_123"
        config = Config()
        assert config.OPENROUTER_API_KEY == "test_api_key_123"

    def test_api_key_validation_with_explicit_key(self):
        """Test that explicit API key bypasses environment variable."""
        # Clear API key from environment
        if "OPENROUTER_API_KEY" in os.environ:
            del os.environ["OPENROUTER_API_KEY"]

        config = Config(OPENROUTER_API_KEY="explicit_key")
        assert config.OPENROUTER_API_KEY == "explicit_key"

    def test_from_yaml(self):
        """Test loading configuration from YAML file."""
        os.environ["OPENROUTER_API_KEY"] = "test_key"

        # Create temporary YAML file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump({
                'NUM_AGENTS': 7,
                'NUM_ROUNDS': 5,
                'MAIN_MODEL': 'yaml/model1',
                'SUB_MODEL': 'yaml/model2'
            }, f)
            temp_path = Path(f.name)

        try:
            config = Config.from_yaml(temp_path)
            assert config.NUM_AGENTS == 7
            assert config.NUM_ROUNDS == 5
            assert config.MAIN_MODEL == "yaml/model1"
            assert config.SUB_MODEL == "yaml/model2"
        finally:
            temp_path.unlink()

    def test_get_api_headers(self):
        """Test API headers generation."""
        os.environ["OPENROUTER_API_KEY"] = "test_bearer_token"

        config = Config()
        headers = config.get_api_headers()

        assert headers["Authorization"] == "Bearer test_bearer_token"
        assert headers["Content-Type"] == "application/json"
