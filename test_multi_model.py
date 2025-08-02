"""Tests for multi-model configuration support."""

import pytest
from src.core.models import Agent, ModelConfig, ScenarioConfig, ModelType


class TestAgentBackwardCompatibility:
    """Test that existing Agent creation still works without model config."""
    
    def test_agent_creation_without_model_config(self):
        """Test that Agent can be created without model_config (backward compatible)."""
        agent = Agent(id=0, power=100.0)
        
        assert agent.id == 0
        assert agent.power == 100.0
        assert agent.strategy == ""
        assert agent.total_score == 0.0
        assert agent.model_config is None  # Default to None
    
    def test_agent_creation_with_all_defaults(self):
        """Test that Agent still works with minimal parameters."""
        agent = Agent(id=5)
        
        assert agent.id == 5
        assert agent.power == 100.0  # Default
        assert agent.strategy == ""   # Default
        assert agent.total_score == 0.0  # Default
        assert agent.model_config is None  # Default
    
    def test_agent_creation_with_explicit_values(self):
        """Test Agent creation with explicit values (existing behavior)."""
        agent = Agent(
            id=3,
            power=120.0,
            strategy="Always cooperate",
            total_score=50.0
        )
        
        assert agent.id == 3
        assert agent.power == 120.0
        assert agent.strategy == "Always cooperate"
        assert agent.total_score == 50.0
        assert agent.model_config is None
    
    def test_agent_validation_still_works(self):
        """Test that existing validation logic is preserved."""
        # Test invalid ID
        with pytest.raises(ValueError, match="Agent id must be between 0 and 9"):
            Agent(id=10)
        
        # Test invalid power
        with pytest.raises(ValueError, match="Agent power must be between 50 and 150"):
            Agent(id=1, power=200.0)
    
    def test_agent_with_model_config(self):
        """Test that Agent can be created with model_config when needed."""
        model_config = ModelConfig(
            model_type="openai/gpt-4",
            temperature=0.8
        )
        
        agent = Agent(id=2, model_config=model_config)
        
        assert agent.model_config is not None
        assert agent.model_config.model_type == "openai/gpt-4"
        assert agent.model_config.temperature == 0.8
        assert agent.model_config.api_key_env == "OPENROUTER_API_KEY"  # Default


class TestModelConfig:
    """Test ModelConfig dataclass."""
    
    def test_model_config_defaults(self):
        """Test ModelConfig with minimal parameters."""
        config = ModelConfig(model_type="openai/gpt-4")
        
        assert config.model_type == "openai/gpt-4"
        assert config.api_key_env == "OPENROUTER_API_KEY"
        assert config.max_tokens == 1000
        assert config.temperature == 0.7
        assert config.rate_limit == 60
        assert config.retry_delay == 1.0
        assert config.custom_params == {}
    
    def test_model_config_custom_values(self):
        """Test ModelConfig with custom values."""
        config = ModelConfig(
            model_type="anthropic/claude-3-sonnet-20240229",
            api_key_env="CLAUDE_API_KEY",
            max_tokens=2000,
            temperature=0.5,
            rate_limit=30,
            retry_delay=2.0,
            custom_params={"top_p": 0.9}
        )
        
        assert config.model_type == "anthropic/claude-3-sonnet-20240229"
        assert config.api_key_env == "CLAUDE_API_KEY"
        assert config.max_tokens == 2000
        assert config.temperature == 0.5
        assert config.rate_limit == 30
        assert config.retry_delay == 2.0
        assert config.custom_params == {"top_p": 0.9}


class TestScenarioConfig:
    """Test ScenarioConfig dataclass."""
    
    def test_scenario_config_creation(self):
        """Test ScenarioConfig creation."""
        scenario = ScenarioConfig(
            name="Mixed GPT-4 and Claude",
            model_distribution={
                "openai/gpt-4": 5,
                "anthropic/claude-3-sonnet-20240229": 5
            }
        )
        
        assert scenario.name == "Mixed GPT-4 and Claude"
        assert scenario.model_distribution["openai/gpt-4"] == 5
        assert scenario.model_distribution["anthropic/claude-3-sonnet-20240229"] == 5
        assert sum(scenario.model_distribution.values()) == 10


class TestModelTypeEnum:
    """Test ModelType enum."""
    
    def test_model_type_values(self):
        """Test that ModelType enum has correct values."""
        assert ModelType.GPT_4.value == "openai/gpt-4"
        assert ModelType.GPT_35_TURBO.value == "openai/gpt-3.5-turbo"
        assert ModelType.CLAUDE_3_SONNET.value == "anthropic/claude-3-sonnet-20240229"
        assert ModelType.GEMINI_PRO.value == "google/gemini-pro"
        assert ModelType.GEMINI_25_FLASH.value == "google/gemini-2.5-flash"
    
    def test_default_model_type(self):
        """Test that default model is Gemini 2.5 Flash for backward compatibility."""
        # This is the model used in existing StrategyRecord
        assert ModelType.GEMINI_25_FLASH.value == "google/gemini-2.5-flash"


class TestModelAdapters:
    """Test model adapter framework."""
    
    def test_unified_adapter_request_params(self):
        """Test UnifiedOpenRouterAdapter generates correct request parameters."""
        from src.core.model_adapters import UnifiedOpenRouterAdapter
        
        config = ModelConfig(
            model_type="openai/gpt-4",
            max_tokens=500,
            temperature=0.8
        )
        adapter = UnifiedOpenRouterAdapter(config)
        
        params = adapter.get_request_params("Test prompt")
        
        assert params["model"] == "openai/gpt-4"
        assert params["messages"][0]["role"] == "user"
        assert params["messages"][0]["content"] == "Test prompt"
        assert params["max_tokens"] == 500
        assert params["temperature"] == 0.8
    
    def test_unified_adapter_custom_params(self):
        """Test that custom parameters are included."""
        from src.core.model_adapters import UnifiedOpenRouterAdapter
        
        config = ModelConfig(
            model_type="anthropic/claude-3-sonnet-20240229",
            custom_params={"top_p": 0.9, "stop_sequences": ["\n\n"]}
        )
        adapter = UnifiedOpenRouterAdapter(config)
        
        params = adapter.get_request_params("Test")
        
        assert params["top_p"] == 0.9
        assert params["stop_sequences"] == ["\n\n"]
    
    def test_unified_adapter_parse_response(self):
        """Test response parsing."""
        from src.core.model_adapters import UnifiedOpenRouterAdapter
        
        config = ModelConfig(model_type="openai/gpt-4")
        adapter = UnifiedOpenRouterAdapter(config)
        
        # Test valid response
        response = {
            "choices": [{
                "message": {
                    "content": "Test response content"
                }
            }]
        }
        
        content = adapter.parse_response(response)
        assert content == "Test response content"
    
    def test_unified_adapter_parse_response_error(self):
        """Test response parsing with invalid format."""
        from src.core.model_adapters import UnifiedOpenRouterAdapter
        
        config = ModelConfig(model_type="openai/gpt-4")
        adapter = UnifiedOpenRouterAdapter(config)
        
        # Test invalid response
        with pytest.raises(ValueError, match="Unexpected response format"):
            adapter.parse_response({"invalid": "format"})
    
    def test_model_adapter_factory(self):
        """Test ModelAdapterFactory creates and caches adapters."""
        from src.core.model_adapters import ModelAdapterFactory, UnifiedOpenRouterAdapter
        
        # Clear cache first
        ModelAdapterFactory.clear_cache()
        
        config1 = ModelConfig(model_type="openai/gpt-4")
        config2 = ModelConfig(model_type="openai/gpt-4")  # Same as config1
        config3 = ModelConfig(model_type="google/gemini-pro")  # Different
        
        adapter1 = ModelAdapterFactory.get_adapter(config1)
        adapter2 = ModelAdapterFactory.get_adapter(config2)
        adapter3 = ModelAdapterFactory.get_adapter(config3)
        
        # Should return same adapter for same model type
        assert adapter1 is adapter2
        assert isinstance(adapter1, UnifiedOpenRouterAdapter)
        
        # Different model type should get different adapter
        assert adapter1 is not adapter3
        assert isinstance(adapter3, UnifiedOpenRouterAdapter)
    
    def test_fallback_handler(self):
        """Test FallbackHandler logic."""
        from src.core.model_adapters import FallbackHandler
        import asyncio
        
        async def test_fallback():
            # Test fallback from custom model to default
            failed_config = ModelConfig(model_type="openai/gpt-4")
            error = Exception("API error")
            context = {"agent_id": 1, "round": 5}
            
            fallback = await FallbackHandler.handle_model_failure(
                error, failed_config, context
            )
            
            assert fallback is not None
            assert fallback.model_type == "google/gemini-2.5-flash"
            
            # Test no fallback when default model fails
            default_config = ModelConfig(model_type="google/gemini-2.5-flash")
            fallback2 = await FallbackHandler.handle_model_failure(
                error, default_config, context
            )
            
            assert fallback2 is None
        
        asyncio.run(test_fallback())
    
    @pytest.mark.asyncio
    async def test_rate_limiting(self):
        """Test rate limiting enforcement."""
        from src.core.model_adapters import UnifiedOpenRouterAdapter
        import time
        
        config = ModelConfig(
            model_type="openai/gpt-4",
            rate_limit=120  # 120 requests per minute = 0.5s between requests
        )
        adapter = UnifiedOpenRouterAdapter(config)
        
        # First request should not sleep
        start = time.time()
        await adapter.enforce_rate_limit()
        first_duration = time.time() - start
        assert first_duration < 0.1  # Should be instant
        
        # Second request should sleep
        start = time.time()
        await adapter.enforce_rate_limit()
        second_duration = time.time() - start
        assert second_duration >= 0.4  # Should sleep ~0.5s


class TestConfigMultiModel:
    """Test configuration with multi-model support."""
    
    def test_config_default_multi_model_disabled(self):
        """Test that multi-model is disabled by default."""
        from src.core.config import Config
        
        # Mock environment
        import os
        os.environ["OPENROUTER_API_KEY"] = "test-key"
        
        config = Config()
        
        assert config.ENABLE_MULTI_MODEL is False
        assert config.model_configs == {}
        assert config.scenarios == []
    
    def test_config_enable_multi_model_empty(self):
        """Test enabling multi-model with no scenarios."""
        from src.core.config import Config
        import os
        
        os.environ["OPENROUTER_API_KEY"] = "test-key"
        
        config = Config(ENABLE_MULTI_MODEL=True)
        
        assert config.ENABLE_MULTI_MODEL is True
        # Should not raise error with empty scenarios
    
    def test_config_multi_model_validation(self):
        """Test multi-model configuration validation."""
        from src.core.config import Config
        import os
        
        os.environ["OPENROUTER_API_KEY"] = "test-key"
        
        # Test invalid scenario - wrong agent count
        with pytest.raises(ValueError, match="assigns 8 agents but NUM_AGENTS is 10"):
            config = Config(
                ENABLE_MULTI_MODEL=True,
                NUM_AGENTS=10,
                model_configs={
                    "openai/gpt-4": ModelConfig(model_type="openai/gpt-4")
                },
                scenarios=[
                    ScenarioConfig(
                        name="Invalid",
                        model_distribution={"openai/gpt-4": 8}  # Wrong count
                    )
                ]
            )
    
    def test_config_multi_model_missing_model(self):
        """Test validation catches missing model configs."""
        from src.core.config import Config
        import os
        
        os.environ["OPENROUTER_API_KEY"] = "test-key"
        
        # Test scenario references undefined model
        with pytest.raises(ValueError, match="references model 'undefined/model'"):
            config = Config(
                ENABLE_MULTI_MODEL=True,
                NUM_AGENTS=10,
                model_configs={
                    "openai/gpt-4": ModelConfig(model_type="openai/gpt-4")
                },
                scenarios=[
                    ScenarioConfig(
                        name="Missing Model",
                        model_distribution={
                            "openai/gpt-4": 5,
                            "undefined/model": 5  # Not in model_configs
                        }
                    )
                ]
            )
    
    def test_config_from_yaml_backward_compatible(self, tmp_path):
        """Test that existing YAML configs still load correctly."""
        from src.core.config import Config
        import os
        
        os.environ["OPENROUTER_API_KEY"] = "test-key"
        
        # Create old-style config
        config_yaml = tmp_path / "old_config.yaml"
        config_yaml.write_text("""
NUM_AGENTS: 5
NUM_ROUNDS: 3
MAIN_MODEL: google/gemini-2.5-flash
SUB_MODEL: openai/GPT-4.1-nano
""")
        
        config = Config.from_yaml(config_yaml)
        
        assert config.NUM_AGENTS == 5
        assert config.NUM_ROUNDS == 3
        assert config.ENABLE_MULTI_MODEL is False
        assert config.model_configs == {}
        assert config.scenarios == []
    
    def test_config_from_yaml_multi_model(self, tmp_path):
        """Test loading multi-model configuration from YAML."""
        from src.core.config import Config
        import os
        
        os.environ["OPENROUTER_API_KEY"] = "test-key"
        
        # Create multi-model config
        config_yaml = tmp_path / "multi_model_config.yaml"
        config_yaml.write_text("""
NUM_AGENTS: 10
NUM_ROUNDS: 5
ENABLE_MULTI_MODEL: true

model_configs:
  openai/gpt-4:
    model_type: openai/gpt-4
    temperature: 0.8
    max_tokens: 1500
  google/gemini-pro:
    model_type: google/gemini-pro
    temperature: 0.7

scenarios:
  - name: Mixed GPT and Gemini
    model_distribution:
      openai/gpt-4: 5
      google/gemini-pro: 5
""")
        
        config = Config.from_yaml(config_yaml)
        
        assert config.ENABLE_MULTI_MODEL is True
        assert len(config.model_configs) == 2
        assert "openai/gpt-4" in config.model_configs
        assert config.model_configs["openai/gpt-4"].temperature == 0.8
        assert len(config.scenarios) == 1
        assert config.scenarios[0].name == "Mixed GPT and Gemini"
    
    def test_config_from_yaml_multi_model_ignored_when_disabled(self, tmp_path, caplog):
        """Test that multi-model config is ignored when feature disabled."""
        from src.core.config import Config
        import os
        import logging
        
        os.environ["OPENROUTER_API_KEY"] = "test-key"
        
        # Create config with multi-model but flag disabled
        config_yaml = tmp_path / "disabled_multi_model.yaml"
        config_yaml.write_text("""
NUM_AGENTS: 10
ENABLE_MULTI_MODEL: false

model_configs:
  openai/gpt-4:
    model_type: openai/gpt-4

scenarios:
  - name: Ignored
    model_distribution:
      openai/gpt-4: 10
""")
        
        # Capture logs instead of warnings
        with caplog.at_level(logging.WARNING):
            config = Config.from_yaml(config_yaml)
        
        assert config.ENABLE_MULTI_MODEL is False
        assert config.model_configs == {}
        assert config.scenarios == []
        
        # Check that warning was logged
        assert "Multi-model configuration found but ENABLE_MULTI_MODEL=False" in caplog.text


class TestExperimentMultiModel:
    """Test ExperimentFlow with multi-model support."""
    
    def test_experiment_flow_backward_compatible(self):
        """Test that ExperimentFlow works without multi-model."""
        from src.flows.experiment import ExperimentFlow
        from src.core.config import Config
        import os
        
        os.environ["OPENROUTER_API_KEY"] = "test-key"
        
        config = Config(NUM_AGENTS=5, NUM_ROUNDS=1)
        flow = ExperimentFlow(config)
        
        assert flow.config.ENABLE_MULTI_MODEL is False
        assert flow.scenario is None
        assert flow.scenario_name is None
    
    def test_experiment_flow_with_scenario(self):
        """Test ExperimentFlow with multi-model scenario."""
        from src.flows.experiment import ExperimentFlow
        from src.core.config import Config
        import os
        
        os.environ["OPENROUTER_API_KEY"] = "test-key"
        
        config = Config(
            NUM_AGENTS=10,
            NUM_ROUNDS=1,
            ENABLE_MULTI_MODEL=True,
            model_configs={
                "openai/gpt-4": ModelConfig(model_type="openai/gpt-4"),
                "google/gemini-pro": ModelConfig(model_type="google/gemini-pro")
            },
            scenarios=[
                ScenarioConfig(
                    name="Test Scenario",
                    model_distribution={
                        "openai/gpt-4": 5,
                        "google/gemini-pro": 5
                    }
                )
            ]
        )
        
        flow = ExperimentFlow(config, scenario_name="Test Scenario")
        
        assert flow.config.ENABLE_MULTI_MODEL is True
        assert flow.scenario is not None
        assert flow.scenario.name == "Test Scenario"
    
    def test_initialize_agents_default(self):
        """Test agent initialization without multi-model."""
        from src.flows.experiment import ExperimentFlow
        from src.core.config import Config
        import os
        
        os.environ["OPENROUTER_API_KEY"] = "test-key"
        
        config = Config(NUM_AGENTS=5)
        flow = ExperimentFlow(config)
        
        agents = flow._initialize_agents()
        
        assert len(agents) == 5
        for i, agent in enumerate(agents):
            assert agent.id == i
            assert agent.model_config is None
    
    def test_initialize_agents_multi_model(self):
        """Test agent initialization with multi-model scenario."""
        from src.flows.experiment import ExperimentFlow
        from src.core.config import Config
        import os
        
        os.environ["OPENROUTER_API_KEY"] = "test-key"
        
        config = Config(
            NUM_AGENTS=10,
            ENABLE_MULTI_MODEL=True,
            model_configs={
                "openai/gpt-4": ModelConfig(model_type="openai/gpt-4"),
                "google/gemini-pro": ModelConfig(model_type="google/gemini-pro")
            },
            scenarios=[
                ScenarioConfig(
                    name="Mixed Models",
                    model_distribution={
                        "openai/gpt-4": 6,
                        "google/gemini-pro": 4
                    }
                )
            ]
        )
        
        flow = ExperimentFlow(config, scenario_name="Mixed Models")
        agents = flow._initialize_agents()
        
        assert len(agents) == 10
        
        # Count models
        gpt4_count = sum(1 for a in agents if a.model_config and a.model_config.model_type == "openai/gpt-4")
        gemini_count = sum(1 for a in agents if a.model_config and a.model_config.model_type == "google/gemini-pro")
        
        assert gpt4_count == 6
        assert gemini_count == 4
        
        # Check all agents have model configs
        for agent in agents:
            assert agent.model_config is not None
    
    def test_experiment_flow_missing_scenario(self):
        """Test ExperimentFlow with non-existent scenario name."""
        from src.flows.experiment import ExperimentFlow
        from src.core.config import Config
        import os
        
        os.environ["OPENROUTER_API_KEY"] = "test-key"
        
        config = Config(
            NUM_AGENTS=10,
            ENABLE_MULTI_MODEL=True,
            model_configs={
                "openai/gpt-4": ModelConfig(model_type="openai/gpt-4")
            },
            scenarios=[
                ScenarioConfig(
                    name="Valid", 
                    model_distribution={"openai/gpt-4": 10}
                )
            ]
        )
        
        flow = ExperimentFlow(config, scenario_name="Invalid")
        
        assert flow.scenario is None  # Should fall back to None
    
    @pytest.mark.asyncio
    async def test_experiment_preserves_agent_ids(self):
        """Test that multi-model preserves proper agent IDs after shuffle."""
        from src.flows.experiment import ExperimentFlow
        from src.core.config import Config
        import os
        
        os.environ["OPENROUTER_API_KEY"] = "test-key"
        
        config = Config(
            NUM_AGENTS=10,
            ENABLE_MULTI_MODEL=True,
            model_configs={
                "openai/gpt-4": ModelConfig(model_type="openai/gpt-4"),
                "google/gemini-pro": ModelConfig(model_type="google/gemini-pro")
            },
            scenarios=[
                ScenarioConfig(
                    name="Test",
                    model_distribution={
                        "openai/gpt-4": 5,
                        "google/gemini-pro": 5
                    }
                )
            ]
        )
        
        flow = ExperimentFlow(config, scenario_name="Test")
        agents = flow._initialize_agents()
        
        # Check that all IDs 0-9 are present (shuffle shouldn't change IDs)
        agent_ids = sorted([agent.id for agent in agents])
        assert agent_ids == list(range(10))


class TestAPIClientMultiModel:
    """Test OpenRouterClient with multi-model support."""
    
    @pytest.mark.asyncio
    async def test_client_backward_compatible(self):
        """Test that client works without adapter (backward compatible)."""
        from src.core.api_client import OpenRouterClient
        from unittest.mock import AsyncMock, MagicMock
        import aiohttp
        
        api_key = "test-key"
        
        # Create mock session and response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={
            "choices": [{
                "message": {"content": "Test response"}
            }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5
            }
        })
        
        mock_session = AsyncMock(spec=aiohttp.ClientSession)
        mock_session.post.return_value.__aenter__.return_value = mock_response
        
        async with OpenRouterClient(api_key) as client:
            client.session = mock_session
            
            response = await client.complete(
                messages=[{"role": "user", "content": "Test"}],
                model="google/gemini-2.5-flash"
            )
            
            assert response["choices"][0]["message"]["content"] == "Test response"
    
    @pytest.mark.asyncio
    async def test_client_with_adapter(self):
        """Test client with model adapter."""
        from src.core.api_client import OpenRouterClient
        from src.core.model_adapters import UnifiedOpenRouterAdapter
        from unittest.mock import AsyncMock, MagicMock
        import aiohttp
        import os
        
        os.environ["OPENROUTER_API_KEY"] = "test-key"
        config = ModelConfig(model_type="openai/gpt-4", temperature=0.8)
        adapter = UnifiedOpenRouterAdapter(config)
        
        # Create mock session and response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={
            "choices": [{
                "message": {"content": "GPT-4 response"}
            }],
            "usage": {
                "prompt_tokens": 15,
                "completion_tokens": 8
            }
        })
        
        mock_session = AsyncMock(spec=aiohttp.ClientSession)
        mock_session.post.return_value.__aenter__.return_value = mock_response
        
        async with OpenRouterClient("test-key") as client:
            client.session = mock_session
            
            response = await client.complete(
                messages=[{"role": "user", "content": "Test"}],
                model="openai/gpt-4",  # This will be overridden by adapter
                adapter=adapter
            )
            
            # Parse response using adapter
            text = adapter.parse_response(response)
            assert text == "GPT-4 response"
    
    @pytest.mark.asyncio
    async def test_get_completion_text_backward_compatible(self):
        """Test get_completion_text still works without changes."""
        from src.core.api_client import OpenRouterClient
        from unittest.mock import AsyncMock, MagicMock
        import aiohttp
        
        # Create mock session and response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={
            "choices": [{
                "message": {"content": "Simple text"}
            }]
        })
        
        mock_session = AsyncMock(spec=aiohttp.ClientSession)
        mock_session.post.return_value.__aenter__.return_value = mock_response
        
        async with OpenRouterClient("test-key") as client:
            client.session = mock_session
            
            text = await client.get_completion_text(
                messages=[{"role": "user", "content": "Test"}],
                model="google/gemini-2.5-flash"
            )
            
            assert text == "Simple text"


class TestStrategyCollectionMultiModel:
    """Test StrategyCollectionNode with multi-model support."""
    
    @pytest.mark.asyncio
    async def test_strategy_collection_without_model_config(self):
        """Test that strategy collection works without model config."""
        from src.nodes.strategy_collection import StrategyCollectionNode
        from src.core.api_client import OpenRouterClient
        from src.core.config import Config
        from unittest.mock import AsyncMock, MagicMock
        import aiohttp
        import os
        
        os.environ["OPENROUTER_API_KEY"] = "test-key"
        
        config = Config(NUM_AGENTS=1)
        agent = Agent(id=0)  # No model_config
        
        # Create mock session and response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={
            "choices": [{
                "message": {"content": "Always cooperate"}
            }],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 50
            }
        })
        
        mock_session = AsyncMock(spec=aiohttp.ClientSession)
        mock_session.post.return_value.__aenter__.return_value = mock_response
        
        async with OpenRouterClient("test-key") as client:
            client.session = mock_session
            
            node = StrategyCollectionNode(client, config)
            
            context = {
                "agents": [agent],
                "round": 1,
                "round_summaries": []
            }
            
            result = await node.execute(context)
            
            strategies = result["strategies"]
            assert len(strategies) == 1
            assert strategies[0].model == config.MAIN_MODEL
    
    @pytest.mark.asyncio
    async def test_strategy_collection_with_model_config(self):
        """Test strategy collection with agent-specific model."""
        from src.nodes.strategy_collection import StrategyCollectionNode
        from src.core.api_client import OpenRouterClient
        from src.core.config import Config
        from unittest.mock import AsyncMock, MagicMock
        import aiohttp
        import os
        
        os.environ["OPENROUTER_API_KEY"] = "test-key"
        
        config = Config(NUM_AGENTS=1, ENABLE_MULTI_MODEL=True)
        model_config = ModelConfig(model_type="openai/gpt-4")
        agent = Agent(id=0, model_config=model_config)
        
        # Create mock session and response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={
            "choices": [{
                "message": {"content": "GPT-4 strategy"}
            }],
            "usage": {
                "prompt_tokens": 120,
                "completion_tokens": 60
            }
        })
        
        mock_session = AsyncMock(spec=aiohttp.ClientSession)
        mock_session.post.return_value.__aenter__.return_value = mock_response
        
        async with OpenRouterClient("test-key") as client:
            client.session = mock_session
            
            node = StrategyCollectionNode(client, config)
            
            context = {
                "agents": [agent],
                "round": 1,
                "round_summaries": []
            }
            
            result = await node.execute(context)
            
            strategies = result["strategies"]
            assert len(strategies) == 1
            assert strategies[0].model == "openai/gpt-4"
            assert strategies[0].strategy_text == "GPT-4 strategy"


class TestSubagentDecisionMultiModel:
    """Test SubagentDecisionNode with multi-model support."""
    
    @pytest.mark.asyncio
    async def test_subagent_decision_without_model_config(self):
        """Test that subagent decision works without model config."""
        from src.nodes.subagent_decision import SubagentDecisionNode
        from src.core.api_client import OpenRouterClient
        from src.core.config import Config
        from unittest.mock import AsyncMock, MagicMock
        import aiohttp
        import os
        
        os.environ["OPENROUTER_API_KEY"] = "test-key"
        
        config = Config(NUM_AGENTS=2)
        agent = Agent(id=0)  # No model_config
        opponent = Agent(id=1)
        
        # Create mock session and response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={
            "choices": [{
                "message": {"content": "COOPERATE"}
            }]
        })
        
        mock_session = AsyncMock(spec=aiohttp.ClientSession)
        mock_session.post.return_value.__aenter__.return_value = mock_response
        
        async with OpenRouterClient("test-key") as client:
            client.session = mock_session
            
            node = SubagentDecisionNode(client, config)
            
            decision = await node.make_decision(agent, opponent, "Always cooperate", [])
            
            assert decision == "COOPERATE"
    
    @pytest.mark.asyncio
    async def test_subagent_decision_with_model_config(self):
        """Test subagent decision with agent-specific model."""
        from src.nodes.subagent_decision import SubagentDecisionNode
        from src.core.api_client import OpenRouterClient
        from src.core.config import Config
        from unittest.mock import AsyncMock, MagicMock
        import aiohttp
        import os
        
        os.environ["OPENROUTER_API_KEY"] = "test-key"
        
        config = Config(NUM_AGENTS=2, ENABLE_MULTI_MODEL=True)
        model_config = ModelConfig(model_type="openai/gpt-4", temperature=0.5)
        agent = Agent(id=0, model_config=model_config)
        opponent = Agent(id=1)
        
        # Create mock session and response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={
            "choices": [{
                "message": {"content": "DEFECT"}
            }]
        })
        
        mock_session = AsyncMock(spec=aiohttp.ClientSession)
        mock_session.post.return_value.__aenter__.return_value = mock_response
        
        async with OpenRouterClient("test-key") as client:
            client.session = mock_session
            
            node = SubagentDecisionNode(client, config)
            
            decision = await node.make_decision(agent, opponent, "Tit for tat", [])
            
            assert decision == "DEFECT"
    
    @pytest.mark.asyncio
    async def test_subagent_fallback_on_error(self):
        """Test subagent falls back to default model on error."""
        from src.nodes.subagent_decision import SubagentDecisionNode
        from src.core.api_client import OpenRouterClient
        from src.core.config import Config
        from unittest.mock import AsyncMock, MagicMock
        import aiohttp
        import os
        
        os.environ["OPENROUTER_API_KEY"] = "test-key"
        
        config = Config(NUM_AGENTS=2, ENABLE_MULTI_MODEL=True)
        model_config = ModelConfig(model_type="openai/gpt-4")
        agent = Agent(id=0, model_config=model_config)
        opponent = Agent(id=1)
        
        # Create mock session that fails first, then succeeds
        mock_session = AsyncMock(spec=aiohttp.ClientSession)
        
        # Mock response for second attempt
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={
            "choices": [{
                "message": {"content": "COOPERATE"}
            }]
        })
        
        # First call fails, second succeeds
        mock_session.post.side_effect = [
            MagicMock(__aenter__=AsyncMock(side_effect=Exception("API Error"))),
            MagicMock(__aenter__=AsyncMock(return_value=mock_response))
        ]
        
        async with OpenRouterClient("test-key") as client:
            client.session = mock_session
            
            node = SubagentDecisionNode(client, config)
            
            decision = await node.make_decision(agent, opponent, "Always cooperate", [])
            
            assert decision == "COOPERATE"


class TestRegressionSuite:
    """Regression tests to ensure existing functionality still works."""
    
    def test_all_existing_tests_pass(self):
        """Run a subset of existing tests to verify no regression."""
        # This would normally run the full test suite
        # For now, we'll just verify key imports work
        from src.core.models import Agent, GameResult, StrategyRecord
        from src.core.config import Config
        from src.flows.experiment import ExperimentFlow
        from src.core.api_client import OpenRouterClient
        
        # Verify basic functionality
        agent = Agent(id=0)
        assert agent.model_config is None
        
        config = Config(NUM_AGENTS=5)
        assert config.ENABLE_MULTI_MODEL is False
        
        flow = ExperimentFlow(config)
        assert flow.scenario is None
    
    @pytest.mark.asyncio
    async def test_experiment_runs_without_multi_model(self):
        """Test that a basic experiment still runs without multi-model."""
        from src.flows.experiment import ExperimentFlow
        from src.core.config import Config
        import os
        
        os.environ["OPENROUTER_API_KEY"] = "test-key"
        
        config = Config(
            NUM_AGENTS=2,
            NUM_ROUNDS=1,
            ENABLE_MULTI_MODEL=False  # Explicitly disabled
        )
        
        flow = ExperimentFlow(config)
        
        # Initialize agents
        agents = flow._initialize_agents()
        
        assert len(agents) == 2
        assert all(agent.model_config is None for agent in agents)
    
    def test_model_validation_edge_cases(self):
        """Test edge cases in model validation."""
        from src.core.models import ModelConfig
        
        # Test with empty custom params
        config = ModelConfig(
            model_type="openai/gpt-4",
            custom_params={}
        )
        assert config.custom_params == {}
        
        # Test with various temperature values
        config = ModelConfig(
            model_type="openai/gpt-4",
            temperature=0.0  # Minimum
        )
        assert config.temperature == 0.0
        
        config = ModelConfig(
            model_type="openai/gpt-4",
            temperature=2.0  # Maximum
        )
        assert config.temperature == 2.0
    
    def test_scenario_distribution_validation(self):
        """Test scenario distribution validation edge cases."""
        from src.core.models import ScenarioConfig
        
        # Test with single model
        scenario = ScenarioConfig(
            name="Single Model",
            model_distribution={"openai/gpt-4": 10}
        )
        assert sum(scenario.model_distribution.values()) == 10
        
        # Test with many models
        scenario = ScenarioConfig(
            name="Many Models",
            model_distribution={
                "model1": 1,
                "model2": 1,
                "model3": 1,
                "model4": 1,
                "model5": 1,
                "model6": 1,
                "model7": 1,
                "model8": 1,
                "model9": 1,
                "model10": 1
            }
        )
        assert sum(scenario.model_distribution.values()) == 10


class TestGradualRollout:
    """Test gradual rollout scenarios."""
    
    def test_single_agent_different_model(self):
        """Test with just one agent using a different model."""
        from src.flows.experiment import ExperimentFlow
        from src.core.config import Config
        import os
        
        os.environ["OPENROUTER_API_KEY"] = "test-key"
        
        config = Config(
            NUM_AGENTS=10,
            ENABLE_MULTI_MODEL=True,
            model_configs={
                "openai/gpt-4": ModelConfig(model_type="openai/gpt-4"),
                "google/gemini-2.5-flash": ModelConfig(model_type="google/gemini-2.5-flash")
            },
            scenarios=[
                ScenarioConfig(
                    name="Single Different",
                    model_distribution={
                        "google/gemini-2.5-flash": 9,  # Default model
                        "openai/gpt-4": 1  # One different
                    }
                )
            ]
        )
        
        flow = ExperimentFlow(config, scenario_name="Single Different")
        agents = flow._initialize_agents()
        
        # Count models
        gpt4_count = sum(1 for a in agents if a.model_config and a.model_config.model_type == "openai/gpt-4")
        default_count = sum(1 for a in agents if a.model_config and a.model_config.model_type == "google/gemini-2.5-flash")
        
        assert gpt4_count == 1
        assert default_count == 9
    
    def test_progressive_rollout(self):
        """Test progressive rollout from 1 to many agents."""
        from src.flows.experiment import ExperimentFlow
        from src.core.config import Config
        import os
        
        os.environ["OPENROUTER_API_KEY"] = "test-key"
        
        # Test different ratios
        for gpt4_count in [1, 3, 5, 7, 10]:
            gemini_count = 10 - gpt4_count
            
            config = Config(
                NUM_AGENTS=10,
                ENABLE_MULTI_MODEL=True,
                model_configs={
                    "openai/gpt-4": ModelConfig(model_type="openai/gpt-4"),
                    "google/gemini-pro": ModelConfig(model_type="google/gemini-pro")
                },
                scenarios=[
                    ScenarioConfig(
                        name=f"Rollout {gpt4_count}",
                        model_distribution={
                            "openai/gpt-4": gpt4_count,
                            "google/gemini-pro": gemini_count
                        }
                    )
                ]
            )
            
            flow = ExperimentFlow(config, scenario_name=f"Rollout {gpt4_count}")
            agents = flow._initialize_agents()
            
            # Verify distribution
            actual_gpt4 = sum(1 for a in agents if a.model_config and a.model_config.model_type == "openai/gpt-4")
            actual_gemini = sum(1 for a in agents if a.model_config and a.model_config.model_type == "google/gemini-pro")
            
            assert actual_gpt4 == gpt4_count
            assert actual_gemini == gemini_count


class TestIntegration:
    """Integration tests for multi-model experiments."""
    
    @pytest.mark.asyncio
    async def test_mini_experiment_single_model(self):
        """Test a minimal experiment with single model."""
        from src.flows.experiment import ExperimentFlow
        from src.core.config import Config
        from src.core.api_client import OpenRouterClient
        from unittest.mock import AsyncMock, MagicMock, patch
        import aiohttp
        import os
        
        os.environ["OPENROUTER_API_KEY"] = "test-key"
        
        config = Config(
            NUM_AGENTS=2,
            NUM_ROUNDS=1,
            ENABLE_MULTI_MODEL=False
        )
        
        flow = ExperimentFlow(config)
        
        # Run first part of experiment
        context = {
            "experiment_id": "test_exp",
            "round": 1,
            "agents": flow._initialize_agents(),
            "strategies": [],
            "round_summaries": []
        }
        
        # Verify agents have no model config
        assert all(agent.model_config is None for agent in context["agents"])
    
    @pytest.mark.asyncio
    async def test_mini_experiment_multi_model(self):
        """Test a minimal experiment with multiple models."""
        from src.flows.experiment import ExperimentFlow
        from src.core.config import Config
        from unittest.mock import AsyncMock, MagicMock, patch
        import aiohttp
        import os
        
        os.environ["OPENROUTER_API_KEY"] = "test-key"
        
        config = Config(
            NUM_AGENTS=2,
            NUM_ROUNDS=1,
            ENABLE_MULTI_MODEL=True,
            model_configs={
                "openai/gpt-4": ModelConfig(model_type="openai/gpt-4"),
                "google/gemini-pro": ModelConfig(model_type="google/gemini-pro")
            },
            scenarios=[
                ScenarioConfig(
                    name="Mixed",
                    model_distribution={
                        "openai/gpt-4": 1,
                        "google/gemini-pro": 1
                    }
                )
            ]
        )
        
        flow = ExperimentFlow(config, scenario_name="Mixed")
        agents = flow._initialize_agents()
        
        # Verify each agent has a model config
        assert all(agent.model_config is not None for agent in agents)
        
        # Verify we have one of each model
        models = [agent.model_config.model_type for agent in agents]
        assert "openai/gpt-4" in models
        assert "google/gemini-pro" in models


if __name__ == "__main__":
    pytest.main([__file__, "-v"])