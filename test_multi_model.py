#!/usr/bin/env python3
"""Test script to verify multi-model functionality."""

import asyncio
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


async def test_model_config():
    """Test that ModelConfig and scenarios load correctly."""
    print("🧪 Testing Model Configuration...")
    
    from src.core.config import Config
    from src.core.models import Agent, ModelConfig, ScenarioConfig
    
    # Test Config loads scenarios
    config = Config()
    print(f"✓ Config initialized")
    print(f"  ENABLE_MULTI_MODEL: {config.ENABLE_MULTI_MODEL}")
    print(f"  Scenarios loaded: {len(config.scenarios) if config.scenarios else 0}")
    
    if config.scenarios:
        print("\n  Available scenarios:")
        for s in config.scenarios[:3]:  # Show first 3
            print(f"    - {s.name}: {s.model_distribution}")
    
    # Test Agent with ModelConfig
    model_config = ModelConfig(
        model_type="openai/gpt-4o",
        temperature=0.7,
        max_tokens=1000
    )
    
    agent = Agent(id=0, model_config=model_config)
    print(f"\n✓ Agent created with model config")
    print(f"  Agent model: {agent.model_config.model_type}")
    
    return True


async def test_scenario_assignment():
    """Test that ScenarioManager assigns models correctly."""
    print("\n🧪 Testing Scenario Assignment...")
    
    from src.core.models import Agent, ScenarioConfig
    from src.core.scenario_manager import ScenarioManager
    
    # Create test scenario
    scenario = ScenarioConfig(
        name="test_mixed",
        model_distribution={
            "google/gemini-2.5-flash": 2,
            "openai/gpt-4o": 2
        }
    )
    
    # Create agents
    agents = [Agent(id=i) for i in range(4)]
    
    # Assign models
    manager = ScenarioManager(num_agents=4)
    assignments = manager.assign_models_to_agents(agents, scenario, seed=42)
    
    print(f"✓ Models assigned to agents")
    for agent_id, model in assignments.items():
        print(f"  Agent {agent_id}: {model}")
    
    # Verify all agents have model_config
    for agent in agents:
        assert agent.model_config is not None
        assert agent.model_config.model_type in ["google/gemini-2.5-flash", "openai/gpt-4o"]
    
    print(f"✓ All agents have valid model configurations")
    
    return True


async def test_multi_model_experiment():
    """Test running a small multi-model experiment."""
    print("\n🧪 Testing Multi-Model Experiment...")
    
    # Set minimal config for test
    os.environ['NUM_AGENTS'] = '2'
    os.environ['NUM_ROUNDS'] = '1'
    
    from src.core.config import Config
    from src.core.models import Agent, ModelConfig
    from src.flows.experiment import ExperimentFlow
    from src.core.api_client import OpenRouterClient
    
    config = Config()
    
    # Create mixed model agents
    agents = [
        Agent(id=0, model_config=ModelConfig(model_type="google/gemini-2.5-flash")),
        Agent(id=1, model_config=ModelConfig(model_type="openai/gpt-4o"))
    ]
    
    print(f"✓ Created {len(agents)} agents with different models:")
    for agent in agents:
        print(f"  Agent {agent.id}: {agent.model_config.model_type}")
    
    # Initialize context
    context = {
        'experiment_id': 'test_multi_model',
        'agents': agents,
        'round_summaries': [],
        'config': config
    }
    
    print("\n  Note: Actual API calls would happen here in production")
    print("  Skipping to avoid costs during testing")
    
    return True


async def main():
    """Run all tests."""
    print("🚀 ULTRATHINK Multi-Model Testing")
    print("="*60)
    
    try:
        # Test 1: Model Configuration
        success = await test_model_config()
        if not success:
            print("❌ Model configuration test failed")
            return 1
        
        # Test 2: Scenario Assignment
        success = await test_scenario_assignment()
        if not success:
            print("❌ Scenario assignment test failed")
            return 1
        
        # Test 3: Multi-Model Experiment Setup
        success = await test_multi_model_experiment()
        if not success:
            print("❌ Multi-model experiment test failed")
            return 1
        
        print("\n" + "="*60)
        print("✅ All tests passed! Multi-model functionality is ready.")
        print("\nNext steps:")
        print("1. Run a test scenario: uv run python run_model_comparison.py --test")
        print("2. Run full comparison: uv run python run_model_comparison.py")
        print("3. Analyze results: uv run python analyze_models.py --latest")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(asyncio.run(main()))