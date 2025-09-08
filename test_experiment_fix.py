#!/usr/bin/env python3
"""Test script to verify experiment fixes."""

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Check for API key
api_key = os.getenv("OPENROUTER_API_KEY")
if not api_key:
    print("ERROR: OPENROUTER_API_KEY not set in environment")
    sys.exit(1)

print(f"✓ API key found: {api_key[:10]}...")

# Test imports
try:
    from src.core.config import Config
    from src.core.models import ExperimentResult, StrategyRecord, GameResult, RoundSummary, Agent
    from src.utils.data_manager import DataManager
    from src.flows.experiment import ExperimentFlow, RoundFlow
    from src.nodes import ContextKeys
    from src.core.api_client import OpenRouterClient
    from src.utils.game_logic import update_powers
    from src.core.scenario_manager import ScenarioManager
    print("✓ All imports successful")
except ImportError as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)

# Test config initialization
try:
    config = Config()
    print(f"✓ Config initialized")
    print(f"  - NUM_AGENTS: {config.NUM_AGENTS}")
    print(f"  - NUM_ROUNDS: {config.NUM_ROUNDS}")
    print(f"  - ENABLE_MULTI_MODEL: {config.ENABLE_MULTI_MODEL}")
    print(f"  - scenarios: {config.scenarios}")
except Exception as e:
    print(f"✗ Config error: {e}")
    sys.exit(1)

# Test agent creation
try:
    agents = [Agent(id=i) for i in range(config.NUM_AGENTS)]
    print(f"✓ Created {len(agents)} agents")
except Exception as e:
    print(f"✗ Agent creation error: {e}")
    sys.exit(1)

# Test StrategyRecord with proper attributes
try:
    strategy = StrategyRecord(
        strategy_id="test_001",
        agent_id=0,
        round=1,
        strategy_text="Always cooperate",
        full_reasoning="Testing strategy creation"
    )
    print(f"✓ StrategyRecord created successfully")
    print(f"  - strategy_text: {strategy.strategy_text}")
    print(f"  - full_reasoning: {strategy.full_reasoning[:30]}...")
    
    # Test that we can access these attributes
    _ = strategy.strategy_text
    _ = strategy.full_reasoning
    print(f"✓ StrategyRecord attributes accessible")
except Exception as e:
    print(f"✗ StrategyRecord error: {e}")
    sys.exit(1)

# Test linking strategy to agent
try:
    agents[0].strategy = strategy.strategy_text
    print(f"✓ Strategy linked to agent")
    print(f"  - Agent 0 strategy: {agents[0].strategy}")
except Exception as e:
    print(f"✗ Strategy linking error: {e}")
    sys.exit(1)

print("\n✅ All tests passed! The experiment should be ready to run.")
print("\nTo run the experiment:")
print("  uv run python run_experiment.py")
print("\nTo run with rate limiting:")
print("  uv run python run_experiment_with_rate_limit.py")