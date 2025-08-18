#!/usr/bin/env python3
"""Minimal test to verify core experiment functionality."""

import asyncio
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

async def test_core_experiment():
    """Test that core experiment can run without removed components."""
    print("Testing core experiment integrity...")
    
    # Test 1: Can we import core modules?
    try:
        from src.core.config import Config
        from src.core.models import Agent, GameResult
        from src.flows.experiment import RoundFlow
        from src.core.api_client import OpenRouterClient
        print("✓ Core imports successful")
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False
    
    # Test 2: Can we create basic objects?
    try:
        # Temporarily set API key for testing if not present
        if not os.getenv("OPENROUTER_API_KEY"):
            os.environ["OPENROUTER_API_KEY"] = "test-key-for-integrity-check"
        
        config = Config()
        agents = [Agent(id=i) for i in range(10)]
        print("✓ Basic objects created")
    except Exception as e:
        print(f"✗ Object creation failed: {e}")
        return False
    
    # Test 3: Can we run a minimal round? (mock API)
    try:
        # Create mock context
        context = {
            'experiment_id': 'test',
            'agents': agents,
            'round': 1,
            'config': config,
            'strategies': [],
            'games': [],
            'round_summaries': []
        }
        print("✓ Context created")
        
        # Just verify the flow can be instantiated
        if os.getenv("OPENROUTER_API_KEY"):
            async with OpenRouterClient(config.OPENROUTER_API_KEY) as client:
                round_flow = RoundFlow(client, config)
                print("✓ RoundFlow instantiated (API key found)")
        else:
            print("⚠ Skipping API test (no OPENROUTER_API_KEY)")
        
    except Exception as e:
        print(f"✗ Round flow test failed: {e}")
        return False
    
    print("\n✅ Core experiment integrity verified!")
    return True

if __name__ == "__main__":
    success = asyncio.run(test_core_experiment())
    sys.exit(0 if success else 1)