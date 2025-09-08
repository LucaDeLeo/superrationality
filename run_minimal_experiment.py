#!/usr/bin/env python3
"""Minimal experiment run with reduced scope for testing."""

import asyncio
import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Override config for minimal test
os.environ['NUM_AGENTS'] = '2'  # Just 2 agents for 1 game per round
os.environ['NUM_ROUNDS'] = '2'  # Just 2 rounds

from run_experiment import main

if __name__ == "__main__":
    print("🚀 Running MINIMAL ULTRATHINK experiment (2 agents, 2 rounds)...")
    print("This will make ~8 API calls total")
    sys.exit(asyncio.run(main()))