#!/usr/bin/env python3
"""Run experiment with proper rate limiting for free tier."""

import asyncio
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set rate limit for free tier
os.environ['RATE_LIMIT_PER_MINUTE'] = '3'  # Stay under 4 req/min limit

# Import and run the main experiment
from run_experiment import main

if __name__ == "__main__":
    print("Running experiment with rate limiting (3 requests per minute)...")
    print("This will take longer but stay within free tier limits.")
    asyncio.run(main())