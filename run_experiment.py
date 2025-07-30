#!/usr/bin/env python3
"""Main script to run the acausal cooperation experiment."""

import asyncio
import json
import logging
import signal
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List
import os

from src.core.config import Config
from src.core.models import ExperimentResult, StrategyRecord, GameResult, RoundSummary, Agent
from src.utils.data_manager import DataManager
from src.flows.experiment import ExperimentFlow, RoundFlow
from src.nodes import ContextKeys
from src.core.api_client import OpenRouterClient
from src.utils.game_logic import update_powers


class RateLimiter:
    """Simple rate limiter for API calls."""
    
    def __init__(self, max_calls: int = 60, window_seconds: int = 60):
        """Initialize rate limiter.
        
        Args:
            max_calls: Maximum calls allowed in window
            window_seconds: Time window in seconds
        """
        self.max_calls = max_calls
        self.window = timedelta(seconds=window_seconds)
        self.calls: List[datetime] = []
    
    async def acquire(self):
        """Wait if necessary to respect rate limits."""
        now = datetime.now()
        # Remove calls outside the current window
        self.calls = [call_time for call_time in self.calls
                     if now - call_time < self.window]
        
        if len(self.calls) >= self.max_calls:
            # Wait until the oldest call expires
            sleep_time = (self.calls[0] + self.window - now).total_seconds()
            if sleep_time > 0:
                logging.info(f"Rate limit reached, waiting {sleep_time:.1f}s")
                await asyncio.sleep(sleep_time)
                return await self.acquire()
        
        self.calls.append(now)


class ExperimentRunner:
    """Handles experiment execution with progress tracking and error recovery."""
    
    def __init__(self):
        """Initialize experiment runner."""
        self.config = Config()
        self.data_manager = DataManager()
        self.rate_limiter = RateLimiter()
        self.round_times: List[float] = []
        self.shutdown_requested = False
        self.current_context: Optional[dict] = None
        
        # Set up signal handlers
        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)
        
    def _handle_shutdown(self, signum, frame):
        """Handle graceful shutdown on Ctrl+C."""
        logging.warning("\n🛑 Shutdown requested. Saving partial results...")
        self.shutdown_requested = True
        
    def _setup_logging(self):
        """Configure logging for console and file output."""
        # Create log directory
        log_dir = self.data_manager.experiment_path / "logs"
        log_dir.mkdir(exist_ok=True)
        
        # Configure logging format
        log_format = '%(asctime)s - %(levelname)s - %(message)s'
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(logging.Formatter(log_format))
        
        # File handler
        file_handler = logging.FileHandler(log_dir / "experiment.log")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(log_format))
        
        # Configure root logger
        logging.root.setLevel(logging.DEBUG)
        logging.root.handlers = [console_handler, file_handler]
        
    def _calculate_time_remaining(self, current_round: int) -> float:
        """Calculate estimated time remaining based on moving average.
        
        Args:
            current_round: Current round number
            
        Returns:
            Estimated seconds remaining
        """
        if not self.round_times:
            return 0.0
            
        # Use last 3 rounds for moving average
        recent_times = self.round_times[-3:]
        avg_round_time = sum(recent_times) / len(recent_times)
        rounds_remaining = self.config.NUM_ROUNDS - current_round
        
        return avg_round_time * rounds_remaining
        
    def _format_time(self, seconds: float) -> str:
        """Format seconds into human-readable time.
        
        Args:
            seconds: Time in seconds
            
        Returns:
            Formatted time string
        """
        if seconds < 60:
            return f"{int(seconds)}s"
        elif seconds < 3600:
            minutes = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{minutes}m {secs}s"
        else:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            return f"{hours}h {minutes}m"
            
    def _display_progress(self, round_num: int, round_summary: Optional[RoundSummary] = None):
        """Display progress information to console.
        
        Args:
            round_num: Current round number
            round_summary: Optional round summary for additional stats
        """
        progress = round_num / self.config.NUM_ROUNDS
        bar_length = 30
        filled = int(bar_length * progress)
        bar = '█' * filled + '░' * (bar_length - filled)
        
        msg = f"\n📊 Round {round_num}/{self.config.NUM_ROUNDS} [{bar}] {progress*100:.0f}%"
        
        if round_summary:
            msg += f"\n   🤝 Cooperation Rate: {round_summary.cooperation_rate:.1%}"
            msg += f"\n   📈 Average Score: {round_summary.average_score:.1f}"
            
        if self.round_times:
            time_remaining = self._calculate_time_remaining(round_num)
            msg += f"\n   ⏱️  Estimated time remaining: {self._format_time(time_remaining)}"
            
        logging.info(msg)
        
    async def run_experiment(self) -> ExperimentResult:
        """Run the complete experiment with progress tracking and error recovery.
        
        Returns:
            Complete experiment results
        """
        self._setup_logging()
        
        logging.info(f"\n🚀 Starting Acausal Cooperation Experiment")
        logging.info(f"📁 Results will be saved to: {self.data_manager.experiment_path}")
        
        # Create custom ExperimentFlow that integrates with our DataManager
        experiment_flow = ExperimentFlow(self.config)
        experiment_flow.experiment_id = self.data_manager.experiment_id
        
        result = None
        
        try:
            # Run experiment with monitoring
            result = await self._run_with_monitoring(experiment_flow)
            
        except Exception as e:
            logging.error(f"❌ Experiment failed: {str(e)}", exc_info=True)
            self.data_manager.save_error_log(
                error_type="ExperimentFailure",
                error_msg=str(e),
                context={"round": self.current_context.get(ContextKeys.ROUND) if self.current_context else 0}
            )
            
            # Save partial results
            if self.current_context:
                await self._save_partial_results()
                
            raise
            
        finally:
            if result:
                # Save final results
                self.data_manager.save_experiment_result(result)
                logging.info(f"\n✅ Experiment completed successfully!")
                logging.info(f"📊 Total rounds: {result.total_rounds}")
                logging.info(f"🎮 Total games: {result.total_games}")
                logging.info(f"💰 Estimated cost: ${result.total_cost:.2f}")
                logging.info(f"📁 Results saved to: {self.data_manager.experiment_path}")
                
        return result
        
    async def _run_with_monitoring(self, experiment_flow: ExperimentFlow) -> ExperimentResult:
        """Run experiment with progress monitoring and data persistence.
        
        Args:
            experiment_flow: Experiment flow to execute
            
        Returns:
            Complete experiment results
        """
        logging.info(f"Starting experiment {self.data_manager.experiment_id}")
        
        # Initialize result
        result = ExperimentResult(
            experiment_id=self.data_manager.experiment_id,
            start_time=datetime.now().isoformat()
        )
        
        # Initialize agents
        agents = [Agent(id=i) for i in range(self.config.NUM_AGENTS)]
        
        # Initialize context
        context = {
            ContextKeys.EXPERIMENT_ID: self.data_manager.experiment_id,
            ContextKeys.AGENTS: agents,
            ContextKeys.ROUND_SUMMARIES: [],
            ContextKeys.CONFIG: self.config,
            'data_manager': self.data_manager  # Add data manager to context
        }
        
        self.current_context = context
        
        # Run experiment with API client
        async with OpenRouterClient(self.config.OPENROUTER_API_KEY) as api_client:
            # Create round flow with rate limiter
            round_flow = RoundFlow(api_client, self.config, self.rate_limiter)
            
            # Run all rounds
            for round_num in range(1, self.config.NUM_ROUNDS + 1):
                if self.shutdown_requested:
                    logging.warning(f"Shutdown requested at round {round_num}")
                    break
                    
                round_start = datetime.now()
                logging.info(f"Starting round {round_num}")
                context[ContextKeys.ROUND] = round_num
                
                try:
                    # Apply rate limiting
                    await self.rate_limiter.acquire()
                    
                    # Run round
                    context = await round_flow.run(context)
                    
                    # Get round data
                    strategies = context.get(ContextKeys.STRATEGIES, [])
                    games = context.get(ContextKeys.GAMES, [])
                    round_summaries = context.get(ContextKeys.ROUND_SUMMARIES, [])
                    
                    # Save round data
                    if strategies:
                        self.data_manager.save_strategies(round_num, strategies)
                    if games:
                        self.data_manager.save_games(round_num, games)
                    if round_summaries and round_summaries[-1].round == round_num:
                        self.data_manager.save_round_summary(round_summaries[-1])
                    
                    # Update powers after round
                    update_powers(agents, games)
                    
                    # Update experiment stats
                    result.total_rounds = round_num
                    result.total_games += len(games)
                    result.total_api_calls += len(agents) + (len(games) * 2)  # strategies + decisions
                    
                    # Track round time
                    round_time = (datetime.now() - round_start).total_seconds()
                    self.round_times.append(round_time)
                    
                    # Display progress
                    if round_summaries and round_summaries[-1].round == round_num:
                        self._display_progress(round_num, round_summaries[-1])
                    else:
                        self._display_progress(round_num)
                    
                    logging.info(f"Completed round {round_num}")
                    
                except Exception as e:
                    logging.error(f"Error in round {round_num}: {str(e)}")
                    self.data_manager.save_error_log(
                        error_type="RoundError",
                        error_msg=str(e),
                        context={"round": round_num, "agents": len(agents)}
                    )
                    # Continue to next round if possible
                    if not isinstance(e, (KeyboardInterrupt, SystemExit)):
                        continue
                    else:
                        raise
        
        # Finalize experiment result
        result.end_time = datetime.now().isoformat()
        result.round_summaries = context[ContextKeys.ROUND_SUMMARIES]
        
        # Calculate acausal indicators
        result.acausal_indicators = {
            "identity_reasoning_frequency": 0.0,  # Would analyze strategy texts
            "cooperation_despite_asymmetry": self._calculate_asymmetric_cooperation(result),
            "surprise_at_defection": 0.0,  # Would analyze strategy reasoning
            "strategy_convergence": self._calculate_strategy_convergence(result)
        }
        
        # Estimate cost (rough estimates based on token usage)
        # Assuming ~1000 tokens per API call at $0.00015 per 1K tokens for Gemini Flash
        result.total_cost = result.total_api_calls * 0.00015
        
        return result
        
    async def _save_partial_results(self):
        """Save partial results in case of failure."""
        if not self.current_context:
            return
            
        round_num = self.current_context.get(ContextKeys.ROUND, 0)
        summaries = self.current_context.get(ContextKeys.ROUND_SUMMARIES, [])
        
        partial_data = {
            "last_round": round_num,
            "round_summaries": [vars(s) for s in summaries],
            "context_keys": list(self.current_context.keys())
        }
        
        self.data_manager.save_partial_results(round_num, partial_data)
        logging.info(f"💾 Partial results saved up to round {round_num}")
    
    def _calculate_asymmetric_cooperation(self, result: ExperimentResult) -> float:
        """Calculate cooperation rate despite power asymmetry.
        
        Args:
            result: Experiment results
            
        Returns:
            Cooperation rate in asymmetric games
        """
        asymmetric_cooperations = 0
        asymmetric_games = 0
        
        for summary in result.round_summaries:
            for game in summary.anonymized_games:
                power_diff = abs(game.agent1_power - game.agent2_power)
                if power_diff > 0:  # Asymmetric game
                    asymmetric_games += 1
                    if game.agent1_action == "COOPERATE" or game.agent2_action == "COOPERATE":
                        asymmetric_cooperations += 1
        
        return asymmetric_cooperations / asymmetric_games if asymmetric_games > 0 else 0.0
    
    def _calculate_strategy_convergence(self, result: ExperimentResult) -> float:
        """Calculate strategy convergence over rounds.
        
        Args:
            result: Experiment results
            
        Returns:
            Strategy convergence metric
        """
        if len(result.round_summaries) < 2:
            return 0.0
        
        convergence_scores = []
        for i in range(1, len(result.round_summaries)):
            prev = result.round_summaries[i-1]
            curr = result.round_summaries[i]
            # Use cooperation rate change as proxy for convergence
            rate_change = abs(curr.cooperation_rate - prev.cooperation_rate)
            convergence_scores.append(1.0 - rate_change)  # Higher score = more convergence
        
        return sum(convergence_scores) / len(convergence_scores)


async def main():
    """Main entry point for the experiment."""
    runner = ExperimentRunner()
    
    try:
        result = await runner.run_experiment()
        return 0
    except KeyboardInterrupt:
        logging.warning("\n⚠️  Experiment interrupted by user")
        return 1
    except Exception as e:
        logging.error(f"\n❌ Experiment failed: {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))