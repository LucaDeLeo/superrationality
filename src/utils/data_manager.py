"""Data persistence management for experiment results."""
import json
import os
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Any, Dict
from dataclasses import asdict
import tempfile
import shutil

from src.core.models import StrategyRecord, GameResult, RoundSummary, ExperimentResult


class DataManager:
    """Handles all file I/O operations for experiment data."""
    
    def __init__(self, base_path: str = "results"):
        """Initialize DataManager with experiment directory structure.
        
        Args:
            base_path: Base directory for all experiment results
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(exist_ok=True)
        self.experiment_id = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.experiment_path = self.base_path / self.experiment_id
        self._setup_directories()
        
    def _setup_directories(self):
        """Create experiment directory structure."""
        self.experiment_path.mkdir(exist_ok=True)
        (self.experiment_path / "rounds").mkdir(exist_ok=True)
        
    def _write_json(self, path: Path, data: Any):
        """Write JSON data atomically to prevent corruption.
        
        Args:
            path: Path to write JSON file
            data: Data to serialize as JSON
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write to temporary file first
        with tempfile.NamedTemporaryFile(
            mode='w', 
            dir=path.parent, 
            delete=False,
            suffix='.tmp'
        ) as tmp_file:
            json.dump(data, tmp_file, indent=2, default=str)
            tmp_path = Path(tmp_file.name)
        
        # Atomic rename to target path
        shutil.move(str(tmp_path), str(path))
        
    def save_strategies(self, round_num: int, strategies: List[StrategyRecord]):
        """Save strategy records for a specific round.
        
        Args:
            round_num: Round number (1-10)
            strategies: List of strategy records from agents
        """
        path = self.experiment_path / "rounds" / f"strategies_r{round_num}.json"
        
        # Convert to Epic 2 format
        formatted_strategies = []
        for s in strategies:
            # Map fields to Epic 2 specification
            formatted_strategy = {
                "round": s.round,
                "agent_id": s.agent_id,
                "timestamp": s.timestamp,
                "model": s.model,
                "strategy": s.strategy_text,  # Map strategy_text to strategy
                "full_reasoning": s.full_reasoning,
                "prompt_tokens": s.prompt_tokens,
                "completion_tokens": s.completion_tokens
            }
            formatted_strategies.append(formatted_strategy)
        
        data = {
            "round": round_num,
            "timestamp": datetime.now().isoformat(),
            "strategies": formatted_strategies
        }
        self._write_json(path, data)
        
    def save_games(self, round_num: int, games: List[GameResult]):
        """Save game results for a specific round.
        
        Args:
            round_num: Round number (1-10)
            games: List of game results from the round
        """
        path = self.experiment_path / "rounds" / f"games_r{round_num}.json"
        data = {
            "round": round_num,
            "timestamp": datetime.now().isoformat(),
            "games": [asdict(g) for g in games]
        }
        self._write_json(path, data)
        
    def save_round_summary(self, round_summary: RoundSummary):
        """Save round summary data.
        
        Args:
            round_summary: Summary statistics for the round
        """
        path = self.experiment_path / "rounds" / f"summary_r{round_summary.round}.json"
        data = asdict(round_summary)
        data["timestamp"] = datetime.now().isoformat()
        self._write_json(path, data)
        
    def save_experiment_result(self, result: ExperimentResult):
        """Save final experiment results.
        
        Args:
            result: Complete experiment results
        """
        path = self.experiment_path / "experiment_results.json"
        data = asdict(result)
        self._write_json(path, data)
        
    def save_error_log(self, error_type: str, error_msg: str, context: Optional[Dict] = None):
        """Append error to experiment error log.
        
        Args:
            error_type: Type/category of error
            error_msg: Error message
            context: Optional context dictionary
        """
        log_path = self.experiment_path / "experiment_errors.log"
        
        error_entry = {
            "timestamp": datetime.now().isoformat(),
            "error_type": error_type,
            "error_message": error_msg,
            "context": context or {}
        }
        
        # Append to log file
        with open(log_path, 'a') as f:
            f.write(json.dumps(error_entry) + '\n')
            
    def save_partial_results(self, round_num: int, partial_data: Dict[str, Any]):
        """Save partial results in case of failure.
        
        Args:
            round_num: Last completed round number
            partial_data: Any partial data to preserve
        """
        path = self.experiment_path / "partial_results.json"
        data = {
            "last_completed_round": round_num,
            "timestamp": datetime.now().isoformat(),
            "partial_data": partial_data
        }
        self._write_json(path, data)
        
    def get_experiment_path(self) -> Path:
        """Get the current experiment directory path.
        
        Returns:
            Path to experiment directory
        """
        return self.experiment_path