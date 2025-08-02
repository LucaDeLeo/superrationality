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
    
    def __init__(self, base_path: str = "results", scenario_name: Optional[str] = None):
        """Initialize DataManager with experiment directory structure.
        
        Args:
            base_path: Base directory for all experiment results
            scenario_name: Optional scenario name for segmented storage
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(exist_ok=True)
        self.experiment_id = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.scenario_name = scenario_name
        
        # Create scenario-specific path if provided
        if scenario_name:
            self.experiment_path = self.base_path / "scenarios" / scenario_name / self.experiment_id
        else:
            self.experiment_path = self.base_path / self.experiment_id
            
        self._setup_directories()
        
    def _setup_directories(self):
        """Create experiment directory structure."""
        self.experiment_path.mkdir(parents=True, exist_ok=True)
        (self.experiment_path / "rounds").mkdir(parents=True, exist_ok=True)
        
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
        # Include scenario in filename if present
        if self.scenario_name:
            filename = f"strategies_{self.scenario_name}_r{round_num}.json"
        else:
            filename = f"strategies_r{round_num}.json"
        path = self.experiment_path / "rounds" / filename
        
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
        # Include scenario in filename if present
        if self.scenario_name:
            filename = f"games_{self.scenario_name}_r{round_num}.json"
        else:
            filename = f"games_r{round_num}.json"
        path = self.experiment_path / "rounds" / filename
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
        # Include scenario in filename if present
        if self.scenario_name:
            filename = f"summary_{self.scenario_name}_r{round_summary.round}.json"
        else:
            filename = f"summary_r{round_summary.round}.json"
        path = self.experiment_path / "rounds" / filename
        data = asdict(round_summary)
        data["timestamp"] = datetime.now().isoformat()
        self._write_json(path, data)
        
    def save_experiment_result(self, result: ExperimentResult):
        """Save final experiment results.
        
        Args:
            result: Complete experiment results
        """
        # Include scenario in filename if present
        if self.scenario_name:
            filename = f"experiment_results_{self.scenario_name}.json"
        else:
            filename = "experiment_results.json"
        path = self.experiment_path / filename
        data = asdict(result)
        
        # Add scenario metadata if present
        if self.scenario_name:
            data["scenario_name"] = self.scenario_name
            
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
    
    def get_scenario_experiments(self, scenario_name: str) -> List[Path]:
        """
        Get all experiment paths for a specific scenario.
        
        Args:
            scenario_name: Name of the scenario
            
        Returns:
            List of paths to experiment directories
        """
        scenario_path = self.base_path / "scenarios" / scenario_name
        if not scenario_path.exists():
            return []
        
        experiments = []
        for exp_dir in scenario_path.iterdir():
            if exp_dir.is_dir() and exp_dir.name.startswith("exp_"):
                experiments.append(exp_dir)
        
        return sorted(experiments)
    
    def load_scenario_data(self, scenario_name: str, data_type: str = "experiment_results") -> List[Dict]:
        """
        Load all data of a specific type for a scenario.
        
        Args:
            scenario_name: Name of the scenario
            data_type: Type of data to load (e.g., "experiment_results", "round_summaries")
            
        Returns:
            List of loaded data dictionaries
        """
        experiments = self.get_scenario_experiments(scenario_name)
        all_data = []
        
        for exp_path in experiments:
            if data_type == "experiment_results":
                result_files = list(exp_path.glob("experiment_results*.json"))
                if result_files:
                    with open(result_files[0], 'r') as f:
                        data = json.load(f)
                        data["experiment_path"] = str(exp_path)
                        all_data.append(data)
            elif data_type == "round_summaries":
                rounds_path = exp_path / "rounds"
                if rounds_path.exists():
                    for summary_file in rounds_path.glob("summary*.json"):
                        with open(summary_file, 'r') as f:
                            data = json.load(f)
                            data["experiment_path"] = str(exp_path)
                            all_data.append(data)
        
        return all_data