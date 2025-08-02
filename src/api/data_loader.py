"""
Data loader for reading experiment results from the file system.
"""
import json
import os
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any
from functools import lru_cache
import time

from src.utils.data_manager import DataManager


class DataLoader:
    """Loads experiment data from the file system with caching."""
    
    def __init__(self, results_path: str = "results", cache_ttl: int = 300):
        """
        Initialize the data loader.
        
        Args:
            results_path: Path to the results directory
            cache_ttl: Cache time-to-live in seconds (default 5 minutes)
        """
        self.results_path = Path(results_path)
        self.cache_ttl = cache_ttl
        self._cache: Dict[str, Any] = {}
        self._cache_timestamps: Dict[str, float] = {}
    
    def _is_cache_valid(self, key: str) -> bool:
        """Check if a cache entry is still valid."""
        if key not in self._cache_timestamps:
            return False
        return time.time() - self._cache_timestamps[key] < self.cache_ttl
    
    def _get_from_cache(self, key: str) -> Optional[Any]:
        """Get a value from cache if it's valid."""
        if self._is_cache_valid(key):
            return self._cache.get(key)
        return None
    
    def _set_cache(self, key: str, value: Any) -> None:
        """Set a value in the cache."""
        self._cache[key] = value
        self._cache_timestamps[key] = time.time()
    
    def list_experiments(self) -> List[dict]:
        """List all experiments with basic information."""
        cache_key = "experiments_list"
        cached = self._get_from_cache(cache_key)
        if cached is not None:
            return cached
        
        experiments = []
        
        if not self.results_path.exists():
            return experiments
        
        for exp_dir in self.results_path.iterdir():
            if not exp_dir.is_dir():
                continue
            
            # Try to load the experiment summary
            summary_path = exp_dir / "experiment_summary.json"
            if summary_path.exists():
                try:
                    with open(summary_path, 'r') as f:
                        data = json.load(f)
                        experiments.append({
                            "experiment_id": data["experiment_id"],
                            "start_time": data["start_time"],
                            "end_time": data["end_time"],
                            "total_rounds": data["total_rounds"],
                            "total_games": data["total_games"],
                            "total_api_calls": data["total_api_calls"],
                            "total_cost": data["total_cost"],
                            "status": "completed"
                        })
                except Exception:
                    # Skip experiments with invalid data
                    continue
        
        self._set_cache(cache_key, experiments)
        return experiments
    
    def get_experiment(self, experiment_id: str) -> Optional[dict]:
        """Get detailed information about a specific experiment."""
        # Validate experiment_id to prevent directory traversal
        if not experiment_id or '..' in experiment_id or '/' in experiment_id or '\\' in experiment_id:
            return None
            
        cache_key = f"experiment_{experiment_id}"
        cached = self._get_from_cache(cache_key)
        if cached is not None:
            return cached
        
        exp_path = self.results_path / experiment_id
        # Ensure the path is within results directory
        try:
            exp_path = exp_path.resolve()
            if not exp_path.is_relative_to(self.results_path.resolve()):
                return None
        except (ValueError, RuntimeError):
            return None
            
        if not exp_path.exists():
            return None
        
        summary_path = exp_path / "experiment_summary.json"
        if not summary_path.exists():
            return None
        
        try:
            with open(summary_path, 'r') as f:
                data = json.load(f)
                
            # Count rounds and agents
            rounds_dir = exp_path / "rounds"
            round_count = len(list(rounds_dir.glob("round_*.json"))) if rounds_dir.exists() else 0
            
            # Get agent count from first round if available
            agent_count = 0
            if round_count > 0:
                first_round_path = rounds_dir / "round_1.json"
                if first_round_path.exists():
                    with open(first_round_path, 'r') as f:
                        round_data = json.load(f)
                        agent_count = len(round_data.get("agents", []))
            
            result = {
                "experiment_id": data["experiment_id"],
                "start_time": data["start_time"],
                "end_time": data["end_time"],
                "total_rounds": data["total_rounds"],
                "total_games": data["total_games"],
                "total_api_calls": data["total_api_calls"],
                "total_cost": data["total_cost"],
                "acausal_indicators": data.get("acausal_indicators", {}),
                "round_count": round_count,
                "agent_count": agent_count
            }
            
            self._set_cache(cache_key, result)
            return result
            
        except Exception:
            return None
    
    def get_round_data(self, experiment_id: str, round_num: int) -> Optional[dict]:
        """Get data for a specific round."""
        # Validate experiment_id to prevent directory traversal
        if not experiment_id or '..' in experiment_id or '/' in experiment_id or '\\' in experiment_id:
            return None
            
        cache_key = f"round_{experiment_id}_{round_num}"
        cached = self._get_from_cache(cache_key)
        if cached is not None:
            return cached
        
        round_path = self.results_path / experiment_id / "rounds" / f"round_{round_num}.json"
        # Ensure the path is within results directory
        try:
            round_path = round_path.resolve()
            if not round_path.is_relative_to(self.results_path.resolve()):
                return None
        except (ValueError, RuntimeError):
            return None
            
        if not round_path.exists():
            return None
        
        try:
            with open(round_path, 'r') as f:
                data = json.load(f)
            
            # Also load strategies and games if available
            strategies_path = self.results_path / experiment_id / "rounds" / f"strategies_r{round_num}.json"
            games_path = self.results_path / experiment_id / "rounds" / f"games_r{round_num}.json"
            
            if strategies_path.exists():
                with open(strategies_path, 'r') as f:
                    data["strategies"] = json.load(f)
            
            if games_path.exists():
                with open(games_path, 'r') as f:
                    data["games"] = json.load(f)
            
            self._set_cache(cache_key, data)
            return data
            
        except Exception:
            return None
    
    def clear_cache(self) -> None:
        """Clear all cached data."""
        self._cache.clear()
        self._cache_timestamps.clear()