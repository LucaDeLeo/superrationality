"""Experiment result caching to avoid re-running expensive API calls."""

import json
import hashlib
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class ExperimentCache:
    """Cache experiment results to avoid re-running identical scenarios."""
    
    def __init__(self, cache_dir: Path = Path("results/.cache")):
        """Initialize experiment cache.
        
        Args:
            cache_dir: Directory to store cached results
        """
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.index_file = self.cache_dir / "cache_index.json"
        self.index = self._load_index()
    
    def _load_index(self) -> Dict[str, Any]:
        """Load cache index from disk."""
        if self.index_file.exists():
            with open(self.index_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_index(self):
        """Save cache index to disk."""
        with open(self.index_file, 'w') as f:
            json.dump(self.index, f, indent=2, default=str)
    
    def _generate_cache_key(self, 
                           scenario_name: str, 
                           num_agents: int, 
                           num_rounds: int,
                           model_distribution: Dict[str, int]) -> str:
        """Generate unique cache key for experiment configuration.
        
        Args:
            scenario_name: Name of scenario
            num_agents: Number of agents
            num_rounds: Number of rounds
            model_distribution: Model distribution dict
            
        Returns:
            Unique hash key for this configuration
        """
        # Create deterministic string representation
        config_str = json.dumps({
            'scenario': scenario_name,
            'agents': num_agents,
            'rounds': num_rounds,
            'models': sorted(model_distribution.items())
        }, sort_keys=True)
        
        # Generate hash
        return hashlib.sha256(config_str.encode()).hexdigest()[:16]
    
    def get_cached_result(self, 
                         scenario_name: str,
                         num_agents: int,
                         num_rounds: int,
                         model_distribution: Dict[str, int],
                         max_age_hours: int = 24 * 7) -> Optional[Dict[str, Any]]:
        """Get cached result if available and not expired.
        
        Args:
            scenario_name: Name of scenario
            num_agents: Number of agents
            num_rounds: Number of rounds
            model_distribution: Model distribution
            max_age_hours: Maximum age of cache in hours (default 7 days)
            
        Returns:
            Cached result or None if not found/expired
        """
        cache_key = self._generate_cache_key(
            scenario_name, num_agents, num_rounds, model_distribution
        )
        
        if cache_key not in self.index:
            logger.debug(f"No cache entry for {scenario_name}")
            return None
        
        entry = self.index[cache_key]
        
        # Check age
        cached_time = datetime.fromisoformat(entry['timestamp'])
        age = datetime.now() - cached_time
        if age > timedelta(hours=max_age_hours):
            logger.info(f"Cache expired for {scenario_name} (age: {age})")
            return None
        
        # Load cached result
        cache_file = self.cache_dir / f"{cache_key}.json"
        if not cache_file.exists():
            logger.warning(f"Cache file missing for {scenario_name}")
            return None
        
        with open(cache_file, 'r') as f:
            result = json.load(f)
        
        logger.info(f"✨ Using cached result for {scenario_name} (age: {age})")
        logger.info(f"   Original experiment: {entry['experiment_id']}")
        logger.info(f"   Saved cost: ${entry.get('cost', 0):.4f}")
        
        return result
    
    def save_result(self,
                   scenario_name: str,
                   num_agents: int,
                   num_rounds: int,
                   model_distribution: Dict[str, int],
                   result: Dict[str, Any],
                   experiment_id: str,
                   cost: float = 0.0):
        """Save result to cache.
        
        Args:
            scenario_name: Name of scenario
            num_agents: Number of agents
            num_rounds: Number of rounds
            model_distribution: Model distribution
            result: Experiment result to cache
            experiment_id: Original experiment ID
            cost: Cost of the experiment
        """
        cache_key = self._generate_cache_key(
            scenario_name, num_agents, num_rounds, model_distribution
        )
        
        # Save result file
        cache_file = self.cache_dir / f"{cache_key}.json"
        with open(cache_file, 'w') as f:
            json.dump(result, f, indent=2, default=str)
        
        # Update index
        self.index[cache_key] = {
            'scenario': scenario_name,
            'experiment_id': experiment_id,
            'timestamp': datetime.now().isoformat(),
            'num_agents': num_agents,
            'num_rounds': num_rounds,
            'model_distribution': model_distribution,
            'cost': cost,
            'cache_file': str(cache_file)
        }
        self._save_index()
        
        logger.info(f"💾 Cached result for {scenario_name} (key: {cache_key})")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get statistics about cached experiments.
        
        Returns:
            Cache statistics
        """
        total_experiments = len(self.index)
        total_cost_saved = sum(entry.get('cost', 0) for entry in self.index.values())
        
        # Group by scenario
        scenarios = {}
        for entry in self.index.values():
            scenario = entry['scenario']
            if scenario not in scenarios:
                scenarios[scenario] = {'count': 0, 'cost_saved': 0}
            scenarios[scenario]['count'] += 1
            scenarios[scenario]['cost_saved'] += entry.get('cost', 0)
        
        return {
            'total_cached': total_experiments,
            'total_cost_saved': total_cost_saved,
            'scenarios': scenarios,
            'cache_size_mb': sum(
                (self.cache_dir / f"{key}.json").stat().st_size 
                for key in self.index.keys()
                if (self.cache_dir / f"{key}.json").exists()
            ) / (1024 * 1024)
        }
    
    def clear_cache(self, older_than_hours: Optional[int] = None):
        """Clear cache entries.
        
        Args:
            older_than_hours: Only clear entries older than this (None = clear all)
        """
        if older_than_hours is None:
            # Clear all
            for key in list(self.index.keys()):
                cache_file = self.cache_dir / f"{key}.json"
                if cache_file.exists():
                    cache_file.unlink()
            self.index = {}
            self._save_index()
            logger.info("🗑️ Cleared all cache entries")
        else:
            # Clear old entries
            cutoff = datetime.now() - timedelta(hours=older_than_hours)
            cleared = 0
            for key in list(self.index.keys()):
                entry = self.index[key]
                if datetime.fromisoformat(entry['timestamp']) < cutoff:
                    cache_file = self.cache_dir / f"{key}.json"
                    if cache_file.exists():
                        cache_file.unlink()
                    del self.index[key]
                    cleared += 1
            self._save_index()
            logger.info(f"🗑️ Cleared {cleared} cache entries older than {older_than_hours} hours")
    
    def list_cached_scenarios(self) -> List[Dict[str, Any]]:
        """List all cached scenarios with details.
        
        Returns:
            List of cached scenario details
        """
        scenarios = []
        for key, entry in self.index.items():
            age = datetime.now() - datetime.fromisoformat(entry['timestamp'])
            scenarios.append({
                'scenario': entry['scenario'],
                'experiment_id': entry['experiment_id'],
                'age_hours': age.total_seconds() / 3600,
                'cost_saved': entry.get('cost', 0),
                'num_agents': entry['num_agents'],
                'num_rounds': entry['num_rounds']
            })
        return sorted(scenarios, key=lambda x: x['age_hours'])