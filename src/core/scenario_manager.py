"""
Scenario Manager for Mixed Model Experiments
Manages predefined model distributions and ensures consistent ratios
"""
from typing import Dict, List, Optional, Tuple
import random
import json
from pathlib import Path
import math

from src.core.models import ScenarioConfig


class ScenarioManager:
    """Manages model distribution scenarios for experiments"""
    
    def __init__(self, num_agents: int = 10):
        self.num_agents = num_agents
        self.current_scenario: Optional[ScenarioConfig] = None
        self.model_assignments: Dict[int, str] = {}  # agent_id -> model_name
        
    def validate_scenario(self, scenario: ScenarioConfig) -> Tuple[bool, str]:
        """
        Validates that a scenario configuration is valid
        Returns: (is_valid, error_message)
        """
        # Check that model counts sum to NUM_AGENTS
        total_agents = sum(scenario.model_distribution.values())
        if total_agents != self.num_agents:
            return False, f"Model counts sum to {total_agents}, expected {self.num_agents}"
        
        # Check for valid model names (non-empty)
        for model_name, count in scenario.model_distribution.items():
            if not model_name or not isinstance(model_name, str):
                return False, f"Invalid model name: {model_name}"
            if count < 0 or not isinstance(count, int):
                return False, f"Invalid count for {model_name}: {count}"
        
        # Check for filesystem-safe scenario name
        if not scenario.name or '/' in scenario.name or '\\' in scenario.name:
            return False, f"Invalid scenario name for filesystem: {scenario.name}"
            
        return True, ""
    
    def assign_models_to_agents(self, agents: List['Agent'], scenario: ScenarioConfig, 
                                seed: Optional[int] = None) -> Dict[int, str]:
        """
        Assigns models to agents maintaining exact ratios with random distribution
        Returns mapping of agent_id -> model_name
        """
        # Validate scenario first
        is_valid, error_msg = self.validate_scenario(scenario)
        if not is_valid:
            raise ValueError(f"Invalid scenario: {error_msg}")
        
        # Set random seed for reproducibility
        if seed is not None:
            random.seed(seed)
        
        # Create pool of model assignments
        model_pool = []
        for model_name, count in scenario.model_distribution.items():
            model_pool.extend([model_name] * count)
        
        # Shuffle for random distribution
        random.shuffle(model_pool)
        
        # Assign models to agents
        self.model_assignments = {}
        for i, agent in enumerate(agents):
            agent_id = agent.id
            assigned_model = model_pool[i]
            self.model_assignments[agent_id] = assigned_model
            # Update agent's model_config with the assigned model
            from src.core.models import ModelConfig
            if not agent.model_config:
                agent.model_config = ModelConfig(model_type=assigned_model)
            else:
                agent.model_config.model_type = assigned_model
        
        self.current_scenario = scenario
        return self.model_assignments
    
    def get_agent_model(self, agent_id: int) -> Optional[str]:
        """Get the assigned model for a specific agent"""
        return self.model_assignments.get(agent_id)
    
    def save_scenario_assignments(self, experiment_path: Path) -> None:
        """Save model assignments for experiment reproducibility"""
        if not self.current_scenario:
            return
            
        scenario_data = {
            "scenario_name": self.current_scenario.name,
            "model_distribution": self.current_scenario.model_distribution,
            "agent_assignments": self.model_assignments,
            "total_agents": self.num_agents
        }
        
        scenario_file = experiment_path / "scenario_config.json"
        scenario_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(scenario_file, 'w') as f:
            json.dump(scenario_data, f, indent=2)
    
    def load_scenario_assignments(self, experiment_path: Path) -> Optional[ScenarioConfig]:
        """Load saved scenario assignments from a previous experiment"""
        scenario_file = experiment_path / "scenario_config.json"
        
        if not scenario_file.exists():
            return None
            
        with open(scenario_file, 'r') as f:
            data = json.load(f)
        
        scenario = ScenarioConfig(
            name=data["scenario_name"],
            model_distribution=data["model_distribution"]
        )
        
        self.current_scenario = scenario
        self.model_assignments = {int(k): v for k, v in data["agent_assignments"].items()}
        self.num_agents = data["total_agents"]
        
        return scenario
    
    def calculate_model_diversity(self) -> float:
        """
        Calculate Shannon entropy for model diversity
        H = -Σ(p_i * log(p_i)) where p_i is proportion of model i
        
        Returns:
            0.0 for homogeneous (single model)
            Higher values for more diverse distributions
        """
        if not self.current_scenario:
            return 0.0
        
        # Count actual assignments (in case of dynamic changes)
        model_counts = {}
        for model_name in self.model_assignments.values():
            model_counts[model_name] = model_counts.get(model_name, 0) + 1
        
        # Calculate proportions
        total = len(self.model_assignments)
        if total == 0:
            return 0.0
        
        # Calculate Shannon entropy
        entropy = 0.0
        for count in model_counts.values():
            if count > 0:
                p_i = count / total
                entropy -= p_i * math.log(p_i)
        
        return entropy
    
    def get_model_proportions(self) -> Dict[str, float]:
        """Get the proportion of each model in current assignments"""
        if not self.model_assignments:
            return {}
            
        model_counts = {}
        for model_name in self.model_assignments.values():
            model_counts[model_name] = model_counts.get(model_name, 0) + 1
        
        total = len(self.model_assignments)
        return {model: count/total for model, count in model_counts.items()}
    
    def get_scenario_summary(self) -> Dict:
        """Get a summary of the current scenario"""
        if not self.current_scenario:
            return {"status": "no_scenario_loaded"}
        
        return {
            "scenario_name": self.current_scenario.name,
            "model_distribution": self.current_scenario.model_distribution,
            "model_proportions": self.get_model_proportions(),
            "diversity_score": self.calculate_model_diversity(),
            "total_agents": self.num_agents,
            "assignments_count": len(self.model_assignments)
        }