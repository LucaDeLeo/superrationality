"""Flexible prompt management system for experimenting with different prompt variations."""

import json
import random
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class PromptExperiment:
    """Configuration for a prompt experiment."""
    id: str
    name: str
    description: str
    prompt_template: str
    include_identity: bool
    include_global_cooperation: bool
    include_round_summaries: bool
    default_on_ambiguity: str  # 'cooperate', 'defect', or 'random'
    fallback_strategy: str  # 'cooperate', 'defect', or 'random'
    include_personal_history: bool = False
    
    def get_default_action(self) -> str:
        """Get the default action for ambiguous responses."""
        if self.default_on_ambiguity == 'random':
            return random.choice(['COOPERATE', 'DEFECT'])
        elif self.default_on_ambiguity == 'cooperate':
            return 'COOPERATE'
        else:
            return 'DEFECT'
    
    def get_fallback_strategy(self) -> str:
        """Get the fallback strategy for failed agents."""
        if self.fallback_strategy == 'random':
            return random.choice(['Always Cooperate', 'Always Defect', 'Tit-for-Tat'])
        elif self.fallback_strategy == 'cooperate':
            return 'Always Cooperate'
        else:
            return 'Always Defect'


class PromptManager:
    """Manages different prompt experiments and variations."""
    
    def __init__(self, experiments_file: str = "prompt_experiments.json"):
        """Initialize the prompt manager.
        
        Args:
            experiments_file: Path to the JSON file containing experiment definitions
        """
        self.experiments_file = Path(experiments_file)
        self.experiments: Dict[str, PromptExperiment] = {}
        self.meta_experiments: Dict[str, Dict] = {}
        self.current_experiment: Optional[PromptExperiment] = None
        self.load_experiments()
    
    def load_experiments(self):
        """Load experiment definitions from JSON file."""
        if not self.experiments_file.exists():
            logger.warning(f"Experiments file {self.experiments_file} not found")
            return
        
        try:
            with open(self.experiments_file, 'r') as f:
                data = json.load(f)
            
            # Load individual experiments
            for exp_data in data.get('prompt_experiments', []):
                exp = PromptExperiment(**exp_data)
                self.experiments[exp.id] = exp
            
            # Load meta experiments
            self.meta_experiments = {
                meta['id']: meta 
                for meta in data.get('meta_experiments', [])
            }
            
            logger.info(f"Loaded {len(self.experiments)} prompt experiments")
            
        except Exception as e:
            logger.error(f"Failed to load experiments: {e}")
    
    def set_experiment(self, experiment_id: str) -> bool:
        """Set the current experiment.
        
        Args:
            experiment_id: ID of the experiment to use
            
        Returns:
            True if successful, False otherwise
        """
        if experiment_id not in self.experiments:
            logger.error(f"Unknown experiment ID: {experiment_id}")
            return False
        
        self.current_experiment = self.experiments[experiment_id]
        logger.info(f"Set current experiment to: {self.current_experiment.name}")
        return True
    
    def get_strategy_prompt(self, context: Dict[str, Any]) -> str:
        """Generate a strategy collection prompt based on current experiment.
        
        Args:
            context: Context containing round information, history, etc.
            
        Returns:
            Formatted prompt string
        """
        if not self.current_experiment:
            raise ValueError("No experiment set. Call set_experiment() first.")
        
        exp = self.current_experiment
        template = exp.prompt_template
        
        # Build substitution dictionary
        substitutions = {}
        
        # Add cooperation rate if included
        if exp.include_global_cooperation and '{coop_rate}' in template:
            coop_rate = context.get('coop_rate', 50.0)
            substitutions['coop_rate'] = f"{coop_rate:.1f}"
        
        # Add distribution if included
        if '{distribution}' in template:
            distribution = context.get('distribution', 'No data available')
            substitutions['distribution'] = distribution
        
        # Add round summaries if included
        if exp.include_round_summaries and '{previous_rounds_detail}' in template:
            rounds_detail = context.get('previous_rounds_detail', '')
            substitutions['previous_rounds_detail'] = rounds_detail
        
        # Add history info (generic)
        if '{history_info}' in template:
            if context.get('round', 1) == 1:
                history_info = "This is the first round."
            else:
                history_info = f"You have played {context.get('round', 1) - 1} rounds so far."
            substitutions['history_info'] = history_info
        
        # Add personal history if included
        if exp.include_personal_history and '{personal_history}' in template:
            personal_history = context.get('personal_history', 'No previous games.')
            substitutions['personal_history'] = personal_history
        
        # Perform substitutions
        prompt = template
        for key, value in substitutions.items():
            prompt = prompt.replace(f'{{{key}}}', str(value))
        
        return prompt
    
    def get_decision_prompt(self, strategy: str, game_history: str, opponent_label: str) -> str:
        """Generate a decision prompt for subagent.
        
        Args:
            strategy: The strategy to follow
            game_history: Formatted game history
            opponent_label: Anonymous opponent label
            
        Returns:
            Formatted decision prompt
        """
        # Use a consistent decision prompt across experiments
        # The variation is mainly in strategy collection
        prompt = f"""You are playing prisoner's dilemma. Your strategy:
{strategy}

Game history so far:
{game_history}

Current opponent: Agent {opponent_label}

Decision (COOPERATE/DEFECT):"""
        
        return prompt
    
    def get_default_action(self) -> str:
        """Get the default action for ambiguous responses."""
        if not self.current_experiment:
            return 'COOPERATE'  # Default fallback
        return self.current_experiment.get_default_action()
    
    def get_fallback_strategy(self) -> str:
        """Get the fallback strategy for failed agents."""
        if not self.current_experiment:
            return 'Always Cooperate'  # Default fallback
        return self.current_experiment.get_fallback_strategy()
    
    def list_experiments(self) -> List[Dict[str, str]]:
        """List all available experiments.
        
        Returns:
            List of experiment info dictionaries
        """
        return [
            {
                'id': exp.id,
                'name': exp.name,
                'description': exp.description
            }
            for exp in self.experiments.values()
        ]
    
    def list_meta_experiments(self) -> List[Dict[str, Any]]:
        """List all meta experiments.
        
        Returns:
            List of meta experiment info
        """
        return [
            {
                'id': meta['id'],
                'name': meta['name'],
                'description': meta['description'],
                'experiments': meta['experiments']
            }
            for meta in self.meta_experiments.values()
        ]
    
    def get_experiment_config(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """Get full configuration for an experiment.
        
        Args:
            experiment_id: ID of the experiment
            
        Returns:
            Configuration dictionary or None if not found
        """
        if experiment_id not in self.experiments:
            return None
        
        exp = self.experiments[experiment_id]
        return {
            'id': exp.id,
            'name': exp.name,
            'description': exp.description,
            'include_identity': exp.include_identity,
            'include_global_cooperation': exp.include_global_cooperation,
            'include_round_summaries': exp.include_round_summaries,
            'default_on_ambiguity': exp.default_on_ambiguity,
            'fallback_strategy': exp.fallback_strategy
        }
    
    def compare_experiments(self, exp_ids: List[str]) -> Dict[str, Any]:
        """Compare configuration of multiple experiments.
        
        Args:
            exp_ids: List of experiment IDs to compare
            
        Returns:
            Comparison dictionary
        """
        comparison = {
            'experiments': {},
            'differences': {
                'include_identity': set(),
                'include_global_cooperation': set(),
                'include_round_summaries': set(),
                'default_on_ambiguity': set(),
                'fallback_strategy': set()
            }
        }
        
        for exp_id in exp_ids:
            if exp_id in self.experiments:
                exp = self.experiments[exp_id]
                comparison['experiments'][exp_id] = {
                    'name': exp.name,
                    'include_identity': exp.include_identity,
                    'include_global_cooperation': exp.include_global_cooperation,
                    'include_round_summaries': exp.include_round_summaries,
                    'default_on_ambiguity': exp.default_on_ambiguity,
                    'fallback_strategy': exp.fallback_strategy
                }
                
                # Track unique values for each field
                comparison['differences']['include_identity'].add(exp.include_identity)
                comparison['differences']['include_global_cooperation'].add(exp.include_global_cooperation)
                comparison['differences']['include_round_summaries'].add(exp.include_round_summaries)
                comparison['differences']['default_on_ambiguity'].add(exp.default_on_ambiguity)
                comparison['differences']['fallback_strategy'].add(exp.fallback_strategy)
        
        # Convert sets to lists for JSON serialization
        for key in comparison['differences']:
            comparison['differences'][key] = list(comparison['differences'][key])
        
        return comparison