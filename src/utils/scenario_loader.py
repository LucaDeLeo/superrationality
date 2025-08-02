"""Utility for loading scenario configurations from YAML files."""
import yaml
from pathlib import Path
from typing import List, Dict, Optional
from src.core.models import ScenarioConfig
import logging

logger = logging.getLogger(__name__)


class ScenarioLoader:
    """Loads and manages scenario configurations from YAML files."""
    
    @staticmethod
    def load_from_yaml(yaml_path: str) -> List[ScenarioConfig]:
        """
        Load scenarios from a YAML file.
        
        Args:
            yaml_path: Path to YAML file containing scenarios
            
        Returns:
            List of ScenarioConfig objects
            
        Raises:
            FileNotFoundError: If YAML file doesn't exist
            yaml.YAMLError: If YAML is malformed
        """
        path = Path(yaml_path)
        if not path.exists():
            raise FileNotFoundError(f"Scenario file not found: {yaml_path}")
        
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        
        if not data or 'scenarios' not in data:
            logger.warning(f"No scenarios found in {yaml_path}")
            return []
        
        scenarios = []
        for scenario_data in data['scenarios']:
            try:
                scenario = ScenarioConfig(
                    name=scenario_data['name'],
                    model_distribution=scenario_data['model_distribution']
                )
                scenarios.append(scenario)
                logger.info(f"Loaded scenario: {scenario.name}")
            except KeyError as e:
                logger.error(f"Invalid scenario format, missing key: {e}")
                continue
        
        return scenarios
    
    @staticmethod
    def load_all_from_directory(directory: str) -> Dict[str, List[ScenarioConfig]]:
        """
        Load all scenario files from a directory.
        
        Args:
            directory: Path to directory containing YAML files
            
        Returns:
            Dictionary mapping filename to list of scenarios
        """
        dir_path = Path(directory)
        if not dir_path.exists():
            logger.warning(f"Scenario directory not found: {directory}")
            return {}
        
        all_scenarios = {}
        
        for yaml_file in dir_path.glob("*.yaml"):
            try:
                scenarios = ScenarioLoader.load_from_yaml(str(yaml_file))
                if scenarios:
                    all_scenarios[yaml_file.stem] = scenarios
                    logger.info(f"Loaded {len(scenarios)} scenarios from {yaml_file.name}")
            except Exception as e:
                logger.error(f"Failed to load {yaml_file}: {e}")
        
        return all_scenarios
    
    @staticmethod
    def find_scenario(name: str, scenarios: List[ScenarioConfig]) -> Optional[ScenarioConfig]:
        """
        Find a scenario by name in a list.
        
        Args:
            name: Scenario name to find
            scenarios: List of scenarios to search
            
        Returns:
            ScenarioConfig if found, None otherwise
        """
        for scenario in scenarios:
            if scenario.name == name:
                return scenario
        return None
    
    @staticmethod
    def list_all_scenarios(directory: str) -> List[str]:
        """
        Get a list of all available scenario names in a directory.
        
        Args:
            directory: Path to directory containing YAML files
            
        Returns:
            List of scenario names
        """
        all_scenarios = ScenarioLoader.load_all_from_directory(directory)
        scenario_names = []
        
        for file_scenarios in all_scenarios.values():
            for scenario in file_scenarios:
                scenario_names.append(scenario.name)
        
        return sorted(scenario_names)