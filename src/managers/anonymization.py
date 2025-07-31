"""Anonymization manager for cross-round agent identity protection."""

import random
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
import json
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class AnonymizationMapping:
    """Stores mapping between real and anonymous IDs for a round."""
    round_num: int
    real_to_anonymous: Dict[int, str]
    anonymous_to_real: Dict[str, int]
    seed_used: int


class AnonymizationManager:
    """Manages deterministic agent ID anonymization across rounds.
    
    Ensures that:
    1. Anonymous IDs are consistent within a round
    2. Anonymous IDs change between rounds
    3. Mappings are deterministic (using round as seed)
    4. Mappings can be saved/loaded for analysis
    """
    
    def __init__(self, round_num: int, num_agents: int):
        """Initialize anonymization for a specific round.
        
        Args:
            round_num: Current round number (used as seed)
            num_agents: Total number of agents
        """
        self.round_num = round_num
        self.num_agents = num_agents
        self.seed = round_num * 12345  # Deterministic seed based on round
        self._mapping: Optional[AnonymizationMapping] = None
        self._create_mapping()
    
    def _create_mapping(self) -> None:
        """Create deterministic but round-specific mapping.
        
        Uses round number as seed to ensure:
        - Same mapping for same round if re-run
        - Different mapping for different rounds
        """
        # Save current random state
        random_state = random.getstate()
        
        try:
            # Set deterministic seed based on round
            random.seed(self.seed)
            
            # Create shuffled positions
            positions = list(range(self.num_agents))
            random.shuffle(positions)
            
            # Create mappings
            real_to_anon = {}
            anon_to_real = {}
            
            for real_id in range(self.num_agents):
                # Anonymous ID format: "Player_{round}_{position}"
                anon_id = f"Player_{self.round_num}_{positions[real_id]}"
                real_to_anon[real_id] = anon_id
                anon_to_real[anon_id] = real_id
            
            self._mapping = AnonymizationMapping(
                round_num=self.round_num,
                real_to_anonymous=real_to_anon,
                anonymous_to_real=anon_to_real,
                seed_used=self.seed
            )
            
            logger.info(f"Created anonymization mapping for round {self.round_num}")
            
        finally:
            # Restore original random state
            random.setstate(random_state)
    
    def anonymize(self, agent_id: int) -> str:
        """Get anonymous ID for an agent.
        
        Args:
            agent_id: Real agent ID
            
        Returns:
            Anonymous ID string
        """
        if self._mapping is None:
            raise RuntimeError("Mapping not initialized")
        
        if agent_id not in self._mapping.real_to_anonymous:
            logger.warning(f"Agent ID {agent_id} not in mapping, returning Unknown")
            return f"Unknown_{agent_id}"
        
        return self._mapping.real_to_anonymous[agent_id]
    
    def deanonymize(self, anonymous_id: str) -> int:
        """Get real agent ID from anonymous ID.
        
        Args:
            anonymous_id: Anonymous ID string
            
        Returns:
            Real agent ID
        """
        if self._mapping is None:
            raise RuntimeError("Mapping not initialized")
        
        if anonymous_id not in self._mapping.anonymous_to_real:
            raise ValueError(f"Anonymous ID {anonymous_id} not found in mapping")
        
        return self._mapping.anonymous_to_real[anonymous_id]
    
    def get_shuffled_order(self) -> List[int]:
        """Get agent IDs in shuffled order for strategy collection.
        
        Returns:
            List of agent IDs in anonymized order
        """
        if self._mapping is None:
            raise RuntimeError("Mapping not initialized")
        
        # Extract position from anonymous IDs and sort by position
        agent_positions = []
        for real_id, anon_id in self._mapping.real_to_anonymous.items():
            # Extract position from "Player_{round}_{position}"
            position = int(anon_id.split('_')[-1])
            agent_positions.append((position, real_id))
        
        # Sort by position and return agent IDs
        agent_positions.sort(key=lambda x: x[0])
        return [real_id for _, real_id in agent_positions]
    
    def anonymize_list(self, agent_ids: List[int]) -> List[str]:
        """Anonymize a list of agent IDs.
        
        Args:
            agent_ids: List of real agent IDs
            
        Returns:
            List of anonymous IDs in same order
        """
        return [self.anonymize(aid) for aid in agent_ids]
    
    def get_mapping(self) -> Dict[str, any]:
        """Get the complete mapping for saving.
        
        Returns:
            Dictionary representation of the mapping
        """
        if self._mapping is None:
            raise RuntimeError("Mapping not initialized")
        
        return asdict(self._mapping)
    
    def save_mapping(self, filepath: Path) -> None:
        """Save anonymization mapping to JSON file.
        
        Args:
            filepath: Path to save the mapping
        """
        if self._mapping is None:
            raise RuntimeError("Mapping not initialized")
        
        mapping_data = self.get_mapping()
        
        # Ensure directory exists
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Save to file
        with open(filepath, 'w') as f:
            json.dump(mapping_data, f, indent=2)
        
        logger.info(f"Saved anonymization mapping to {filepath}")
    
    @classmethod
    def load_mapping(cls, filepath: Path) -> 'AnonymizationManager':
        """Load anonymization mapping from JSON file.
        
        Args:
            filepath: Path to load the mapping from
            
        Returns:
            AnonymizationManager instance with loaded mapping
        """
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Create instance without auto-generating mapping
        manager = cls.__new__(cls)
        manager.round_num = data['round_num']
        manager.num_agents = len(data['real_to_anonymous'])
        manager.seed = data['seed_used']
        
        # Restore mapping with correct types
        real_to_anon = {int(k): v for k, v in data['real_to_anonymous'].items()}
        anon_to_real = {k: int(v) for k, v in data['anonymous_to_real'].items()}
        
        manager._mapping = AnonymizationMapping(
            round_num=data['round_num'],
            real_to_anonymous=real_to_anon,
            anonymous_to_real=anon_to_real,
            seed_used=data['seed_used']
        )
        
        logger.info(f"Loaded anonymization mapping from {filepath}")
        return manager
    
    def validate_consistency(self) -> bool:
        """Validate that the mapping is internally consistent.
        
        Returns:
            True if mapping is valid and consistent
        """
        if self._mapping is None:
            return False
        
        # Check bidirectional consistency
        for real_id, anon_id in self._mapping.real_to_anonymous.items():
            if self._mapping.anonymous_to_real.get(anon_id) != real_id:
                logger.error(f"Inconsistent mapping: {real_id} -> {anon_id}")
                return False
        
        # Check all agents are mapped
        expected_ids = set(range(self.num_agents))
        actual_ids = set(self._mapping.real_to_anonymous.keys())
        if expected_ids != actual_ids:
            logger.error(f"Missing agent IDs: {expected_ids - actual_ids}")
            return False
        
        return True