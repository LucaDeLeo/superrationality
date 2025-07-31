"""Simple test for anonymization without dependencies."""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from src.managers.anonymization import AnonymizationManager


def test_cross_round_prevention():
    """Test that anonymization prevents cross-round tracking."""
    print("\n=== Testing Cross-Round Tracking Prevention ===\n")
    
    num_agents = 6
    num_rounds = 5
    
    # Track what each agent sees about others across rounds
    tracking_data = {}
    
    for round_num in range(1, num_rounds + 1):
        print(f"Round {round_num}:")
        manager = AnonymizationManager(round_num=round_num, num_agents=num_agents)
        
        # Show mappings
        for agent_id in range(num_agents):
            anon_id = manager.anonymize(agent_id)
            print(f"  Agent {agent_id} -> {anon_id}")
            
            # Store for tracking analysis
            if agent_id not in tracking_data:
                tracking_data[agent_id] = []
            tracking_data[agent_id].append(anon_id)
        print()
    
    # Analyze tracking
    print("Checking if any anonymous IDs repeat across rounds:")
    tracking_possible = False
    
    for agent_id, anon_ids in tracking_data.items():
        unique_ids = set(anon_ids)
        if len(unique_ids) < len(anon_ids):
            print(f"  ❌ Agent {agent_id} has repeated anonymous IDs!")
            tracking_possible = True
        else:
            print(f"  ✅ Agent {agent_id}: All {len(anon_ids)} anonymous IDs are unique")
    
    if not tracking_possible:
        print("\n✅ SUCCESS: No agent can be tracked across rounds!")
    else:
        print("\n❌ FAILURE: Some agents can be tracked!")
    
    # Check determinism
    print("\nChecking determinism:")
    for round_num in [1, 3, 5]:
        m1 = AnonymizationManager(round_num=round_num, num_agents=num_agents)
        m2 = AnonymizationManager(round_num=round_num, num_agents=num_agents)
        
        match = all(m1.anonymize(i) == m2.anonymize(i) for i in range(num_agents))
        print(f"  Round {round_num}: {'✅ Deterministic' if match else '❌ Not deterministic'}")


if __name__ == "__main__":
    test_cross_round_prevention()