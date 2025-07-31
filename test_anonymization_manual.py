"""Manual test for anonymization implementation."""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from src.managers.anonymization import AnonymizationManager


def test_basic_functionality():
    """Test basic anonymization functionality."""
    print("\n=== Testing Basic Functionality ===")
    
    # Test round 1
    manager1 = AnonymizationManager(round_num=1, num_agents=5)
    
    print("\nRound 1 mappings:")
    for i in range(5):
        anon_id = manager1.anonymize(i)
        print(f"  Agent {i} -> {anon_id}")
    
    # Test consistency
    print("\nTesting consistency:")
    agent_2_anon = manager1.anonymize(2)
    print(f"  Agent 2 anonymized again: {agent_2_anon}")
    print(f"  Deanonymize {agent_2_anon}: {manager1.deanonymize(agent_2_anon)}")
    
    # Test round 2  
    print("\n\nRound 2 mappings (should be different):")
    manager2 = AnonymizationManager(round_num=2, num_agents=5)
    for i in range(5):
        anon_id = manager2.anonymize(i)
        print(f"  Agent {i} -> {anon_id}")
    
    # Test shuffled order
    print("\nShuffled order for strategy collection:")
    print(f"  Round 1: {manager1.get_shuffled_order()}")
    print(f"  Round 2: {manager2.get_shuffled_order()}")
    
    # Test determinism
    print("\n\nTesting determinism:")
    manager1_copy = AnonymizationManager(round_num=1, num_agents=5)
    print("  Creating another manager for round 1...")
    matches = True
    for i in range(5):
        if manager1.anonymize(i) != manager1_copy.anonymize(i):
            matches = False
            break
    print(f"  Mappings match: {matches}")
    
    print("\n✅ Basic functionality test complete")
    

def test_validation():
    """Test validation methods."""
    print("\n=== Testing Validation ===")
    
    manager = AnonymizationManager(round_num=1, num_agents=3)
    
    # Test validation
    is_valid = manager.validate_consistency()
    print(f"  Mapping consistency: {'✅ Valid' if is_valid else '❌ Invalid'}")
    
    # Test unknown agent
    try:
        unknown_id = manager.anonymize(99)
        print(f"  Unknown agent 99 -> {unknown_id}")
    except Exception as e:
        print(f"  Error handling unknown agent: {e}")
    
    print("\n✅ Validation test complete")


def test_format_display():
    """Test the anonymous ID format."""
    print("\n=== Testing ID Format ===")
    
    for round_num in [1, 5, 10]:
        manager = AnonymizationManager(round_num=round_num, num_agents=3)
        print(f"\nRound {round_num}:")
        for i in range(3):
            print(f"  {manager.anonymize(i)}")
    
    print("\n✅ Format test complete")


if __name__ == "__main__":
    test_basic_functionality()
    test_validation()
    test_format_display()
    print("\n\n🎉 All manual tests completed!")