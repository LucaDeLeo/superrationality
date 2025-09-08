#!/usr/bin/env python3
"""Test script to verify power randomization creates a bell curve distribution."""

from src.core.models import Agent
from src.utils.game_logic import randomize_powers_for_round
import statistics

def test_power_distribution():
    """Test that power randomization creates a proper bell curve."""
    
    # Create agents
    num_agents = 100
    agents = [Agent(id=i % 10) for i in range(num_agents)]  # Use mod 10 for valid IDs
    
    # Run multiple rounds to collect power distribution data
    all_powers = []
    num_rounds = 100
    
    for round_num in range(num_rounds):
        # Randomize powers
        randomize_powers_for_round(agents, mean=100.0, std_dev=15.0)
        
        # Collect power values
        round_powers = [agent.power for agent in agents]
        all_powers.extend(round_powers)
        
        # Print stats for first few rounds
        if round_num < 3:
            print(f"\nRound {round_num + 1} Power Distribution:")
            print(f"  Min: {min(round_powers):.2f}")
            print(f"  Max: {max(round_powers):.2f}")
            print(f"  Mean: {statistics.mean(round_powers):.2f}")
            print(f"  Std Dev: {statistics.stdev(round_powers):.2f}")
            
            # Show power distribution across ranges
            ranges = [(50, 70), (70, 85), (85, 100), (100, 115), (115, 130), (130, 150)]
            print("  Distribution:")
            for low, high in ranges:
                count = sum(1 for p in round_powers if low <= p < high)
                bar = '█' * (count // 2)
                print(f"    {low:3d}-{high:3d}: {count:3d} {bar}")
    
    # Overall statistics
    print(f"\nOverall Power Distribution ({len(all_powers)} samples):")
    print(f"  Min: {min(all_powers):.2f}")
    print(f"  Max: {max(all_powers):.2f}")
    print(f"  Mean: {statistics.mean(all_powers):.2f}")
    print(f"  Std Dev: {statistics.stdev(all_powers):.2f}")
    print(f"  Powers at bounds (50 or 150): {sum(1 for p in all_powers if p in [50.0, 150.0])}")
    
    # Show overall distribution
    print("\n  Overall Distribution (Bell Curve):")
    ranges = [(50, 60), (60, 70), (70, 80), (80, 90), (90, 100), 
              (100, 110), (110, 120), (120, 130), (130, 140), (140, 150)]
    max_count = 0
    counts = []
    for low, high in ranges:
        count = sum(1 for p in all_powers if low <= p < high)
        counts.append(count)
        max_count = max(max_count, count)
    
    # Normalize and display
    for i, (low, high) in enumerate(ranges):
        count = counts[i]
        bar_len = int(50 * count / max_count) if max_count > 0 else 0
        bar = '█' * bar_len
        percentage = 100 * count / len(all_powers)
        print(f"    {low:3d}-{high:3d}: {count:4d} ({percentage:5.1f}%) {bar}")
    
    # Test that it approximates a normal distribution
    mean = statistics.mean(all_powers)
    std_dev = statistics.stdev(all_powers)
    
    # Check if mean is close to 100
    assert 99 < mean < 101, f"Mean {mean:.2f} should be close to 100"
    
    # Check if std dev is close to 15 (will be slightly less due to bounds)
    assert 13 < std_dev < 16, f"Std dev {std_dev:.2f} should be close to 15"
    
    # Check bell curve shape - middle ranges should have more values
    middle_count = sum(1 for p in all_powers if 85 <= p <= 115)
    middle_percentage = 100 * middle_count / len(all_powers)
    print(f"\n  Powers within 1 std dev (85-115): {middle_percentage:.1f}%")
    print(f"  Expected for normal distribution: ~68%")
    
    assert 60 < middle_percentage < 75, f"Middle range should contain ~68% of values, got {middle_percentage:.1f}%"
    
    print("\n✅ Power randomization test passed! Bell curve distribution confirmed.")

if __name__ == "__main__":
    test_power_distribution()