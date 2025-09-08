#!/usr/bin/env python3
"""List all available scenarios for model comparison experiments."""

import json
from pathlib import Path
from collections import defaultdict

def list_scenarios():
    """List and categorize all available scenarios."""
    
    scenarios_path = Path("scenarios.json")
    if not scenarios_path.exists():
        print("❌ scenarios.json not found")
        return
    
    with open(scenarios_path, 'r') as f:
        data = json.load(f)
    
    scenarios = data.get('scenarios', [])
    
    # Categorize scenarios
    categories = defaultdict(list)
    all_models = set()
    
    for s in scenarios:
        name = s['name']
        desc = s['description']
        models = s['model_distribution']
        
        # Collect all unique models
        all_models.update(models.keys())
        
        # Categorize
        if name.startswith('homogeneous'):
            categories['Homogeneous (Single Model)'].append((name, desc, models))
        elif name.startswith('mixed'):
            categories['Mixed (Two Models)'].append((name, desc, models))
        elif name.startswith('diverse'):
            categories['Diverse (Multiple Models)'].append((name, desc, models))
        elif 'budget' in name:
            categories['Budget/Cost-Optimized'].append((name, desc, models))
        elif 'premium' in name:
            categories['Premium/High-Cost'].append((name, desc, models))
        elif 'test' in name:
            categories['Test Scenarios'].append((name, desc, models))
        else:
            categories['Special Configurations'].append((name, desc, models))
    
    # Print summary
    print("="*80)
    print("ULTRATHINK MODEL COMPARISON - AVAILABLE SCENARIOS")
    print("="*80)
    print(f"\nTotal Scenarios: {len(scenarios)}")
    print(f"Unique Models: {len(all_models)}")
    print()
    
    # Print by category
    for category, items in categories.items():
        print(f"\n{category} ({len(items)} scenarios)")
        print("-"*60)
        for name, desc, models in items:
            print(f"  📌 {name}")
            print(f"     {desc}")
            if len(models) <= 3:
                for model, count in models.items():
                    model_short = model.split('/')[-1]
                    print(f"       • {count}x {model_short}")
            else:
                print(f"       • {len(models)} different models")
    
    # Print all unique models
    print("\n" + "="*80)
    print("ALL AVAILABLE MODELS")
    print("="*80)
    
    # Group models by provider
    models_by_provider = defaultdict(list)
    for model in sorted(all_models):
        provider = model.split('/')[0]
        models_by_provider[provider].append(model)
    
    for provider, models in sorted(models_by_provider.items()):
        print(f"\n{provider.upper()}:")
        for model in models:
            model_name = model.split('/')[-1]
            print(f"  • {model_name} ({model})")
    
    # Print usage instructions
    print("\n" + "="*80)
    print("HOW TO RUN EXPERIMENTS")
    print("="*80)
    print("\n1. Run a specific scenario:")
    print("   uv run python run_experiment.py --scenario <scenario_name>")
    print("\n2. Run multiple scenarios:")
    print("   uv run python run_model_comparison.py --scenarios mixed_gemini_gpt mixed_opus_vs_gpt4turbo")
    print("\n3. Run all homogeneous scenarios:")
    print("   uv run python run_model_comparison.py --scenarios", end="")
    for name, _, _ in categories['Homogeneous (Single Model)'][:3]:
        print(f" {name}", end="")
    print(" ...")
    print("\n4. Run test scenario:")
    print("   uv run python run_model_comparison.py --test")
    print("\n5. Analyze results:")
    print("   uv run python analyze_models.py --latest --save")

if __name__ == "__main__":
    list_scenarios()