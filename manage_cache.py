#!/usr/bin/env python3
"""Cache management tool for ULTRATHINK experiments."""

import argparse
import sys
from pathlib import Path
from datetime import datetime
from src.utils.experiment_cache import ExperimentCache


def main():
    """Main entry point for cache management."""
    parser = argparse.ArgumentParser(description="Manage ULTRATHINK experiment cache")
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Stats command
    stats_parser = subparsers.add_parser('stats', help='Show cache statistics')
    
    # List command
    list_parser = subparsers.add_parser('list', help='List cached experiments')
    list_parser.add_argument('--verbose', '-v', action='store_true', 
                            help='Show detailed information')
    
    # Clear command
    clear_parser = subparsers.add_parser('clear', help='Clear cache')
    clear_parser.add_argument('--older-than', type=int, metavar='HOURS',
                             help='Only clear entries older than N hours')
    clear_parser.add_argument('--confirm', action='store_true',
                             help='Skip confirmation prompt')
    
    # Info command
    info_parser = subparsers.add_parser('info', help='Show cache directory info')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    cache = ExperimentCache()
    
    if args.command == 'stats':
        stats = cache.get_cache_stats()
        print("\n📊 ULTRATHINK CACHE STATISTICS")
        print("="*60)
        print(f"Total cached experiments: {stats['total_cached']}")
        print(f"Total cost saved: ${stats['total_cost_saved']:.4f}")
        print(f"Cache size: {stats['cache_size_mb']:.2f} MB")
        
        if stats['scenarios']:
            print("\n📁 Cached scenarios:")
            for scenario, info in sorted(stats['scenarios'].items()):
                print(f"  {scenario:<30} {info['count']} runs, ${info['cost_saved']:.4f} saved")
        else:
            print("\n📁 No cached scenarios")
    
    elif args.command == 'list':
        scenarios = cache.list_cached_scenarios()
        if not scenarios:
            print("No cached experiments found")
            return 0
        
        print("\n📋 CACHED EXPERIMENTS")
        print("="*60)
        
        if args.verbose:
            for s in scenarios:
                age_str = f"{s['age_hours']:.1f}h" if s['age_hours'] < 24 else f"{s['age_hours']/24:.1f}d"
                print(f"\n{s['scenario']}")
                print(f"  Experiment ID: {s['experiment_id']}")
                print(f"  Age: {age_str}")
                print(f"  Cost saved: ${s['cost_saved']:.4f}")
                print(f"  Configuration: {s['num_agents']} agents, {s['num_rounds']} rounds")
        else:
            print(f"{'Scenario':<30} {'Age':<10} {'Cost Saved':<12}")
            print("-"*52)
            for s in scenarios:
                age_str = f"{s['age_hours']:.1f}h" if s['age_hours'] < 24 else f"{s['age_hours']/24:.1f}d"
                print(f"{s['scenario']:<30} {age_str:<10} ${s['cost_saved']:.4f}")
        
        print(f"\nTotal: {len(scenarios)} cached experiments")
    
    elif args.command == 'clear':
        stats = cache.get_cache_stats()
        
        if stats['total_cached'] == 0:
            print("Cache is already empty")
            return 0
        
        if args.older_than:
            print(f"Will clear cache entries older than {args.older_than} hours")
        else:
            print(f"Will clear ALL {stats['total_cached']} cached experiments")
            print(f"This will delete ${stats['total_cost_saved']:.4f} worth of cached results")
        
        if not args.confirm:
            response = input("\nAre you sure? (y/N): ")
            if response.lower() != 'y':
                print("Cancelled")
                return 0
        
        cache.clear_cache(older_than_hours=args.older_than)
        print("✅ Cache cleared successfully")
    
    elif args.command == 'info':
        print("\n📂 CACHE INFORMATION")
        print("="*60)
        print(f"Cache directory: {cache.cache_dir}")
        print(f"Index file: {cache.index_file}")
        
        if cache.cache_dir.exists():
            cache_files = list(cache.cache_dir.glob("*.json"))
            print(f"Cache files: {len(cache_files)}")
            
            if cache_files:
                total_size = sum(f.stat().st_size for f in cache_files) / (1024 * 1024)
                print(f"Total size: {total_size:.2f} MB")
                
                # Find oldest and newest
                oldest = min(cache_files, key=lambda f: f.stat().st_mtime)
                newest = max(cache_files, key=lambda f: f.stat().st_mtime)
                
                oldest_time = datetime.fromtimestamp(oldest.stat().st_mtime)
                newest_time = datetime.fromtimestamp(newest.stat().st_mtime)
                
                print(f"Oldest entry: {oldest_time.strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"Newest entry: {newest_time.strftime('%Y-%m-%d %H:%M:%S')}")
        else:
            print("Cache directory does not exist")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())