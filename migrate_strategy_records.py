#!/usr/bin/env python3
"""Migration script to update existing strategy records with new metadata fields."""

import json
import os
import glob
import argparse
from datetime import datetime
from typing import Dict, Any


def migrate_strategy_record(record: Dict[str, Any]) -> Dict[str, Any]:
    """Add new metadata fields to a strategy record if missing.
    
    Args:
        record: Existing strategy record
        
    Returns:
        Updated record with new fields
    """
    # Add new fields with sensible defaults if they don't exist
    if 'model_version' not in record:
        # Try to extract version from model string if possible
        model = record.get('model', '')
        model_version = None
        if '-' in model and model.split('-')[-1].isdigit():
            model_version = model.split('-')[-1]
        record['model_version'] = model_version
    
    if 'response_format' not in record:
        # Default to unknown for existing records
        record['response_format'] = 'unknown'
    
    if 'model_params' not in record:
        # Assume default parameters were used
        record['model_params'] = {
            'temperature': 0.7,
            'max_tokens': 1000
        }
    
    if 'inference_latency' not in record:
        # Cannot determine historical latency
        record['inference_latency'] = None
    
    return record


def migrate_strategies_file(filepath: str, dry_run: bool = False) -> int:
    """Migrate a single strategies file.
    
    Args:
        filepath: Path to the strategies JSON file
        dry_run: If True, don't write changes
        
    Returns:
        Number of records migrated
    """
    print(f"Processing {filepath}...")
    
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Check if this is a strategies file
        if 'strategies' not in data:
            print(f"  Skipping - not a strategies file")
            return 0
        
        # Migrate each strategy record
        migrated_count = 0
        for strategy in data['strategies']:
            # Check if migration is needed
            if any(field not in strategy for field in ['model_version', 'response_format', 'model_params', 'inference_latency']):
                migrate_strategy_record(strategy)
                migrated_count += 1
        
        if migrated_count > 0:
            print(f"  Migrated {migrated_count} records")
            
            if not dry_run:
                # Create backup
                backup_path = filepath + '.backup'
                with open(backup_path, 'w') as f:
                    json.dump(data, f, indent=2)
                print(f"  Created backup at {backup_path}")
                
                # Write updated data
                with open(filepath, 'w') as f:
                    json.dump(data, f, indent=2)
                print(f"  Updated {filepath}")
        else:
            print(f"  No migration needed")
        
        return migrated_count
        
    except Exception as e:
        print(f"  Error: {e}")
        return 0


def main():
    """Main migration function."""
    parser = argparse.ArgumentParser(description='Migrate strategy records to new format')
    parser.add_argument('path', help='Path to results directory or specific file')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be changed without modifying files')
    args = parser.parse_args()
    
    # Find all strategy files
    if os.path.isfile(args.path):
        files = [args.path]
    else:
        # Look for strategies files in rounds subdirectories
        pattern = os.path.join(args.path, '**/rounds/strategies_r*.json')
        files = glob.glob(pattern, recursive=True)
    
    if not files:
        print(f"No strategy files found in {args.path}")
        return 1
    
    print(f"Found {len(files)} strategy files to process")
    if args.dry_run:
        print("DRY RUN - No files will be modified")
    print()
    
    # Process each file
    total_migrated = 0
    for filepath in files:
        migrated = migrate_strategies_file(filepath, args.dry_run)
        total_migrated += migrated
    
    print()
    print(f"Total records migrated: {total_migrated}")
    
    if args.dry_run:
        print("\nThis was a dry run. Run without --dry-run to apply changes.")
    
    return 0


if __name__ == '__main__':
    exit(main())