#!/usr/bin/env python3
"""
Monitor progress of GLM-4.5V parallel inference jobs.
"""

import os
import json
import glob
from typing import Dict, List, Tuple


def count_samples_in_file(file_path: str) -> int:
    """Count the number of samples in a JSON file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return len(data) if isinstance(data, list) else 0
    except Exception:
        return 0


def get_expected_samples() -> Dict[str, int]:
    """Get expected number of samples for each dataset."""
    return {
        "pvqa": 6176,  # After deduplication
        "slake": 19100,
        "rad": 2000
    }


def monitor_progress(output_base_dir: str = "/scratch/xc1490/projects/medvqa_2025/output/glm45v"):
    """Monitor progress across all GLM-4.5V instances."""
    
    expected_samples = get_expected_samples()
    
    print("GLM-4.5V Parallel Progress Monitor")
    print("=" * 50)
    
    # Find all instance directories
    instance_dirs = []
    if os.path.exists(output_base_dir):
        for item in os.listdir(output_base_dir):
            item_path = os.path.join(output_base_dir, item)
            if os.path.isdir(item_path) and item.isdigit():
                instance_dirs.append(int(item))
    
    instance_dirs.sort()
    
    if not instance_dirs:
        print("No instance directories found.")
        return
    
    print(f"Found {len(instance_dirs)} instance directories: {instance_dirs}")
    print()
    
    total_processed = 0
    total_expected = 0
    
    # Check each dataset across all instances
    for dataset in ["pvqa", "slake", "rad"]:
        print(f"Dataset: {dataset.upper()}")
        print("-" * 30)
        
        dataset_processed = 0
        dataset_expected = expected_samples[dataset]
        
        for instance_id in instance_dirs:
            instance_dir = os.path.join(output_base_dir, str(instance_id))
            output_file = os.path.join(instance_dir, f"test_{dataset}.json")
            
            if os.path.exists(output_file):
                count = count_samples_in_file(output_file)
                dataset_processed += count
                print(f"  Instance {instance_id}: {count:,} samples")
            else:
                print(f"  Instance {instance_id}: No output file found")
        
        remaining = dataset_expected - dataset_processed
        completion_pct = (dataset_processed / dataset_expected * 100) if dataset_expected > 0 else 0
        
        print(f"  Total processed: {dataset_processed:,} / {dataset_expected:,} ({completion_pct:.1f}%)")
        print(f"  Remaining: {remaining:,}")
        print()
        
        total_processed += dataset_processed
        total_expected += dataset_expected
    
    # Overall summary
    overall_completion = (total_processed / total_expected * 100) if total_expected > 0 else 0
    total_remaining = total_expected - total_processed
    
    print("OVERALL SUMMARY")
    print("=" * 50)
    print(f"Total processed: {total_processed:,} / {total_expected:,} ({overall_completion:.1f}%)")
    print(f"Total remaining: {total_remaining:,}")
    
    if total_remaining > 0:
        print(f"Estimated completion: {overall_completion:.1f}%")
    else:
        print("All datasets completed!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Monitor GLM-4.5V parallel inference progress")
    parser.add_argument("--output_dir", type=str, 
                       default="/scratch/xc1490/projects/medvqa_2025/output/glm45v",
                       help="Base output directory containing instance subdirectories")
    
    args = parser.parse_args()
    monitor_progress(args.output_dir)

