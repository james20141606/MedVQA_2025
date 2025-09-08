#!/usr/bin/env python3
"""
Monitor GLM4V progress and show remaining samples
"""

import os
import json
import time
from typing import Dict, List, Any

def count_samples_in_file(file_path: str) -> int:
    """Count samples in a JSON file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return len(data)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return 0

def get_expected_samples() -> Dict[str, int]:
    """Get expected sample counts for each dataset"""
    return {
        'pvqa': 6176,  # After removing duplicates
        'slake': 1061,
        'rad': 451
    }

def get_current_progress(output_dir: str) -> Dict[str, Dict[str, Any]]:
    """Get current progress for each dataset"""
    expected = get_expected_samples()
    progress = {}
    
    for dataset in ['pvqa', 'slake', 'rad']:
        file_path = os.path.join(output_dir, f"test_{dataset}.json")
        current_samples = count_samples_in_file(file_path)
        expected_samples = expected[dataset]
        remaining = max(0, expected_samples - current_samples)
        completion_pct = (current_samples / expected_samples) * 100 if expected_samples > 0 else 0
        
        progress[dataset] = {
            'current': current_samples,
            'expected': expected_samples,
            'remaining': remaining,
            'completion_pct': completion_pct,
            'file_exists': os.path.exists(file_path)
        }
    
    return progress

def print_progress(progress: Dict[str, Dict[str, Any]]):
    """Print formatted progress information"""
    print("\n" + "="*80)
    print("GLM4V PROGRESS MONITOR")
    print("="*80)
    
    total_expected = 0
    total_current = 0
    total_remaining = 0
    
    print(f"{'Dataset':<10} {'Current':<10} {'Expected':<10} {'Remaining':<10} {'Completion':<12} {'Status':<10}")
    print("-" * 80)
    
    for dataset, info in progress.items():
        status = "✅ Complete" if info['remaining'] == 0 else "🔄 Running" if info['file_exists'] else "❌ Not Started"
        print(f"{dataset:<10} {info['current']:<10} {info['expected']:<10} {info['remaining']:<10} {info['completion_pct']:>8.1f}% {status:<10}")
        
        total_expected += info['expected']
        total_current += info['current']
        total_remaining += info['remaining']
    
    print("-" * 80)
    overall_completion = (total_current / total_expected) * 100 if total_expected > 0 else 0
    print(f"{'TOTAL':<10} {total_current:<10} {total_expected:<10} {total_remaining:<10} {overall_completion:>8.1f}%")
    
    # Show detailed breakdown
    print(f"\nDetailed Breakdown:")
    for dataset, info in progress.items():
        if info['remaining'] > 0:
            print(f"  {dataset}: {info['remaining']} samples remaining")
    
    if total_remaining == 0:
        print(f"\n🎉 All datasets completed!")
    else:
        print(f"\n⏳ {total_remaining} total samples remaining across all datasets")

def main():
    output_dir = "/scratch/xc1490/projects/medvqa_2025/output/glm4v"
    
    if not os.path.exists(output_dir):
        print(f"Output directory not found: {output_dir}")
        return
    
    try:
        while True:
            progress = get_current_progress(output_dir)
            print_progress(progress)
            
            # Check if all datasets are complete
            total_remaining = sum(info['remaining'] for info in progress.values())
            if total_remaining == 0:
                print(f"\n🎉 All GLM4V datasets are complete!")
                break
            
            print(f"\nNext update in 60 seconds... (Press Ctrl+C to stop)")
            time.sleep(60)
            
    except KeyboardInterrupt:
        print(f"\nMonitoring stopped by user")

if __name__ == "__main__":
    main()


