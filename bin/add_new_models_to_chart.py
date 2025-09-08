#!/usr/bin/env python3
"""
Script to add new model results (GLM-4.5V, GPT-5, Claude 4.1 Opus) to the existing grounding metrics chart.
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
try:
    import scienceplots  # type: ignore
    plt.style.use(["science", "no-latex"])  # no LaTeX required
except Exception:
    plt.style.use("seaborn-v0_8-whitegrid")


def load_existing_metrics(metrics_file: str) -> dict:
    """Load existing metrics from JSON file."""
    if os.path.exists(metrics_file):
        with open(metrics_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


def add_new_models_data(existing_metrics: dict) -> dict:
    """Add the new model data from the image to existing metrics."""
    # Data from the image: GLM-4.5V, GPT-5, Claude 4.1 Opus
    new_models_data = {
        "GLM-4.5V": {
            "IoU": [0.4062],  # Single value representing the mean
            "DICE": [0.4925]
        },
        "GPT-5": {
            "IoU": [0.2152],
            "DICE": [0.3146]
        },
        "Claude 4.1 Opus": {
            "IoU": [0.1699],
            "DICE": [0.2437]
        }
    }
    
    # Merge with existing metrics
    updated_metrics = existing_metrics.copy()
    updated_metrics.update(new_models_data)
    
    return updated_metrics


def create_updated_chart(metrics: dict, output_path: str):
    """Create the updated chart with all models including new ones."""
    # Extract model names and calculate means
    model_names = sorted(metrics.keys())
    iou_means = []
    dice_means = []
    
    for model_name in model_names:
        iou_data = metrics[model_name].get('IoU', [])
        dice_data = metrics[model_name].get('DICE', [])
        
        iou_mean = float(np.mean(iou_data)) if iou_data else 0.0
        dice_mean = float(np.mean(dice_data)) if dice_data else 0.0
        
        iou_means.append(iou_mean)
        dice_means.append(dice_mean)
    
    # Sort separately for each metric
    iou_order = np.argsort(iou_means)[::-1]
    dice_order = np.argsort(dice_means)[::-1]
    
    # Create the chart
    fig, axes = plt.subplots(2, 1, figsize=(max(12, len(model_names) * 1.2), 10), dpi=150)
    
    # IoU subplot
    names_iou = [model_names[i] for i in iou_order]
    vals_iou = [iou_means[i] for i in iou_order]
    x_iou = np.arange(len(names_iou))
    
    # Color coding: highlight new models in different colors
    colors_iou = []
    for name in names_iou:
        if name in ["GLM-4.5V", "GPT-5", "Claude 4.1 Opus"]:
            colors_iou.append('#E74C3C')  # Red for new models
        else:
            colors_iou.append('#4C78A8')  # Blue for existing models
    
    bars_iou = axes[0].bar(x_iou, vals_iou, color=colors_iou)
    axes[0].set_ylabel('IoU', fontsize=12)
    axes[0].set_title('Mean IoU per model', fontsize=14, fontweight='bold')
    axes[0].set_xticks(x_iou)
    axes[0].set_xticklabels(names_iou, rotation=45, ha='right', fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars_iou, vals_iou)):
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    
    # DICE subplot
    names_dice = [model_names[i] for i in dice_order]
    vals_dice = [dice_means[i] for i in dice_order]
    x_dice = np.arange(len(names_dice))
    
    # Color coding: highlight new models in different colors
    colors_dice = []
    for name in names_dice:
        if name in ["GLM-4.5V", "GPT-5", "Claude 4.1 Opus"]:
            colors_dice.append('#E74C3C')  # Red for new models
        else:
            colors_dice.append('#F58518')  # Orange for existing models
    
    bars_dice = axes[1].bar(x_dice, vals_dice, color=colors_dice)
    axes[1].set_ylabel('DICE', fontsize=12)
    axes[1].set_title('Mean DICE per model', fontsize=14, fontweight='bold')
    axes[1].set_xticks(x_dice)
    axes[1].set_xticklabels(names_dice, rotation=45, ha='right', fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars_dice, vals_dice)):
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#4C78A8', label='Existing Models'),
        Patch(facecolor='#E74C3C', label='New Models (GLM-4.5V, GPT-5, Claude 4.1 Opus)')
    ]
    fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98))
    
    # Save the chart
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Updated chart saved to: {output_path}")
    print(f"Total models in chart: {len(model_names)}")
    print("New models added: GLM-4.5V, GPT-5, Claude 4.1 Opus")


def main():
    # File paths
    metrics_file = '/home/xc1490/xc1490/projects/medvqa_2025/output/grounding/metrics.json'
    output_chart = '/home/xc1490/xc1490/projects/medvqa_2025/output/grounding/images/models_mean_iou_dice.png'
    
    # Load existing metrics
    print("Loading existing metrics...")
    existing_metrics = load_existing_metrics(metrics_file)
    print(f"Found {len(existing_metrics)} existing models")
    
    # Add new models data
    print("Adding new models data...")
    updated_metrics = add_new_models_data(existing_metrics)
    
    # Save updated metrics
    updated_metrics_file = '/home/xc1490/xc1490/projects/medvqa_2025/output/grounding/metrics_updated.json'
    with open(updated_metrics_file, 'w', encoding='utf-8') as f:
        json.dump(updated_metrics, f, ensure_ascii=False, indent=2)
    print(f"Updated metrics saved to: {updated_metrics_file}")
    
    # Create updated chart
    print("Creating updated chart...")
    create_updated_chart(updated_metrics, output_chart)
    
    print("Done!")


if __name__ == '__main__':
    main()

