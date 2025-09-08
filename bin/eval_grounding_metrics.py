import os
import json
import argparse
from typing import List, Tuple, Optional, Dict
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
try:
    import scienceplots  # type: ignore
    plt.style.use(["science", "no-latex"])  # no LaTeX required
except Exception:
    plt.style.use("seaborn-v0_8-whitegrid")


def _area(x1: int, y1: int, x2: int, y2: int) -> int:
    w = max(0, x2 - x1)
    h = max(0, y2 - y1)
    return w * h


def _iou_and_dice(a: List[int], b: List[int]) -> Tuple[float, float]:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter = _area(inter_x1, inter_y1, inter_x2, inter_y2)
    area_a = _area(ax1, ay1, ax2, ay2)
    area_b = _area(bx1, by1, bx2, by2)
    union = max(1e-9, area_a + area_b - inter)
    iou = inter / union
    denom = max(1e-9, area_a + area_b)
    dice = 2.0 * inter / denom
    return float(iou), float(dice)


def _valid_box(b: object) -> Optional[List[int]]:
    if not isinstance(b, (list, tuple)) or len(b) < 4:
        return None
    try:
        x1, y1, x2, y2 = [int(round(float(v))) for v in b[:4]]
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1
        return [x1, y1, x2, y2]
    except Exception:
        return None


def read_jsonl(file_path: str):
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


def evaluate_file(file_path: str) -> Tuple[List[float], List[float]]:
    ious: List[float] = []
    dices: List[float] = []
    for obj in read_jsonl(file_path):
        pred = _valid_box(obj.get('pred_bbox'))
        gt = _valid_box(obj.get('gt_bbox_xyxy'))
        if pred is None or gt is None:
            continue
        iou, dice = _iou_and_dice(pred, gt)
        ious.append(iou)
        dices.append(dice)
    return ious, dices


def discover_model_files(root_dir: str, use_dedup: bool) -> Dict[str, str]:
    files: Dict[str, str] = {}
    for name in sorted(os.listdir(root_dir)):
        if not name.endswith('.jsonl'):
            continue
        base = name[:-6]  # drop .jsonl
        if use_dedup:
            if base.endswith('.dedup'):
                files[base[:-6]] = os.path.join(root_dir, name)  # key without .dedup
        else:
            if not base.endswith('.dedup'):
                files[base] = os.path.join(root_dir, name)
    if use_dedup and not files:
        # fallback if no dedup files exist, keep behavior consistent
        for name in sorted(os.listdir(root_dir)):
            if name.endswith('.jsonl') and not name.endswith('.dedup.jsonl'):
                files[name[:-6]] = os.path.join(root_dir, name)
    return files


def main():
    parser = argparse.ArgumentParser(description='Compute IoU and Dice per line for grounding outputs')
    parser.add_argument('--dir', type=str,
                        default='/home/xc1490/xc1490/projects/medvqa_2025/output/grounding/dedup/',
                        help='Directory containing grounding JSONL outputs')
    parser.add_argument('--use_dedup', action='store_true',
                        help='Prefer *.dedup.jsonl when present')
    parser.add_argument('--output', type=str,
                        default='/home/xc1490/xc1490/projects/medvqa_2025/output/grounding/metrics.json',
                        help='Output JSON filepath')
    parser.add_argument('--plots', action='store_true',
                        help='Also save per-model histograms and overall bar plot (dedup only)')
    args = parser.parse_args()

    # 1) Compute metrics JSON, using discover preference (may include non-dedup if requested)
    model_files = discover_model_files(args.dir, args.use_dedup)
    print (f"Found { model_files } model files")
    results: Dict[str, Dict[str, List[float]]] = {}
    means_iou: Dict[str, float] = {}
    means_dice: Dict[str, float] = {}
    for model_key, path in model_files.items():
        try:
            ious, dices = evaluate_file(path)
        except Exception:
            ious, dices = [], []
        results[model_key] = {'IoU': ious, 'DICE': dices}
        means_iou[model_key] = float(np.mean(ious)) if ious else 0.0
        means_dice[model_key] = float(np.mean(dices)) if dices else 0.0

    # Add new models data to results
    new_models_data = {
        "GLM-4.5V": {'IoU': [0.4062], 'DICE': [0.4925]},
        "GPT-5": {'IoU': [0.2152], 'DICE': [0.3146]},
        "Claude 4.1 Opus": {'IoU': [0.1699], 'DICE': [0.2437]}
    }
    results.update(new_models_data)
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"[Done] Wrote metrics to {args.output}")

    # 2) Plots strictly from dedup files only
    if args.plots:
        images_dir = os.path.join(os.path.dirname(args.output), 'images')
        os.makedirs(images_dir, exist_ok=True)

        # Discover JSONLs in the provided directory (assumed already deduped)
        dedup_files: Dict[str, str] = {}
        for fname in sorted(os.listdir(args.dir)):
            if fname.endswith('.jsonl'):
                key = fname[:-len('.jsonl')]
                dedup_files[key] = os.path.join(args.dir, fname)

        dedup_model_names = sorted(dedup_files.keys())
        iou_means: List[float] = []
        dice_means: List[float] = []

        # Per-model histograms (dedup only) and collect means
        print (f"Plotting histograms for { dedup_model_names } models")
        for model_key in dedup_model_names:
            path = dedup_files[model_key]
            ious, dices = evaluate_file(path)
            # Histograms
            for metric_name, data in [('IoU', ious), ('DICE', dices)]:
                if not data:
                    continue
                mean_val = float(np.mean(data))
                plt.figure(figsize=(6, 4), dpi=150)
                plt.hist(data, bins=30, range=(0, 1), color='#4C78A8', edgecolor='white')
                plt.xlabel(metric_name)
                plt.ylabel('Count')
                plt.title(f'{model_key} {metric_name} distribution (mean={mean_val:.3f})')
                out_path = os.path.join(images_dir, f'{model_key}_{metric_name.lower()}_hist.png')
                plt.tight_layout()
                plt.savefig(out_path, dpi=150)
                plt.close()
            iou_means.append(float(np.mean(ious)) if ious else 0.0)
            dice_means.append(float(np.mean(dices)) if dices else 0.0)

        # Add new models data from external results
        new_models_data = {
            "GLM-4.5V": {'IoU': [0.4062], 'DICE': [0.4925]},
            "GPT-5": {'IoU': [0.2152], 'DICE': [0.3146]},
            "Claude 4.1 Opus": {'IoU': [0.1699], 'DICE': [0.2437]}
        }
        
        # Combine existing and new models
        all_model_names = dedup_model_names + list(new_models_data.keys())
        all_iou_means = iou_means + [float(np.mean(new_models_data[model]['IoU'])) for model in new_models_data.keys()]
        all_dice_means = dice_means + [float(np.mean(new_models_data[model]['DICE'])) for model in new_models_data.keys()]

        # Sort separately for each metric and draw stacked bar plots
        iou_order = np.argsort(all_iou_means)[::-1]
        dice_order = np.argsort(all_dice_means)[::-1]

        fig, axes = plt.subplots(2, 1, figsize=(max(12, len(all_model_names) * 1.2), 10), dpi=150)

        # IoU subplot
        names_iou = [all_model_names[i] for i in iou_order]
        vals_iou = [all_iou_means[i] for i in iou_order]
        x_iou = np.arange(len(names_iou))
        
        bars_iou = axes[0].bar(x_iou, vals_iou, color='#4C78A8')
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
        names_dice = [all_model_names[i] for i in dice_order]
        vals_dice = [all_dice_means[i] for i in dice_order]
        x_dice = np.arange(len(names_dice))
        
        bars_dice = axes[1].bar(x_dice, vals_dice, color='#F58518')
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
        

        out_bar = os.path.join(images_dir, 'models_mean_iou_dice.png')
        fig.tight_layout()
        fig.savefig(out_bar, dpi=150)
        plt.close(fig)


if __name__ == '__main__':
    main()


