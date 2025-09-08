import os
import json
import re
from typing import Dict, List, Any
import matplotlib.pyplot as plt
import numpy as np
from evaluate_vqa import simple_tokenize, extract_closed_answer, is_closed_question, evaluate_vqa

def extract_answer_from_tags(assistant_response: str) -> str:
    """Extract answer from assistant_response which contains <answer> tags"""
    if not assistant_response:
        return ""
    
    # Look for <answer>...</answer> pattern (case insensitive)
    answer_match = re.search(r'<answer>(.*?)</answer>', assistant_response, re.DOTALL | re.IGNORECASE)
    if answer_match:
        answer = answer_match.group(1).strip()
        # Clean up common artifacts
        answer = re.sub(r'^\s*[-*]\s*', '', answer)  # Remove leading bullets
        answer = re.sub(r'\s+', ' ', answer)  # Normalize whitespace
        return answer
    
    # If no <answer> tags found, return empty string (don't fallback to full response)
    return ""

def extract_model_response(item: Dict[str, Any], stats: Dict[str, int] = None) -> str:
    """Extract model response from various field names"""
    # First priority: if model_answer exists, use it (this is the extracted answer)
    if 'model_answer' in item and item['model_answer']:
        if stats: stats['model_answer'] += 1
        return item['model_answer']
    
    # Second priority: extract from assistant_response if it contains <answer> tags
    if 'assistant_response' in item and item['assistant_response']:
        response = item['assistant_response']
        
        # If assistant_response contains <answer> tags, extract from them
        if '<answer>' in response:
            extracted_answer = extract_answer_from_tags(response)
            if extracted_answer:
                if stats: stats['answer_tags'] += 1
                return extracted_answer
        
        # If no <answer> tags or extraction failed, use the full response
        if stats: stats['full_response'] += 1
        return response
    
    # Fallback: try other possible field names
    for field in ['response', 'prediction', 'output']:
        if field in item and item[field]:
            if stats: stats['full_response'] += 1
            return item[field]
    
    if stats: stats['empty'] += 1
    return ""

def extract_ground_truth(item: Dict[str, Any]) -> str:
    """Extract ground truth from various field names"""
    # Try different possible field names for ground truth
    for field in ['answer', 'gt_answer', 'ground_truth', 'reference']:
        if field in item and item[field]:
            return item[field]
    
    return ""

def calculate_open_recall(pred: str, ref: str) -> float:
    """Calculate recall for open-ended questions"""
    if not pred or not ref:
        return 0.0
    
    # Simple token-based recall
    pred_tokens = set(simple_tokenize(pred))
    ref_tokens = set(simple_tokenize(ref))
    
    if not ref_tokens:
        return 0.0
    
    # Calculate recall
    intersection = pred_tokens & ref_tokens
    recall = len(intersection) / len(ref_tokens)
    
    return recall

def evaluate_vqa_file(test_json_path: str, model_name: str, dataset_name: str) -> Dict[str, Any]:
    """Evaluate a single VQA file and return results"""
    closed_correct = 0
    closed_total = 0
    open_recalls = []
    total = 0
    wrong_samples = []
    extraction_stats = {'model_answer': 0, 'answer_tags': 0, 'full_response': 0, 'empty': 0}
    
    try:
        with open(test_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Expected dataset sizes (after removing duplicates)
        # These are the actual unique samples that should be processed
        expected_sizes = {
            'pvqa': 6176,  # 6761 - 585 duplicates
            'slake': 1061,
            'rad': 451
        }
        expected_total = expected_sizes.get(dataset_name, len(data))
            
        for item in data:
            pred = extract_model_response(item, extraction_stats)
            ref = extract_ground_truth(item)
            
            if not pred or not ref:
                continue
                
            total += 1
            
            if is_closed_question(ref):
                closed_total += 1
                pred_ans = extract_closed_answer(pred)
                ref_ans = extract_closed_answer(ref) if extract_closed_answer(ref) else ref.strip().lower()
                
                if pred_ans is not None and pred_ans == ref_ans.lower():
                    closed_correct += 1
                else:
                    wrong_samples.append({
                        'model': model_name,
                        'dataset': dataset_name,
                        'type': 'closed',
                        'predicted': pred,
                        'predicted_extracted': pred_ans,
                        'gt_answer': ref,
                        'gt_extracted': ref_ans,
                        'question': item.get('question', None),
                        'image': item.get('image', None),
                        'id': item.get('id', None)
                    })
            else:
                recall = calculate_open_recall(pred, ref)
                open_recalls.append(recall)
                
                if recall < 0.1:
                    wrong_samples.append({
                        'model': model_name,
                        'dataset': dataset_name,
                        'type': 'open',
                        'recall': recall,
                        'predicted': pred,
                        'gt_answer': ref,
                        'question': item.get('question', None),
                        'image': item.get('image', None),
                        'id': item.get('id', None)
                    })
        
        closed_acc = closed_correct / closed_total if closed_total > 0 else None
        open_recall = sum(open_recalls) / len(open_recalls) if open_recalls else None
        
        return {
            'model': model_name,
            'dataset': dataset_name,
            'total_samples': total,
            'total_processed': len(data),  # Total samples in the file
            'expected_total': expected_total,
            'closed_total': closed_total,
            'closed_correct': closed_correct,
            'closed_accuracy': closed_acc,
            'open_total': len(open_recalls),
            'open_recall': open_recall,
            'extraction_stats': extraction_stats,
            'wrong_samples': wrong_samples
        }
        
    except Exception as e:
        print(f"Error evaluating {test_json_path}: {e}")
        return {
            'model': model_name,
            'dataset': dataset_name,
            'error': str(e)
        }

def calculate_model_overall_score(model_data: Dict[str, Dict[str, Any]]) -> float:
    """Calculate overall score for a model based on all datasets"""
    total_closed_correct = 0
    total_closed = 0
    total_open_recalls = []
    
    for dataset_name, result in model_data.items():
        if result['closed_accuracy'] is not None:
            total_closed_correct += result['closed_correct']
            total_closed += result['closed_total']
        if result['open_recall'] is not None:
            total_open_recalls.extend([result['open_recall']] * result['open_total'])
    
    overall_closed_acc = total_closed_correct / total_closed if total_closed > 0 else 0.0
    overall_open_recall = sum(total_open_recalls) / len(total_open_recalls) if total_open_recalls else 0.0
    
    # Combine closed accuracy and open recall (equal weight)
    overall_score = (overall_closed_acc + overall_open_recall) / 2.0
    return overall_score

def print_results(results: List[Dict[str, Any]]):
    """Print results in separated tables for closed and open questions"""
    print("\n" + "="*80)
    print("VQA EVALUATION RESULTS")
    print("="*80)
    
    # Group results by model
    models = {}
    for result in results:
        if 'error' in result:
            continue
        model = result['model']
        if model not in models:
            models[model] = {}
        models[model][result['dataset']] = result
    
    # Expected dataset sizes (after removing duplicates)
    expected_sizes = {
        'pvqa': 6176,  # 6761 - 585 duplicates
        'slake': 1061,
        'rad': 451
    }
    
    # Calculate overall scores and sort models
    model_scores = {}
    model_completions = {}
    for model_name, datasets in models.items():
        overall_score = calculate_model_overall_score(datasets)
        model_scores[model_name] = overall_score
        
        # Calculate completion percentage
        total_processed = 0
        total_expected = 0
        for dataset_name in ['pvqa', 'slake', 'rad']:
            if dataset_name in datasets:
                result = datasets[dataset_name]
                total_processed += result['total_processed']
                total_expected += result['expected_total']
            else:
                total_expected += expected_sizes.get(dataset_name, 0)
        
        completion = (total_processed / total_expected) * 100 if total_expected > 0 else 0
        model_completions[model_name] = completion
    
    # Sort models by overall score (descending), but only include those with >= 95% completion
    sorted_models = []
    filtered_models = []
    for model_name, score in model_scores.items():
        if model_completions[model_name] >= 95.0:
            sorted_models.append((model_name, score))
        else:
            filtered_models.append((model_name, model_completions[model_name]))
    
    sorted_models.sort(key=lambda x: x[1], reverse=True)
    
    # Print filtering information
    total_models = len(model_scores)
    shown_models = len(sorted_models)
    filtered_count = len(filtered_models)
    
    print(f"\nShowing {shown_models}/{total_models} models with ≥95% completion")
    if filtered_count > 0:
        print(f"Filtered out {filtered_count} models with <95% completion:")
        for model_name, completion in filtered_models:
            print(f"  {model_name}: {completion:.1f}%")
    
    print("\n" + "="*60)
    print("MODEL RANKING BY OVERALL PERFORMANCE (≥95% COMPLETION)")
    print("="*60)
    print(f"{'Rank':<5} {'Model':<20} {'Overall Score':<15}")
    print("-" * 60)
    
    for rank, (model_name, score) in enumerate(sorted_models, 1):
        print(f"{rank:<5} {model_name:<20} {score:.3f}")
    
    # Print closed questions table (sorted by model performance)
    print("\n" + "="*60)
    print("CLOSED QUESTIONS ACCURACY (SORTED BY MODEL PERFORMANCE)")
    print("="*60)
    print(f"{'Model':<15} {'Dataset':<10} {'Total':<8} {'Correct':<8} {'Accuracy':<10}")
    print("-" * 60)
    
    for model_name, _ in sorted_models:
        if model_name in models:
            for dataset_name, result in models[model_name].items():
                if result['closed_total'] > 0:
                    closed_acc_str = f"{result['closed_accuracy']:.3f}" if result['closed_accuracy'] is not None else "N/A"
                    print(f"{model_name:<15} {dataset_name:<10} {result['closed_total']:<8} "
                          f"{result['closed_correct']:<8} {closed_acc_str:<10}")
    
    # Print open questions table (sorted by model performance)
    print("\n" + "="*60)
    print("OPEN QUESTIONS RECALL (SORTED BY MODEL PERFORMANCE)")
    print("="*60)
    print(f"{'Model':<15} {'Dataset':<10} {'Total':<8} {'Avg Recall':<12}")
    print("-" * 60)
    
    for model_name, _ in sorted_models:
        if model_name in models:
            for dataset_name, result in models[model_name].items():
                if result['open_total'] > 0:
                    open_recall_str = f"{result['open_recall']:.3f}" if result['open_recall'] is not None else "N/A"
                    print(f"{model_name:<15} {dataset_name:<10} {result['open_total']:<8} {open_recall_str:<12}")
    
    # Print summary by model (sorted by performance)
    print("\n" + "="*80)
    print("SUMMARY BY MODEL (SORTED BY PERFORMANCE)")
    print("="*80)
    
    if not sorted_models:
        print("No models with ≥95% completion found.")
        return models, sorted_models
    
    for model_name, overall_score in sorted_models:
        if model_name in models:
            print(f"\n{model_name} (Overall Score: {overall_score:.3f}):")
            total_closed_correct = 0
            total_closed = 0
            total_open_recalls = []
            
            for dataset_name, result in models[model_name].items():
                if result['closed_accuracy'] is not None:
                    total_closed_correct += result['closed_correct']
                    total_closed += result['closed_total']
                if result['open_recall'] is not None:
                    total_open_recalls.extend([result['open_recall']] * result['open_total'])
            
            overall_closed_acc = total_closed_correct / total_closed if total_closed > 0 else None
            overall_open_recall = sum(total_open_recalls) / len(total_open_recalls) if total_open_recalls else None
            
            closed_acc_str = f"{overall_closed_acc:.3f}" if overall_closed_acc is not None else "N/A"
            open_recall_str = f"{overall_open_recall:.3f}" if overall_open_recall is not None else "N/A"
            
            print(f"  Overall Closed Accuracy: {closed_acc_str}")
            print(f"  Overall Open Recall: {open_recall_str}")
    
    return models, sorted_models

def create_bar_plots(models: Dict[str, Dict[str, Dict[str, Any]]], sorted_models: List[tuple], output_dir: str):
    """Create bar plots for closed accuracy and open recall"""
    # Ensure images directory exists
    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    
    # Set up matplotlib for better looking plots
    plt.style.use('default')
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 10
    
    # Get sorted model names and datasets
    model_names = [model_name for model_name, _ in sorted_models]
    datasets = ['pvqa', 'slake', 'rad']
    
    # Create closed questions accuracy plot
    fig, ax = plt.subplots(figsize=(14, 8))
    
    x = np.arange(len(datasets))
    width = 0.8 / len(model_names)
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(model_names)))
    
    for i, model_name in enumerate(model_names):
        accuracies = []
        for dataset in datasets:
            if dataset in models[model_name] and models[model_name][dataset]['closed_accuracy'] is not None:
                accuracies.append(models[model_name][dataset]['closed_accuracy'])
            else:
                accuracies.append(0)
        
        bars = ax.bar(x + i * width, accuracies, width, label=model_name, 
                     color=colors[i], alpha=0.8, edgecolor='black', linewidth=0.5)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel('Dataset', fontweight='bold')
    ax.set_ylabel('Accuracy', fontweight='bold')
    ax.set_title('Closed Questions Accuracy by Model and Dataset', fontweight='bold', fontsize=14)
    ax.set_xticks(x + width * (len(model_names) - 1) / 2)
    ax.set_xticklabels(datasets)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.set_ylim(0, 1.1)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    closed_plot_path = os.path.join(images_dir, "closed_questions_accuracy.png")
    plt.savefig(closed_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create open questions recall plot
    fig, ax = plt.subplots(figsize=(14, 8))
    
    for i, model_name in enumerate(model_names):
        recalls = []
        for dataset in datasets:
            if dataset in models[model_name] and models[model_name][dataset]['open_recall'] is not None:
                recalls.append(models[model_name][dataset]['open_recall'])
            else:
                recalls.append(0)
        
        bars = ax.bar(x + i * width, recalls, width, label=model_name, 
                     color=colors[i], alpha=0.8, edgecolor='black', linewidth=0.5)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel('Dataset', fontweight='bold')
    ax.set_ylabel('Recall', fontweight='bold')
    ax.set_title('Open Questions Recall by Model and Dataset', fontweight='bold', fontsize=14)
    ax.set_xticks(x + width * (len(model_names) - 1) / 2)
    ax.set_xticklabels(datasets)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.set_ylim(0, 1.1)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    open_plot_path = os.path.join(images_dir, "open_questions_recall.png")
    plt.savefig(open_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create overall performance plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    
    # Overall closed accuracy
    overall_closed_accs = []
    for model_name in model_names:
        total_closed_correct = 0
        total_closed = 0
        for dataset_name, result in models[model_name].items():
            if result['closed_accuracy'] is not None:
                total_closed_correct += result['closed_correct']
                total_closed += result['closed_total']
        
        overall_closed_acc = total_closed_correct / total_closed if total_closed > 0 else 0
        overall_closed_accs.append(overall_closed_acc)
    
    bars1 = ax1.bar(model_names, overall_closed_accs, color=colors[:len(model_names)], 
                    alpha=0.8, edgecolor='black', linewidth=0.5)
    
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax1.set_ylabel('Overall Accuracy', fontweight='bold')
    ax1.set_title('Overall Closed Questions Accuracy', fontweight='bold', fontsize=14)
    ax1.set_ylim(0, 1.1)
    ax1.grid(axis='y', alpha=0.3)
    plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
    
    # Overall open recall
    overall_open_recalls = []
    for model_name in model_names:
        total_open_recalls = []
        for dataset_name, result in models[model_name].items():
            if result['open_recall'] is not None:
                total_open_recalls.extend([result['open_recall']] * result['open_total'])
        
        overall_open_recall = sum(total_open_recalls) / len(total_open_recalls) if total_open_recalls else 0
        overall_open_recalls.append(overall_open_recall)
    
    bars2 = ax2.bar(model_names, overall_open_recalls, color=colors[:len(model_names)], 
                    alpha=0.8, edgecolor='black', linewidth=0.5)
    
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax2.set_ylabel('Overall Recall', fontweight='bold')
    ax2.set_title('Overall Open Questions Recall', fontweight='bold', fontsize=14)
    ax2.set_ylim(0, 1.1)
    ax2.grid(axis='y', alpha=0.3)
    plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    overall_plot_path = os.path.join(images_dir, "overall_performance.png")
    plt.savefig(overall_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create grouped by model plots (each model as a group, datasets as bars within group)
    fig, ax = plt.subplots(figsize=(16, 8))
    
    x = np.arange(len(model_names))
    width = 0.25  # width of each bar
    
    dataset_colors = {'pvqa': '#FF6B6B', 'slake': '#4ECDC4', 'rad': '#45B7D1'}
    
    for i, dataset in enumerate(datasets):
        accuracies = []
        for model_name in model_names:
            if dataset in models[model_name] and models[model_name][dataset]['closed_accuracy'] is not None:
                accuracies.append(models[model_name][dataset]['closed_accuracy'])
            else:
                accuracies.append(0)
        
        bars = ax.bar(x + i * width, accuracies, width, label=dataset.upper(), 
                     color=dataset_colors[dataset], alpha=0.8, edgecolor='black', linewidth=0.5)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=8, rotation=0)
    
    ax.set_xlabel('Model', fontweight='bold')
    ax.set_ylabel('Accuracy', fontweight='bold')
    ax.set_title('Closed Questions Accuracy - Grouped by Model', fontweight='bold', fontsize=14)
    ax.set_xticks(x + width)
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    closed_grouped_path = os.path.join(images_dir, "closed_questions_accuracy_grouped.png")
    plt.savefig(closed_grouped_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create grouped by model plot for open questions
    fig, ax = plt.subplots(figsize=(16, 8))
    
    for i, dataset in enumerate(datasets):
        recalls = []
        for model_name in model_names:
            if dataset in models[model_name] and models[model_name][dataset]['open_recall'] is not None:
                recalls.append(models[model_name][dataset]['open_recall'])
            else:
                recalls.append(0)
        
        bars = ax.bar(x + i * width, recalls, width, label=dataset.upper(), 
                     color=dataset_colors[dataset], alpha=0.8, edgecolor='black', linewidth=0.5)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=8, rotation=0)
    
    ax.set_xlabel('Model', fontweight='bold')
    ax.set_ylabel('Recall', fontweight='bold')
    ax.set_title('Open Questions Recall - Grouped by Model', fontweight='bold', fontsize=14)
    ax.set_xticks(x + width)
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    open_grouped_path = os.path.join(images_dir, "open_questions_recall_grouped.png")
    plt.savefig(open_grouped_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create overall performance plot with grouped bars (each model has 3 bars for 3 datasets)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Overall closed accuracy with grouped bars
    x = np.arange(len(model_names))
    width = 0.25
    dataset_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']  # Distinct colors for datasets
    
    for i, dataset in enumerate(datasets):
        accuracies = []
        for model_name in model_names:
            if dataset in models[model_name] and models[model_name][dataset]['closed_accuracy'] is not None:
                accuracies.append(models[model_name][dataset]['closed_accuracy'])
            else:
                accuracies.append(0)
        
        bars = ax1.bar(x + i * width, accuracies, width, label=dataset.upper(), 
                      color=dataset_colors[i], alpha=0.8, edgecolor='black', linewidth=0.5)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    ax1.set_xlabel('Model', fontweight='bold')
    ax1.set_ylabel('Accuracy', fontweight='bold')
    ax1.set_title('Overall Closed Questions Accuracy (Grouped by Model)', fontweight='bold', fontsize=14)
    ax1.set_xticks(x + width)
    ax1.set_xticklabels(model_names, rotation=45, ha='right')
    ax1.legend()
    ax1.set_ylim(0, 1.1)
    ax1.grid(axis='y', alpha=0.3)
    
    # Overall open recall with grouped bars
    for i, dataset in enumerate(datasets):
        recalls = []
        for model_name in model_names:
            if dataset in models[model_name] and models[model_name][dataset]['open_recall'] is not None:
                recalls.append(models[model_name][dataset]['open_recall'])
            else:
                recalls.append(0)
        
        bars = ax2.bar(x + i * width, recalls, width, label=dataset.upper(), 
                      color=dataset_colors[i], alpha=0.8, edgecolor='black', linewidth=0.5)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    ax2.set_xlabel('Model', fontweight='bold')
    ax2.set_ylabel('Recall', fontweight='bold')
    ax2.set_title('Overall Open Questions Recall (Grouped by Model)', fontweight='bold', fontsize=14)
    ax2.set_xticks(x + width)
    ax2.set_xticklabels(model_names, rotation=45, ha='right')
    ax2.legend()
    ax2.set_ylim(0, 1.1)
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    overall_grouped_path = os.path.join(images_dir, "overall_performance_grouped.png")
    plt.savefig(overall_grouped_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nSaved plots:")
    print(f"  Closed questions accuracy (by dataset): {closed_plot_path}")
    print(f"  Open questions recall (by dataset): {open_plot_path}")
    print(f"  Overall performance: {overall_plot_path}")
    print(f"  Closed questions accuracy (grouped by model): {closed_grouped_path}")
    print(f"  Open questions recall (grouped by model): {open_grouped_path}")
    print(f"  Overall performance (grouped by model): {overall_grouped_path}")

def save_wrong_samples(results: List[Dict[str, Any]], output_dir: str):
    """Save wrong samples to JSONL file"""
    all_wrong_samples = []
    for result in results:
        if 'wrong_samples' in result:
            all_wrong_samples.extend(result['wrong_samples'])
    
    if all_wrong_samples:
        output_path = os.path.join(output_dir, "all_wrong_samples.jsonl")
        with open(output_path, "w", encoding="utf-8") as f:
            for item in all_wrong_samples:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"\nSaved {len(all_wrong_samples)} wrong samples to {output_path}")

def print_completion_status(results: List[Dict[str, Any]]):
    """Print completion status for each model and dataset"""
    print("\n" + "="*80)
    print("DATASET COMPLETION STATUS")
    print("="*80)
    print("Note: Expected sizes account for duplicate samples that are automatically skipped during processing")
    print("PVQA: 6761 total samples -> 6176 unique samples (585 duplicates removed)")
    print("SLAKE: 1061 total samples -> 1061 unique samples (no duplicates)")
    print("RAD: 451 total samples -> 451 unique samples (no duplicates)")
    print("Note: 'Missing' includes both unprocessed samples and processed samples with empty responses")
    print("="*80)
    
    # Group results by model
    models = {}
    for result in results:
        if 'error' in result:
            continue
        model = result['model']
        if model not in models:
            models[model] = {}
        models[model][result['dataset']] = result
    
    # Expected dataset sizes (after removing duplicates)
    # These are the actual unique samples that should be processed
    expected_sizes = {
        'pvqa': 6176,  # 6761 - 585 duplicates
        'slake': 1061,
        'rad': 451
    }
    
    print(f"{'Model':<20} {'Dataset':<10} {'Processed':<10} {'Valid':<10} {'Expected':<10} {'Missing':<10} {'Completion %':<12}")
    print("-" * 90)
    
    for model_name in sorted(models.keys()):
        # Calculate overall completion for this model
        total_processed = 0
        total_expected = 0
        for dataset_name in ['pvqa', 'slake', 'rad']:
            if dataset_name in models[model_name]:
                result = models[model_name][dataset_name]
                total_processed += result['total_processed']
                total_expected += result['expected_total']
            else:
                total_expected += expected_sizes.get(dataset_name, 0)
        
        overall_completion = (total_processed / total_expected) * 100 if total_expected > 0 else 0
        
        # Show all models for completion analysis
        for dataset_name in ['pvqa', 'slake', 'rad']:
                if dataset_name in models[model_name]:
                    result = models[model_name][dataset_name]
                    processed = result['total_processed']  # Total samples in file
                    valid = result['total_samples']  # Samples with valid responses
                    expected = result['expected_total']
                    missing = expected - processed
                    completion_pct = (processed / expected) * 100 if expected > 0 else 0
                    
                    # Highlight incomplete runs
                    if missing > 0:
                        status_marker = "❌ "  # Missing samples
                    elif valid < processed:
                        status_marker = "⚠️ "  # Some samples have empty responses
                    else:
                        status_marker = "✅ "  # All samples processed with valid responses
                    print(f"{status_marker}{model_name:<17} {dataset_name:<10} {processed:<10} {valid:<10} {expected:<10} {missing:<10} {completion_pct:>8.1f}%")
                else:
                    # Dataset not found for this model
                    expected = expected_sizes.get(dataset_name, 0)
                    print(f"❌ {model_name:<17} {dataset_name:<10} {'0':<10} {'0':<10} {expected:<10} {expected:<10} {'0.0':>8}%")
    
    # Summary by model
    print("\n" + "="*60)
    print("COMPLETION SUMMARY BY MODEL")
    print("="*60)
    
    for model_name in sorted(models.keys()):
        total_processed = 0
        total_valid = 0
        total_expected = 0
        incomplete_datasets = []
        
        for dataset_name in ['pvqa', 'slake', 'rad']:
            if dataset_name in models[model_name]:
                result = models[model_name][dataset_name]
                processed = result['total_processed']
                valid = result['total_samples']
                expected = result['expected_total']
                total_processed += processed
                total_valid += valid
                total_expected += expected
                
                if processed < expected:
                    missing = expected - processed
                    incomplete_datasets.append(f"{dataset_name}(-{missing})")
                elif valid < processed:
                    empty_responses = processed - valid
                    incomplete_datasets.append(f"{dataset_name}({empty_responses} empty)")
            else:
                expected = expected_sizes.get(dataset_name, 0)
                total_expected += expected
                incomplete_datasets.append(f"{dataset_name}(missing)")
        
        overall_completion = (total_processed / total_expected) * 100 if total_expected > 0 else 0
        
        # Only show models with completion >= 95%
        if overall_completion >= 95.0:
            if incomplete_datasets:
                print(f"⚠️  {model_name:<20} {overall_completion:>6.1f}% complete - Issues: {', '.join(incomplete_datasets)}")
            else:
                print(f"✅ {model_name:<20} {overall_completion:>6.1f}% complete - All datasets complete")

def main():
    output_dir = "output"
    results = []
    
    # Find all model directories
    model_dirs = [d for d in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, d))]
    
    print(f"Found {len(model_dirs)} model directories: {model_dirs}")
    
    # Evaluate each model
    for model_name in model_dirs:
        model_dir = os.path.join(output_dir, model_name)
        
        # Find test files for this model
        test_files = []
        for file in os.listdir(model_dir):
            if file.startswith("test_") and file.endswith(".json"):
                dataset_name = file[5:-5]  # Remove "test_" prefix and ".json" suffix
                test_files.append((file, dataset_name))
        
        print(f"\nEvaluating {model_name} ({len(test_files)} test files)")
        
        for filename, dataset_name in test_files:
            file_path = os.path.join(model_dir, filename)
            print(f"  Processing {dataset_name}...")
            
            result = evaluate_vqa_file(file_path, model_name, dataset_name)
            results.append(result)
            
            if 'error' not in result:
                closed_acc_str = f"{result['closed_accuracy']:.3f}" if result['closed_accuracy'] is not None else "N/A"
                open_recall_str = f"{result['open_recall']:.3f}" if result['open_recall'] is not None else "N/A"
                print(f"    Closed: {result['closed_total']} samples, Accuracy: {closed_acc_str}")
                print(f"    Open: {result['open_total']} samples, Recall: {open_recall_str}")
                
                # Show extraction statistics
                stats = result.get('extraction_stats', {})
                if stats:
                    print(f"    Extraction: model_answer={stats.get('model_answer', 0)}, "
                          f"answer_tags={stats.get('answer_tags', 0)}, "
                          f"full_response={stats.get('full_response', 0)}, "
                          f"empty={stats.get('empty', 0)}")
            else:
                print(f"    Error: {result['error']}")
    
    # Print results and get models data
    models, sorted_models = print_results(results)
    
    # Print completion status
    print_completion_status(results)
    
    # Create bar plots
    create_bar_plots(models, sorted_models, output_dir)
    
    # Save wrong samples
    save_wrong_samples(results, output_dir)

if __name__ == "__main__":
    main()
