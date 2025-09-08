import json
import re
from typing import List
from evaluate_vqa import simple_tokenize, extract_closed_answer, is_closed_question

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

def debug_open_recall():
    """Debug why open-set recall is so high"""
    
    thyme_file = "/home/xc1490/xc1490/projects/medvqa_2025/output/thyme/test_pvqa.json"
    with open(thyme_file, 'r', encoding='utf-8') as f:
        thyme_data = json.load(f)
    
    open_samples = []
    
    for item in thyme_data:
        gt_answer = item.get('answer', '')
        pred_response = item.get('assistant_response', '')
        
        # Extract thyme answer
        if '<answer>' in pred_response:
            answer_match = re.search(r'<answer>(.*?)</answer>', pred_response, re.DOTALL)
            if answer_match:
                pred_answer = answer_match.group(1).strip()
            else:
                pred_answer = pred_response
        else:
            pred_answer = pred_response
        
        is_closed = is_closed_question(gt_answer)
        
        if not is_closed:
            open_samples.append({
                'gt_answer': gt_answer,
                'pred_answer': pred_answer,
                'question': item.get('question', ''),
                'id': item.get('id', '')
            })
    
    print(f"Total open samples: {len(open_samples)}")
    
    # Check first 20 open samples
    print("\n" + "="*80)
    print("OPEN SAMPLES (first 20):")
    print("="*80)
    
    recalls = []
    for i, sample in enumerate(open_samples[:20]):
        gt_answer = sample['gt_answer']
        pred_answer = sample['pred_answer']
        
        recall = calculate_open_recall(pred_answer, gt_answer)
        recalls.append(recall)
        
        print(f"Sample {i+1}:")
        print(f"  Question: {sample['question'][:100]}...")
        print(f"  GT Answer: '{gt_answer}'")
        print(f"  Pred Answer: '{pred_answer}'")
        print(f"  Recall: {recall:.3f}")
        print()
    
    print(f"Average recall (first 20): {sum(recalls)/len(recalls):.3f}")
    
    # Check all open samples
    all_recalls = []
    for sample in open_samples:
        gt_answer = sample['gt_answer']
        pred_answer = sample['pred_answer']
        recall = calculate_open_recall(pred_answer, gt_answer)
        all_recalls.append(recall)
    
    print(f"Average recall (all): {sum(all_recalls)/len(all_recalls):.3f}")
    print(f"Min recall: {min(all_recalls):.3f}")
    print(f"Max recall: {max(all_recalls):.3f}")
    
    # Check distribution
    high_recall = sum(1 for r in all_recalls if r >= 0.8)
    print(f"Samples with recall >= 0.8: {high_recall}/{len(all_recalls)} ({high_recall/len(all_recalls)*100:.1f}%)")

if __name__ == "__main__":
    debug_open_recall()

