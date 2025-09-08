import json
import re
from typing import List
from evaluate_vqa import simple_tokenize, extract_closed_answer, is_closed_question

def debug_thyme_closed():
    """Debug thyme's closed questions specifically"""
    
    thyme_file = "/home/xc1490/xc1490/projects/medvqa_2025/output/thyme/test_pvqa.json"
    with open(thyme_file, 'r', encoding='utf-8') as f:
        thyme_data = json.load(f)
    
    closed_samples = []
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
        
        if is_closed:
            closed_samples.append({
                'gt_answer': gt_answer,
                'pred_answer': pred_answer,
                'question': item.get('question', ''),
                'id': item.get('id', '')
            })
        else:
            open_samples.append({
                'gt_answer': gt_answer,
                'pred_answer': pred_answer,
                'question': item.get('question', ''),
                'id': item.get('id', '')
            })
    
    print(f"Total samples: {len(thyme_data)}")
    print(f"Closed samples: {len(closed_samples)}")
    print(f"Open samples: {len(open_samples)}")
    
    print("\n" + "="*80)
    print("CLOSED SAMPLES:")
    print("="*80)
    
    correct_closed = 0
    for i, sample in enumerate(closed_samples[:20]):  # Show first 20
        gt_answer = sample['gt_answer']
        pred_answer = sample['pred_answer']
        
        # Extract closed answers
        gt_extracted = extract_closed_answer(gt_answer)
        pred_extracted = extract_closed_answer(pred_answer)
        
        is_correct = (pred_extracted is not None and 
                     gt_extracted is not None and 
                     pred_extracted == gt_extracted)
        
        if is_correct:
            correct_closed += 1
        
        print(f"Sample {i+1}:")
        print(f"  Question: {sample['question'][:100]}...")
        print(f"  GT Answer: '{gt_answer}'")
        print(f"  Pred Answer: '{pred_answer}'")
        print(f"  GT Extracted: '{gt_extracted}'")
        print(f"  Pred Extracted: '{pred_extracted}'")
        print(f"  Correct: {is_correct}")
        print()
    
    print(f"Correct closed samples (first 20): {correct_closed}/20")
    
    # Check all closed samples
    total_correct = 0
    for sample in closed_samples:
        gt_answer = sample['gt_answer']
        pred_answer = sample['pred_answer']
        
        gt_extracted = extract_closed_answer(gt_answer)
        pred_extracted = extract_closed_answer(pred_answer)
        
        if (pred_extracted is not None and 
            gt_extracted is not None and 
            pred_extracted == gt_extracted):
            total_correct += 1
    
    print(f"Total correct closed samples: {total_correct}/{len(closed_samples)}")
    print(f"Closed accuracy: {total_correct/len(closed_samples):.3f}")

if __name__ == "__main__":
    debug_thyme_closed()

