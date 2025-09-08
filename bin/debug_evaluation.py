import json
import re
from typing import List, Dict, Any

def simple_tokenize(text: str) -> List[str]:
    return re.findall(r'\w+', text.lower())

def extract_closed_answer(text: str) -> str:
    text = text.lower()
    closed_keywords = [
        'yes', 'no',
        'a', 'b', 'c', 'd',
        'male', 'female',
        'present', 'absent',
        'normal', 'abnormal',
        'left', 'right',
        'one', 'two', 'three', 'four', 'five',
        'smaller', 'larger', 'smaller than', 'larger than',
        'pa', 'ap',
    ]
    for kw in closed_keywords:
        if re.search(r'\b' + re.escape(kw) + r'\b', text):
            return kw
    m = re.search(r'\b([a-d])\b', text)
    if m:
        return m.group(1)
    return None

def is_closed_question(gt_answer: str) -> bool:
    tokens = simple_tokenize(gt_answer)
    closed_keywords = {'yes', 'no', 'male', 'female', 'normal', 'abnormal', 'left', 'right', 'present', 'absent', 'a', 'b', 'c', 'd', 'pa', 'ap', 'one', 'two', 'three', 'four', 'five', 'smaller', 'larger'}
    if len(tokens) <= 3:
        if all(t in closed_keywords or t.isdigit() for t in tokens):
            return True
        if all(len(t) == 1 for t in tokens):
            return True
    return False

def debug_evaluation(file_path: str, num_samples: int = 10):
    """Debug evaluation by examining actual data"""
    print(f"Debugging {file_path}")
    print("="*80)
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    closed_count = 0
    open_count = 0
    exact_match_count = 0
    
    for i, item in enumerate(data[:num_samples]):
        print(f"\nSample {i+1}:")
        print(f"Question: {item.get('question', 'N/A')}")
        
        # Extract responses
        pred = item.get('assistant_response', 'N/A')
        ref = item.get('answer', 'N/A')
        
        print(f"Model Response: {pred}")
        print(f"Ground Truth: {ref}")
        
        # Check if they're identical
        if pred == ref:
            exact_match_count += 1
            print("*** EXACT MATCH ***")
        
        # Check question type
        is_closed = is_closed_question(ref)
        if is_closed:
            closed_count += 1
            print(f"Type: CLOSED")
            
            # Extract answers
            pred_ans = extract_closed_answer(pred)
            ref_ans = extract_closed_answer(ref) if extract_closed_answer(ref) else ref.strip().lower()
            
            print(f"Extracted Pred: {pred_ans}")
            print(f"Extracted GT: {ref_ans}")
            
            if pred_ans == ref_ans:
                print("*** CORRECT ***")
            else:
                print("*** WRONG ***")
        else:
            open_count += 1
            print(f"Type: OPEN")
            
            # Calculate recall
            pred_tokens = set(simple_tokenize(pred))
            ref_tokens = set(simple_tokenize(ref))
            
            if ref_tokens:
                recall = len(pred_tokens & ref_tokens) / len(ref_tokens)
                print(f"Recall: {recall:.3f}")
                print(f"Pred tokens: {pred_tokens}")
                print(f"Ref tokens: {ref_tokens}")
                print(f"Intersection: {pred_tokens & ref_tokens}")
    
    print(f"\nSummary:")
    print(f"Total samples examined: {num_samples}")
    print(f"Closed questions: {closed_count}")
    print(f"Open questions: {open_count}")
    print(f"Exact matches: {exact_match_count}")
    print(f"Exact match rate: {exact_match_count/num_samples:.3f}")

if __name__ == "__main__":
    # Debug a few files
    files_to_debug = [
        "output/internvl8b/test_pvqa.json",
        "output/internvl30b/test_rad.json",
        "output/qwen25vl7b/test_rad.json"
    ]
    
    for file_path in files_to_debug:
        try:
            debug_evaluation(file_path, num_samples=5)
            print("\n" + "="*80 + "\n")
        except Exception as e:
            print(f"Error debugging {file_path}: {e}")
            print("\n" + "="*80 + "\n")

