import json
import re
from typing import List
import sys

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

def evaluate_vqa(test_json_path: str, wrong_samples: list, dataset_tag: str):
    closed_correct = 0
    closed_total = 0
    open_recalls = []
    total = 0
    with open(test_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        for item in data:
            pred = item['model_answer']
            ref = item['gt_answer']
            total += 1
            if is_closed_question(ref):
                closed_total += 1
                pred_ans = extract_closed_answer(pred)
                ref_ans = extract_closed_answer(ref) if extract_closed_answer(ref) else ref.strip().lower()
                if pred_ans is not None and pred_ans == ref_ans.lower():
                    closed_correct += 1
                else:
                    wrong_samples.append({
                        'dataset': dataset_tag,
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
                ref_tokens = set(simple_tokenize(ref))
                pred_tokens = set(simple_tokenize(pred))
                if ref_tokens:
                    recall = len(ref_tokens & pred_tokens) / len(ref_tokens)
                else:
                    recall = 0.0
                open_recalls.append(recall)
                if recall < 0.1:
                    wrong_samples.append({
                        'dataset': dataset_tag,
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
    print(f"File: {test_json_path}")
    print(f"Total samples: {total}")
    if closed_acc is not None:
        print(f"Closed-set questions: {closed_total}, Accuracy: {closed_acc:.3f}")
    else:
        print(f"Closed-set questions: {closed_total}, Accuracy: N/A")
    if open_recall is not None:
        print(f"Open-set questions: {len(open_recalls)}, Recall: {open_recall:.3f}")
    else:
        print(f"Open-set questions: {len(open_recalls)}, Recall: N/A")
    print()
    return open_recall, closed_acc

def format_num(val):
    if val is None:
        return "  N/A "
    return f"{val:05.2f}"

def print_comparison_table(results):
    print("\n## Janus Pro vs LLaVA-Med (Markdown Table)")
    print("|           | VQA-RAD         | SLAKE           | PathVQA         |")
    print("|-----------|-----------------|-----------------|-----------------|")
    print("|           |  Open | Closed  |  Open | Closed  |  Open | Closed  |")
    print("| LLaVA-Med | 20.74 |  59.19  | 26.82 |  50.24  | 08.74 |  45.65  |")
    print("| Janus Pro | {} | {}  | {} | {}  | {} | {}  |".format(
        format_num(results['rad']['open']*100 if results['rad']['open'] is not None else None),
        format_num(results['rad']['closed']*100 if results['rad']['closed'] is not None else None),
        format_num(results['slake']['open']*100 if results['slake']['open'] is not None else None),
        format_num(results['slake']['closed']*100 if results['slake']['closed'] is not None else None),
        format_num(results['pvqa']['open']*100 if results['pvqa']['open'] is not None else None),
        format_num(results['pvqa']['closed']*100 if results['pvqa']['closed'] is not None else None),
    ))

if __name__ == "__main__":
    files = [
        ("output/janus/zeroshot_sysprompt/test_rad_client.json", 'rad'),
        ("output/janus/zeroshot_sysprompt/test_slake_client.json", 'slake'),
        ("output/janus/zeroshot_sysprompt/test_pvqa_client.json", 'pvqa'),
    ]
    results = {'rad': {'open': None, 'closed': None}, 'slake': {'open': None, 'closed': None}, 'pvqa': {'open': None, 'closed': None}}
    wrong_samples = []
    for fpath, key in files:
        try:
            open_recall, closed_acc = evaluate_vqa(fpath, wrong_samples, key)
            results[key]['open'] = open_recall
            results[key]['closed'] = closed_acc
        except Exception as e:
            print(f"Error evaluating {fpath}: {e}\n")
    print_comparison_table(results)
    # 保存错误和低recall样本
    with open("output/janus/zeroshot_sysprompt/januspro_wrong_or_lowrecall.jsonl", "w", encoding="utf-8") as fout:
        for item in wrong_samples:
            fout.write(json.dumps(item, ensure_ascii=False) + "\n") 