import json
import requests
from tqdm import tqdm
import os

def load_test_cases(json_path):
    with open(json_path, 'r') as f:
        cases = json.load(f)
    return cases

def load_existing_results(result_path):
    if os.path.exists(result_path):
        with open(result_path, 'r') as f:
            try:
                results = json.load(f)
            except Exception:
                results = []
        done_keys = set()
        for r in results:
            if 'id' in r:
                done_keys.add(r['id'])
            else:
                done_keys.add(r['image'] + '||' + r['question'])
        return results, done_keys
    else:
        return [], set()

def infer_once(cases, result_path, api_url):
    # 确保目录存在
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    results, done_keys = load_existing_results(result_path)
    batch = []
    for idx, case in tqdm(enumerate(cases), total=len(cases)):
        case_id = case.get('id', None)
        if case_id:
            key = case_id
        else:
            key = case["image"] + '||' + case["conversations"][0]['value']
        if key in done_keys:
            continue
        image_path = case["image"]
        question = case["conversations"][0]['value']
        gt_answer = case["conversations"][1]['value']
        try:
            resp = requests.post(api_url, json={"image": image_path, "question": question}, timeout=600)
            resp_json = resp.json()
            answer = resp_json.get("answer", resp_json.get("error", "[NO ANSWER]"))
        except Exception as e:
            answer = f"[ERROR] {e}"
            resp_json = {}
        print ('resp_json', resp_json)
        print(f"Case {idx+1} | Image: {image_path} | Q: {question}\nA: {answer}\n{'-'*60}")
        # 跳过兜底回答，下次继续
        sorry_prefix = "I'm sorry"
        if isinstance(answer, str) and answer.strip().startswith(sorry_prefix):
            continue
        result_item = {
            "image": image_path,
            "question": question,
            "model_answer": answer,
            'gt_answer': gt_answer
        }
        if case_id:
            result_item['id'] = case_id
        results.append(result_item)
        done_keys.add(key)
        batch.append(result_item)
        if len(batch) >= 5:
            with open(result_path, 'w') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            batch = []
    if batch:
        with open(result_path, 'w') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

def main(test_json, result_path):
    api_url = "http://127.0.0.1:5000/answer"
    cases = load_test_cases(test_json)
    for i in range(3):
        print(f"==== Inference round {i+1}/3 for {test_json} ====")
        infer_once(cases, result_path, api_url)

if __name__ == "__main__":
    test_sets = [
        ("test_rad.json", "test_rad_client.json"),
        ("test_pvqa.json", "test_pvqa_client.json"),
        ("test_slake.json", "test_slake_client.json"),
    ]
    base_in = "/scratch/xc1490/projects/medvqa_2025/data/medvqa/"
    base_out = "/scratch/xc1490/projects/medvqa_2025/output/janus/zeroshot_sysprompt/"
    for test_file, out_file in test_sets:
        main(base_in + test_file, base_out + out_file) 