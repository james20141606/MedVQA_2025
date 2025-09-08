import json
import torch
from PIL import Image
from transformers import AutoModelForCausalLM
from janus.models import VLChatProcessor, MultiModalityCausalLM
from tqdm import tqdm  as tqdm
# 1. 加载模型和processor
def load_model_and_processor(model_path):
    processor = VLChatProcessor.from_pretrained(model_path)
    tokenizer = processor.tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True
    )
    model = model.to(torch.bfloat16).cuda().eval()
    return model, processor, tokenizer

# 2. 读取测试集json
def load_test_cases(json_path):
    with open(json_path, 'r') as f:
        cases = json.load(f)
    return cases

# 3. 单条推理
def infer_one(model, processor, tokenizer, image_path, question):
    image = Image.open('/home/xc1490/xc1490/projects/medvqa_2025/data/medvqa/3vqa/images/'+image_path)
    conversation = [
        {"role": "<|User|>", "content": f"{question}", "images": [image]},
        {"role": "<|Assistant|>", "content": ""},
    ]
    inputs = processor(conversations=conversation, images=[image], force_batchify=True)
    inputs = inputs.to(model.device)
    inputs_embeds = model.prepare_inputs_embeds(**inputs)
    with torch.inference_mode():
        gen_tokens = model.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=inputs.attention_mask,
            pad_token_id=tokenizer.eos_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            max_new_tokens=512,
            do_sample=False,
            use_cache=True,
        )
    answer = tokenizer.decode(gen_tokens[0], skip_special_tokens=True)
    return answer

# 4. 主流程
def main():
    model_path = "deepseek-ai/Janus-Pro-7B"
    test_json = "/scratch/xc1490/projects/medvqa_2025/data/medvqa/test_pvqa.json"
    model, processor, tokenizer = load_model_and_processor(model_path)
    cases = load_test_cases(test_json)
    results = []
    for idx, case in tqdm(enumerate(cases)):
        image_path = case["image"]
        question = case["question"]
        try:
            answer = infer_one(model, processor, tokenizer, image_path, question)
        except Exception as e:
            answer = f"[ERROR] {e}"
        print(f"Case {idx+1} | Image: {image_path} | Q: {question}\nA: {answer}\n{'-'*60}")
        results.append({
            "image": image_path,
            "question": question,
            "answer": answer
        })
    #可选：保存结果
    if not os.path.exists('/scratch/xc1490/projects/medvqa_2025/output/janus/zeroshot'):
        os.makedirs('/scratch/xc1490/projects/medvqa_2025/output/janus/zeroshot',exist_ok=True)
    with open('/scratch/xc1490/projects/medvqa_2025/output/janus/zeroshot/test_pvqa.json', 'w') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main() 