import torch
from PIL import Image
from transformers import AutoModelForCausalLM
from janus.models import VLChatProcessor, MultiModalityCausalLM
from flask import Flask, request, jsonify
import traceback

app = Flask(__name__)

# 只加载一次模型
MODEL_PATH = "deepseek-ai/Janus-Pro-7B"
processor = VLChatProcessor.from_pretrained(MODEL_PATH)
tokenizer = processor.tokenizer
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, trust_remote_code=True
)
model = model.to(torch.bfloat16).cuda().eval()

@app.route('/answer', methods=['POST'])
def answer():
    try:
        data = request.json
        image_path = data['image']
        question = data['question']
        # 拼接图片绝对路径
        abs_image_path = '/scratch/xc1490/projects/medvqa_2025/data/medvqa/3vqa/images/' + image_path
        try:
            image = Image.open(abs_image_path).convert("RGB")
        except Exception as img_e:
            return jsonify({"error": f"Failed to open image: {abs_image_path}, error: {img_e}"}), 400
        system_prompt = """You are a biomedical VQA assistant being evaluated by automatic metrics.
- Closed-set questions use exact-match accuracy.
- Open-set questions use token recall (ground-truth tokens must appear in your answer).

Follow ALL rules strictly:

1) Output format
- Respond in English only.
- Return ONLY the final answer, without any explanation, reasoning, or preface.
- For yes/no questions, output exactly: \"yes\" or \"no\" (lowercase).
- For counting questions (\"How many\"), output a bare integer (e.g., \"2\").
- Otherwise, answer with a single medical term or a very short noun phrase (≤ 3 words).
- Include laterality/location ONLY if the question explicitly asks for it (e.g., “right pneumothorax”).
- NEVER repeat the question or any special tokens like \"<image>\".

2) Visual & domain focus
- Base answers strictly on visible evidence in the image (X-ray, CT, MRI, ultrasound, etc.).
- Identify imaging modality or view/orientation **only when asked**, and use canonical terms (e.g., \"posterior-anterior\", \"lateral\", \"CT\", \"MRI\").
- Prefer canonical, unabbreviated medical terms unless the question itself uses an abbreviation.
  Examples: \"cardiomegaly\", \"pneumothorax\", \"pleural effusion\", \"endotracheal tube\".
- When multiple findings are present, choose the single most clinically relevant keyword that answers the question.

3) Prohibited language
- Do NOT start with fillers: \"There is/are\", \"It shows\", etc.
- Do NOT use hedging words: \"likely\", \"possibly\", \"probably\", \"appears to\", \"suggests\", \"could be\", \"may be\".
- Do NOT add disclaimers or medical advice.

Keep answers minimal so that the key medical tokens are explicit and unambiguous."""
        conversation = [
            {"role": "system", "content": system_prompt},
            {"role": "<|User|>", "content": f"<image>\n{question}", "images": [image]},
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
        return jsonify({"answer": answer})
    except Exception as e:
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000) 