#!/usr/bin/env python3
"""
Unified GLM-4.xV training entrypoint for MedVQA

- Modes: thinking | base
- Methods: lora (SFT) | dpo (preference RL)

This wraps LLaMA-Factory by generating minimal dataset_info and training config
and invoking its train.py. It reuses existing converted data in this repo:
  medvqa_2025/bin/glm4v_train/convert_medvqa_to_glm4v.py
  medvqa_2025/bin/glm4v_train/convert_medvqa_to_glm4v_with_think.py
"""
import argparse
import json
import os
import subprocess
import sys
from typing import Dict


DATASET_FILE_MAP = {
    # base format (no <think>)
    ("base", "individual"): {
        "pvqa": "train_pvqa.json",
        "slake": "train_slake.json",
        "rad": "train_rad.json",
    },
    ("base", "combined"): {
        "combined": "train_combined.json",
    },
    # thinking format (<think> + <answer>)
    ("thinking", "individual"): {
        "pvqa": "train_pvqa_with_think.json",
        "slake": "train_slake_with_think.json",
        "rad": "train_rad_with_think.json",
    },
    ("thinking", "combined"): {
        "combined": "train_combined_with_think.json",
    },
}


def resolve_train_file(dataset: str, mode: str, data_scope: str, data_dir: str) -> str:
    key = (mode, data_scope)
    mapping = DATASET_FILE_MAP.get(key)
    if not mapping:
        raise ValueError(f"Unsupported mode/scope: {mode}/{data_scope}")
    ds_key = dataset.lower()
    if ds_key not in mapping:
        raise ValueError(f"Unsupported dataset for {data_scope}: {dataset}")
    fn = mapping[ds_key]
    train_file = os.path.join(data_dir, fn)
    if not os.path.exists(train_file):
        raise FileNotFoundError(
            f"Training file not found: {train_file}. Run conversion first.")
    return train_file


def build_dataset_info(dataset_name: str, train_file: str) -> Dict:
    return {
        dataset_name: {
            "file_name": train_file,
            "formatting": "sharegpt",
            "columns": {"messages": "messages", "images": "images"},
            "tags": {
                "role_tag": "role",
                "content_tag": "content",
                "user_tag": "user",
                "assistant_tag": "assistant",
            },
        }
    }


def build_sft_config(
    dataset_name: str,
    output_dir: str,
    model_name_or_path: str,
    template: str,
    finetuning_type: str = "lora",
    per_device_train_batch_size: int = 1,
    grad_accum: int = 8,
    learning_rate: float = 1e-4,
    num_train_epochs: float = 3.0,
) -> Dict:
    return {
        # model
        "model_name_or_path": model_name_or_path,
        "trust_remote_code": True,
        "use_fast_tokenizer": True,
        # multimodal
        "is_multimodal": True,
        "visual_inputs": True,
        "vision_tower": "auto",
        "image_resolution": 896,
        # method
        "stage": "sft",
        "do_train": True,
        "finetuning_type": finetuning_type,
        # common LoRA defaults
        "lora_target": "all",
        "lora_rank": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.1,
        # dataset
        "dataset": dataset_name,
        "template": template,
        "cutoff_len": 2048,
        "overwrite_cache": True,
        "preprocessing_num_workers": 8,
        # output
        "output_dir": output_dir,
        "logging_steps": 10,
        "save_steps": 500,
        "plot_loss": True,
        "overwrite_output_dir": True,
        # train
        "per_device_train_batch_size": per_device_train_batch_size,
        "gradient_accumulation_steps": grad_accum,
        "learning_rate": learning_rate,
        "num_train_epochs": num_train_epochs,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.1,
        "bf16": True,
        "gradient_checkpointing": True,
        "ddp_timeout": 180000000,
        # eval
        "val_size": 0.1,
        "per_device_eval_batch_size": 1,
        "evaluation_strategy": "steps",
        "eval_steps": 500,
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "report_to": "none",
    }


def build_dpo_config(
    output_dir: str,
    base_model_path: str,
    preference_file: str,
    template: str,
    finetuning_type: str = "lora",
    per_device_train_batch_size: int = 1,
    grad_accum: int = 8,
    learning_rate: float = 5e-6,
    num_train_epochs: float = 1.0,
) -> Dict:
    if not os.path.exists(preference_file):
        raise FileNotFoundError(f"Preference file not found: {preference_file}")
    return {
        # model
        "model_name_or_path": base_model_path,  # SFT base
        "trust_remote_code": True,
        "use_fast_tokenizer": True,
        # multimodal
        "is_multimodal": True,
        "visual_inputs": True,
        "vision_tower": "auto",
        "image_resolution": 896,
        # method
        "stage": "dpo",
        "do_train": True,
        "finetuning_type": finetuning_type,
        # LoRA for DPO as default
        "lora_target": "all",
        "lora_rank": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.1,
        # dataset (preference pairs)
        "preference_file": preference_file,
        "template": template,
        # output
        "output_dir": output_dir,
        "logging_steps": 10,
        "save_steps": 500,
        "plot_loss": True,
        "overwrite_output_dir": True,
        # train
        "per_device_train_batch_size": per_device_train_batch_size,
        "gradient_accumulation_steps": grad_accum,
        "learning_rate": learning_rate,
        "num_train_epochs": num_train_epochs,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.1,
        "bf16": True,
        "gradient_checkpointing": True,
        "ddp_timeout": 180000000,
        "report_to": "none",
    }


def run_llamafactory(train_cfg: Dict, dataset_info: Dict | None, llamafactory_path: str) -> None:
    cfg_file = "/tmp/llamafactory_train_cfg.json"
    with open(cfg_file, "w", encoding="utf-8") as f:
        json.dump(train_cfg, f, indent=2, ensure_ascii=False)
    cmd = ["python", f"{llamafactory_path}/src/train.py", "--config", cfg_file]
    ds_file = None
    if dataset_info is not None:
        ds_file = "/tmp/llamafactory_dataset_info.json"
        with open(ds_file, "w", encoding="utf-8") as f:
            json.dump(dataset_info, f, indent=2, ensure_ascii=False)
        cmd += ["--dataset_info", ds_file]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    p = argparse.ArgumentParser(description="Train GLM-4.xV on MedVQA (LoRA/DPO)")
    p.add_argument("--method", choices=["lora", "dpo"], required=True, help="Training method")
    p.add_argument("--dataset", choices=["pvqa", "slake", "rad", "combined"], required=True)
    p.add_argument("--mode", choices=["thinking", "base"], default="base", help="Data format mode")
    p.add_argument("--model_version", choices=["4.1v", "4.5v"], default="4.1v")
    p.add_argument("--llamafactory_path", default="/home/xc1490/LLaMA-Factory")
    p.add_argument("--data_scope", choices=["individual", "combined"], help="Use individual files or combined", default=None)
    p.add_argument("--data_dir_base", default="/home/xc1490/xc1490/projects/medvqa_2025/data",
                   help="Base dir containing glm4v_format(_with_think)")
    p.add_argument("--model_name_or_path", default="zai-org/GLM-4.1V-9B-Base",
                   help="HuggingFace model or local path for SFT")
    p.add_argument("--template", default="glm4", help="LLaMA-Factory chat template (glm4 or glm4v)")
    p.add_argument("--output_dir", default=None)
    # DPO specific
    p.add_argument("--base_model", default=None, help="Base SFT model for DPO")
    p.add_argument("--preference_file", default=None, help="Preference pairs JSON for DPO")
    # Common knobs
    p.add_argument("--finetuning_type", choices=["lora", "full"], default="lora")
    p.add_argument("--per_device_train_batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=8)
    p.add_argument("--learning_rate", type=float, default=None)
    p.add_argument("--num_train_epochs", type=float, default=None)
    args = p.parse_args()

    # Resolve scope default from dataset
    data_scope = args.data_scope or ("combined" if args.dataset == "combined" else "individual")

    # Pick data_dir by mode
    data_dir = os.path.join(
        args.data_dir_base, "glm4v_format_with_think" if args.mode == "thinking" else "glm4v_format"
    )

    # Determine train file when doing SFT
    dataset_info = None
    dataset_name = f"medvqa_{args.dataset}_{args.mode}"
    output_dir = args.output_dir or os.path.join(
        "/home/xc1490/xc1490/projects/medvqa_2025/models",
        f"glm{args.model_version}_{args.dataset}_{args.mode}_{args.method}"
    )
    os.makedirs(output_dir, exist_ok=True)

    template = args.template

    try:
        if args.method == "lora":
            train_file = resolve_train_file(args.dataset, args.mode, data_scope, data_dir)
            dataset_info = build_dataset_info(dataset_name, train_file)
            lr = args.learning_rate or 1e-4
            epochs = args.num_train_epochs or 3.0
            train_cfg = build_sft_config(
                dataset_name=dataset_name,
                output_dir=output_dir,
                model_name_or_path=args.model_name_or_path,
                template=template,
                finetuning_type=args.finetuning_type,
                per_device_train_batch_size=args.per_device_train_batch_size,
                grad_accum=args.gradient_accumulation_steps,
                learning_rate=lr,
                num_train_epochs=epochs,
            )
            run_llamafactory(train_cfg, dataset_info, args.llamafactory_path)
        else:  # dpo
            if not args.base_model:
                print("--base_model is required for DPO", file=sys.stderr)
                sys.exit(2)
            if not args.preference_file:
                print("--preference_file is required for DPO", file=sys.stderr)
                sys.exit(2)
            lr = args.learning_rate or 5e-6
            epochs = args.num_train_epochs or 1.0
            train_cfg = build_dpo_config(
                output_dir=output_dir,
                base_model_path=args.base_model,
                preference_file=args.preference_file,
                template=template,
                finetuning_type=args.finetuning_type,
                per_device_train_batch_size=args.per_device_train_batch_size,
                grad_accum=args.gradient_accumulation_steps,
                learning_rate=lr,
                num_train_epochs=epochs,
            )
            run_llamafactory(train_cfg, None, args.llamafactory_path)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

