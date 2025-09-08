from test_internvl8b_grounding import *


def parse_args():
    parser = argparse.ArgumentParser(description="Grounding with InternVL3.5-30B-A3B on SLAKE index")
    parser.add_argument("--index", type=str, default="/home/xc1490/xc1490/projects/medvqa_2025/data/medvqa/3vqa/images/slake/grounding_index.json")
    parser.add_argument("--output", type=str, default="/home/xc1490/xc1490/projects/medvqa_2025/output/grounding/internvl30b_slake.jsonl")
    parser.add_argument("--model_path", type=str, default="OpenGVLab/InternVL3_5-30B-A3B")
    parser.add_argument("--device_map", type=str, default="auto", choices=["auto", "cuda:0"])
    parser.add_argument("--skip_empty", action="store_true")
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--limit_pairs", type=int, default=None)
    parser.add_argument("--per_case_progress", action="store_true")
    parser.add_argument("--offline", action="store_true", default=True,
                       help="Load model from local cache only (default: True)")
    return parser.parse_args()


if __name__ == "__main__":
    main()


