import os
import json

def debug_internvl38b():
    """Debug why internvl38b wasn't fully fixed"""
    output_path = "/home/xc1490/xc1490/projects/medvqa_2025/output/internvl38b/test_pvqa.json"
    
    with open(output_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Total samples in internvl38b: {len(data)}")
    
    # Check first few samples
    for i in range(5):
        item = data[i]
        image_path = item.get("image", "")
        question = item.get("question", "")
        assistant_response = item.get("assistant_response", "")
        answer = item.get("answer", "")
        
        print(f"\nSample {i+1}:")
        print(f"Image: {image_path}")
        print(f"Question: {question}")
        print(f"Assistant Response: {assistant_response[:100]}...")
        print(f"Answer: {answer[:100]}...")
        print(f"Same: {assistant_response == answer}")

if __name__ == "__main__":
    debug_internvl38b()

