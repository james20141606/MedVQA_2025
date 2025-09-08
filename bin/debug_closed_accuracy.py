import json
import re
from typing import List
from evaluate_vqa import simple_tokenize, extract_closed_answer, is_closed_question

def debug_closed_accuracy():
    """Debug why closed accuracy is so high"""
    
    # Test with some sample data
    test_cases = [
        "yes",
        "no", 
        "a",
        "b",
        "c",
        "d",
        "male",
        "female",
        "present",
        "absent",
        "normal",
        "abnormal",
        "left",
        "right",
        "one",
        "two",
        "three",
        "four",
        "five",
        "smaller",
        "larger",
        "pa",
        "ap",
        # Test some longer answers that might be incorrectly classified
        "the histone subunits",
        "positively charged",
        "cells",
        "apoptosis",
        "progress",
        "yes, the histone subunits are positively charged",
        "no, the methylation of particular histone residues is not illustrated",
        "the principal cellular alterations that characterize reversible cell injury and necrosis",
        "reversible injury",
        "culminate in necrosis",
        "early (reversible) ischemic injury",
        "surface blebs",
        "the necrotic cells",
        "the wall of the artery",
        "a circumferential bright pink area of necrosis",
        "with protein deposition and inflammation",
        "the cellular alterations in apoptosis are illustrated",
        "apoptotic cells in colonic epithelium",
        "a gravid uterus",
        "postpartum bleeding",
        "an area of central necrosis",
        "multiple multinucleate giant cells",
        "thrombus",
        "white fibrous scar",
        "friable mural thrombi",
        "calcification",
        "inflammation",
        "vasodilation",
        "the granuloma",
        "multinucleate giant cells",
        "immunoperoxidase stain",
        "the tumor cells",
        "strikingly similar to normal squamous epithelial cells",
        "with intercellular bridges and nests of keratin (arrow)",
        "high-power view of another region",
        "failure of normal differentiation",
        "the intact basement membrane",
        "the microscopic view of breast carcinoma",
        "the invasion of breast stroma and fat by nests and cords of tumor cells",
        "by nests and cords of tumor cells",
        "fluid accumulation particularly prominent in the soft tissues of the neck",
        "cystic hygroma",
        "cystic hygromas",
        "constitutional chromosomal anomalies such as 45",
        "poorly cohesive tumor in retina",
        "the optic nerve",
        "wilms tumor with tightly packed blue cells",
        "the blastemal component and interspersed primitive tubules",
        "focal anaplasia",
        "focal anaplasia",
        "predominance of blastemal morphology and diffuse anaplasia",
        "specific molecular lesions",
        "granulomatous host response",
        "mycobacterium avium infection",
        "massive intracellular macrophage infection with acid-fast organisms"
    ]
    
    print("Testing closed question detection:")
    print("="*80)
    
    closed_count = 0
    for answer in test_cases:
        is_closed = is_closed_question(answer)
        extracted = extract_closed_answer(answer)
        print(f"Answer: '{answer}'")
        print(f"  Is closed: {is_closed}")
        print(f"  Extracted: '{extracted}'")
        print()
        
        if is_closed:
            closed_count += 1
    
    print(f"Total closed answers: {closed_count}/{len(test_cases)}")
    
    # Now let's check some actual model outputs
    print("\n" + "="*80)
    print("Checking actual model outputs:")
    print("="*80)
    
    # Check a few samples from thyme
    thyme_file = "/home/xc1490/xc1490/projects/medvqa_2025/output/thyme/test_pvqa.json"
    with open(thyme_file, 'r', encoding='utf-8') as f:
        thyme_data = json.load(f)
    
    closed_samples = 0
    total_samples = 0
    
    for i, item in enumerate(thyme_data[:20]):  # Check first 20 samples
        gt_answer = item.get('answer', '')
        is_closed = is_closed_question(gt_answer)
        extracted = extract_closed_answer(gt_answer)
        
        print(f"Sample {i+1}:")
        print(f"  GT Answer: '{gt_answer}'")
        print(f"  Is closed: {is_closed}")
        print(f"  Extracted: '{extracted}'")
        print()
        
        total_samples += 1
        if is_closed:
            closed_samples += 1
    
    print(f"Closed samples in thyme: {closed_samples}/{total_samples}")

if __name__ == "__main__":
    debug_closed_accuracy()

