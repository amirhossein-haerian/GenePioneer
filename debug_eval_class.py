#!/usr/bin/env python3
"""
Debug the evaluation class
"""
import sys
import os
sys.path.append('.')

from genepioneer.evaluation import Evaluation

def main():
    print("Testing Evaluation class...")
    
    # Test with default paths
    print("\n=== Testing with default paths ===")
    eval_obj = Evaluation(data_path="./GenesData")
    
    print(f"Module data path: {eval_obj.module_data_path}")
    print(f"Modules loaded: {list(eval_obj.modules.keys())}")
    
    if eval_obj.modules:
        for cancer_type, modules in eval_obj.modules.items():
            print(f"  {cancer_type}: {len(modules)} modules")
    
    print(f"Module results: {list(eval_obj.module_results.keys())}")
    
    if eval_obj.module_results:
        for cancer_type, results in eval_obj.module_results.items():
            print(f"  {cancer_type}: {len(results)} evaluated modules")
    
    # Test with explicit path to tests directory
    print("\n=== Testing with tests directory ===")
    eval_obj2 = Evaluation(
        data_path="./GenesData",
        module_data_path="tests"
    )
    
    print(f"Module data path: {eval_obj2.module_data_path}")
    print(f"Modules loaded: {list(eval_obj2.modules.keys())}")
    
    if eval_obj2.modules:
        for cancer_type, modules in eval_obj2.modules.items():
            print(f"  {cancer_type}: {len(modules)} modules")
    
    print(f"Module results: {list(eval_obj2.module_results.keys())}")
    
    if eval_obj2.module_results:
        for cancer_type, results in eval_obj2.module_results.items():
            print(f"  {cancer_type}: {len(results)} evaluated modules")

if __name__ == "__main__":
    main()
