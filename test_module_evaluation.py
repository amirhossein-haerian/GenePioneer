#!/usr/bin/env python3
"""
Test the updated module evaluation
"""
import sys
import json
sys.path.append('.')

from genepioneer.evaluation import Evaluation

def main():
    print("Testing module evaluation...")
    
    # Load modules
    with open('tests/Prostate_filtered.json', 'r') as f:
        modules_data = json.load(f)
    
    print(f"Loaded {len(modules_data)} modules")
    
    # Create evaluation object (minimal setup)
    eval_obj = Evaluation(data_path="./GenesData")
    
    if not eval_obj.hallmarks_data:
        print("Failed to load hallmarks data")
        return
    
    print(f"Loaded {len(eval_obj.hallmarks_data)} hallmark pathways")
    
    # Test with just a few modules
    test_modules = {"Prostate_filtered": modules_data[:5]}  # First 5 modules
    
    print("\\nEvaluating modules...")
    results = eval_obj.evaluate_modules(test_modules)
    
    print(f"\\nEvaluation complete!")
    print(f"Results for Prostate_filtered: {len(results.get('Prostate_filtered', []))} modules passed")
    
    # Show results
    if 'Prostate_filtered' in results and results['Prostate_filtered']:
        for i, result in enumerate(results['Prostate_filtered']):
            print(f"\\nModule {i+1}: {result['module_genes']}")
            print(f"  Score1: {result['score1']}")
            print(f"  Score2: {result['score2']}")
            print(f"  Significant pathways: {len(result['significant_pathways'])}")
            for pathway in result['significant_pathways']:
                print(f"    - {pathway['name']}: p={pathway['p_value']:.2e}")

if __name__ == "__main__":
    main()
