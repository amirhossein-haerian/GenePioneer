#!/usr/bin/env python3

import sys
import os
sys.path.append('/Users/amirho3in/Documents/Stockholm University/Thesis/Project/GenePioneer')

import json
from genepioneer.evaluation import Evaluation

def main():
    print("Starting module evaluation with updated network size...")
    
    # Load the modules from the JSON file
    with open('tests/Prostate_filtered.json', 'r') as f:
        modules_data = json.load(f)
    
    print(f"Loaded {len(modules_data)} modules from Prostate_filtered.json")
    
    # Create evaluation instance
    evaluator = Evaluation(data_path=".")
    
    # Get the actual network size
    network_size = evaluator.get_network_size()
    print(f"Using network size: {network_size} genes")
    
    # Prepare modules in the expected format
    modules = {"Prostate_filtered": modules_data}
    
    # Run evaluation
    print("Running module evaluation...")
    results = evaluator.evaluate_modules(modules)
    
    # Save results
    with open('evaluated_modules_result_updated.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    if results:
        total_modules = len(modules_data)
        accepted_modules = len(results.get("Prostate_filtered", []))
        print(f"\nEvaluation Summary:")
        print(f"Total modules: {total_modules}")
        print(f"Accepted modules: {accepted_modules}")
        print(f"Success rate: {accepted_modules/total_modules*100:.1f}%")
        
        # Show some example significant pathways
        if accepted_modules > 0:
            print(f"\nExample significant pathways from first accepted module:")
            first_module = results["Prostate_filtered"][0]
            for pathway in first_module['significant_pathways'][:3]:
                print(f"  - {pathway['name']}: p={pathway['p_value']:.6f}")
    else:
        print("No modules passed the evaluation criteria with updated network size")

if __name__ == "__main__":
    main()
