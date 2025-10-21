#!/usr/bin/env python3
"""
Simple test for Evaluation class
"""
import sys
import os
sys.path.append('.')

from genepioneer.evaluation import Evaluation

def main():
    print("Creating Evaluation object...")
    
    try:
        # Create evaluation with corrected paths
        eval_obj = Evaluation(data_path="./GenesData")
        
        print(f"Hallmarks data loaded: {len(eval_obj.hallmarks_data)} pathways")
        print(f"Modules loaded: {eval_obj.modules}")
        
        if 'Prostate_filtered' in eval_obj.modules:
            print(f"Prostate_filtered has {len(eval_obj.modules['Prostate_filtered'])} modules")
        
        print(f"Module evaluation results: {eval_obj.module_results}")
        
        if 'Prostate_filtered' in eval_obj.module_results:
            print(f"Prostate_filtered evaluation: {len(eval_obj.module_results['Prostate_filtered'])} results")
        
        # Save results
        eval_obj.print_module_evaluation()
        print("Results saved!")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
