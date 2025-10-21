#!/usr/bin/env python3
"""
Run only step 4 (evaluation) on already detected modules
"""

import os
import sys
import json

sys.path.append('.')

from genepioneer.evaluation import Evaluation

def main():
    output_dir = "./validation_cohort"
    modules_file = os.path.join(output_dir, "all_modules.json")
    
    # Check if modules file exists
    if not os.path.exists(modules_file):
        print(f"❌ ERROR: Modules file not found: {modules_file}")
        print("Please run step 3 first to detect modules")
        return
    
    # Load modules
    print("Loading modules...")
    with open(modules_file, 'r') as f:
        all_modules = json.load(f)
    
    print(f"✓ Loaded {len(all_modules)} modules")
    
    # Check module structure
    print("\nChecking module structure...")
    if all_modules:
        print(f"First module type: {type(all_modules[0])}")
        print(f"First module: {all_modules[0]}")
    
    # Prepare modules for evaluation
    module_dict = {"ValidationCohort": all_modules}
    
    # Create Evaluation object
    print("\nInitializing evaluation...")
    evaluator = Evaluation(data_path="./GenesData")
    
    if not evaluator.hallmarks_data:
        print("⚠️  Failed to load Hallmark pathways")
        print(f"GenesData path exists: {os.path.exists('./GenesData')}")
        return
    
    print(f"✓ Loaded {len(evaluator.hallmarks_data)} Hallmark pathways")
    
    # Evaluate modules
    print("\nPerforming pathway enrichment analysis...")
    try:
        results = evaluator.evaluate_modules(module_dict)
        
        # Save results
        results_file = os.path.join(output_dir, "evaluation_results.json")
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n✓ Evaluation results saved to: {results_file}")
        
        # Print summary
        if "ValidationCohort" in results:
            validated_modules = results["ValidationCohort"]
            print(f"✓ Validated modules: {len(validated_modules)}")
            
            # Find CHD1L modules
            chd1l_evaluated = []
            for module_result in validated_modules:
                if 'CHD1L' in module_result['module_genes']:
                    chd1l_evaluated.append(module_result)
            
            if chd1l_evaluated:
                print(f"✓ CHD1L modules with enrichment: {len(chd1l_evaluated)}")
                
                print("\n" + "="*70)
                print("CHD1L MODULE ENRICHMENT RESULTS:")
                print("="*70)
                
                for i, module_result in enumerate(chd1l_evaluated, 1):
                    print(f"\nModule {i}:")
                    print(f"  Genes: {', '.join(sorted(module_result['module_genes']))}")
                    print(f"  Significant pathways: {len(module_result['significant_pathways'])}")
                    
                    if module_result['significant_pathways']:
                        print("\n  Top pathways:")
                        for pathway in module_result['significant_pathways'][:5]:
                            print(f"    - {pathway['name']}")
                            print(f"      p-value: {pathway['p_value']:.2e}")
            else:
                print("\n⚠️  No CHD1L modules passed enrichment criteria")
        else:
            print("⚠️  No results for ValidationCohort")
            
    except Exception as e:
        print(f"\n❌ ERROR during evaluation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
