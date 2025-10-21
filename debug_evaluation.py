#!/usr/bin/env python3
"""
Simple test for module evaluation debugging
"""
import sys
import os
import json

# Add current directory to path
sys.path.append('.')

def test_single_module():
    # Load a module from your results
    with open('tests/Prostate_filtered.json', 'r') as f:
        modules = json.load(f)
    
    if not modules:
        print("No modules found in file")
        return
    
    # Test first module
    test_module = modules[0]
    genes = test_module[0]
    
    print(f"Testing module with genes: {genes}")
    
    # Manual enrichment test
    from genepioneer.evaluation import Evaluation
    
    try:
        eval_obj = Evaluation(data_path="./GenesData")
        
        if hasattr(eval_obj, 'hallmarks_data') and eval_obj.hallmarks_data:
            print(f"Loaded {len(eval_obj.hallmarks_data)} hallmark pathways")
            
            # Test manual enrichment
            enrichment_results = eval_obj.manual_pathway_enrichment(genes)
            print(f"Found {len(enrichment_results)} enrichment results")
            
            # Show top 5 results
            for i, result in enumerate(enrichment_results[:5]):
                print(f"{i+1}. {result['pathway']}: p={result['p_value']:.2e}, overlap={result['overlap_size']}")
                
            # Check significant ones
            significant = [r for r in enrichment_results if r['p_value'] <= 0.05]
            print(f"Significant results (p<=0.05): {len(significant)}")
            
        else:
            print("Failed to load hallmarks data")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_single_module()
