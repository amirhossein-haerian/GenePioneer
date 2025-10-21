#!/usr/bin/env python3

import sys
import os
sys.path.append('/Users/amirho3in/Documents/Stockholm University/Thesis/Project/GenePioneer')

from genepioneer.evaluation import Evaluation
import json

def analyze_updated_results():
    """Analyze the results with updated network size"""
    
    print("UPDATED GENEPIONEER EVALUATION RESULTS")
    print("="*50)
    
    evaluator = Evaluation(data_path=".")
    network_size = evaluator.get_network_size()
    print(f"Actual network size: {network_size} genes")
    print(f"Previous estimate: 20,000 genes")
    print(f"Difference: {((20000 - network_size) / 20000) * 100:.1f}% smaller than estimated")
    print()
    
    # Load updated results
    try:
        with open('evaluated_modules_result_updated.json', 'r') as f:
            updated_results = json.load(f)
    except:
        print("Updated results file not found. Running quick evaluation...")
        
        # Load modules
        with open('tests/Prostate_filtered.json', 'r') as f:
            modules_data = json.load(f)
        
        modules = {"Prostate_filtered": modules_data}
        updated_results = evaluator.evaluate_modules(modules)
    
    # Analyze results
    if "Prostate_filtered" in updated_results:
        accepted_modules = updated_results["Prostate_filtered"]
        total_modules = 57  # Based on previous runs
        
        print(f"UPDATED EVALUATION SUMMARY:")
        print(f"- Total modules evaluated: {total_modules}")
        print(f"- Modules with significant pathways: {len(accepted_modules)}")
        print(f"- Success rate: {len(accepted_modules)/total_modules*100:.1f}%")
        
        if accepted_modules:
            # Count total significant pathways
            total_pathways = sum(len(module['significant_pathways']) for module in accepted_modules)
            print(f"- Total significant pathway enrichments: {total_pathways}")
            
            # Get unique pathways
            all_pathways = set()
            for module in accepted_modules:
                for pathway in module['significant_pathways']:
                    all_pathways.add(pathway['name'])
            print(f"- Unique cancer hallmark pathways: {len(all_pathways)}")
            
            # Show CHD1L and DPF2 modules
            chd1l_modules = [m for m in accepted_modules if 'CHD1L' in m['module_genes']]
            dpf2_modules = [m for m in accepted_modules if 'DPF2' in m['module_genes']]
            
            print(f"- Modules containing CHD1L: {len(chd1l_modules)}")
            print(f"- Modules containing DPF2: {len(dpf2_modules)}")
            
            # Show most significant enrichments
            print(f"\nTOP 5 MOST SIGNIFICANT ENRICHMENTS (Updated p-values):")
            all_enrichments = []
            for module in accepted_modules:
                for pathway in module['significant_pathways']:
                    all_enrichments.append({
                        'module': module['module_genes'][:3],
                        'pathway': pathway['name'],
                        'p_value': pathway['p_value'],
                        'enrichment_ratio': pathway['enrichment_ratio']
                    })
            
            all_enrichments.sort(key=lambda x: x['p_value'])
            
            for i, enrichment in enumerate(all_enrichments[:5]):
                module_str = ', '.join(enrichment['module'])
                print(f"{i+1}. {enrichment['pathway']}")
                print(f"   Module: [{module_str}...]")
                print(f"   p-value: {enrichment['p_value']:.2e}")
                print(f"   Enrichment: {enrichment['enrichment_ratio']:.1f}x")
                print()
        
        else:
            print("No modules passed the updated statistical criteria.")
    
    else:
        print("No results found for Prostate_filtered dataset.")

if __name__ == "__main__":
    analyze_updated_results()
