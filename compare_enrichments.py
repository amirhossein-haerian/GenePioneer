#!/usr/bin/env python3

import sys
import os
sys.path.append('/Users/amirho3in/Documents/Stockholm University/Thesis/Project/GenePioneer')

from genepioneer.evaluation import Evaluation
from scipy.stats import hypergeom

def compare_enrichment_calculations():
    """Compare enrichment calculations with old vs new background sizes"""
    
    evaluator = Evaluation(data_path=".")
    
    # Example module: CHD1L, GATA3, PTEN, PIK3CA, EDN1
    test_genes = ['CHD1L', 'GATA3', 'PTEN', 'PIK3CA', 'EDN1']
    
    print("Comparing enrichment calculations:")
    print(f"Test module: {test_genes}")
    print("="*60)
    
    # Old method (background = 20000)
    print("OLD METHOD (Background = 20,000 genes):")
    old_results = evaluator.manual_pathway_enrichment(test_genes, background_size=20000)
    
    # New method (actual network size)
    print(f"\nNEW METHOD (Background = actual network size):")
    new_results = evaluator.manual_pathway_enrichment(test_genes, background_size=None)
    
    # Compare top 5 pathways
    print("\nCOMPARISON OF TOP 5 PATHWAYS:")
    print("Pathway | Old p-value | New p-value | Change")
    print("-" * 60)
    
    # Create dictionaries for easy comparison
    old_dict = {r['pathway']: r for r in old_results[:10]}
    new_dict = {r['pathway']: r for r in new_results[:10]}
    
    # Find common pathways
    common_pathways = set(old_dict.keys()) & set(new_dict.keys())
    
    for pathway in list(common_pathways)[:5]:
        old_p = old_dict[pathway]['p_value']
        new_p = new_dict[pathway]['p_value']
        change = "More sig." if new_p < old_p else "Less sig."
        print(f"{pathway[:25]:<25} | {old_p:.6f} | {new_p:.6f} | {change}")
    
    print(f"\nSIGNIFICANT PATHWAYS (p ≤ 0.05):")
    old_sig = [r for r in old_results if r['p_value'] <= 0.05]
    new_sig = [r for r in new_results if r['p_value'] <= 0.05]
    
    print(f"Old method: {len(old_sig)} significant pathways")
    print(f"New method: {len(new_sig)} significant pathways")
    
    if new_sig:
        print(f"\nTop 3 significant pathways with new method:")
        for i, result in enumerate(new_sig[:3]):
            print(f"  {i+1}. {result['pathway']}: p={result['p_value']:.6f}, enrichment={result['enrichment_ratio']:.2f}x")

if __name__ == "__main__":
    compare_enrichment_calculations()
