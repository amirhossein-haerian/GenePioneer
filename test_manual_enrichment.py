#!/usr/bin/env python3
"""
Test script for manual pathway enrichment analysis
"""
import sys
import os
sys.path.append('.')

from genepioneer.evaluation import Evaluation

# Test with a small module from your results
test_module = [
    ["CHD1L", "GATA3", "PTEN", "PIK3CA", "EDN1"],
    100.0,  # score1
    6.2     # score2 (quality)
]

print("Testing manual pathway enrichment...")

# Initialize evaluation
try:
    eval_obj = Evaluation(data_path="./GenesData")
    
    # Test manual enrichment
    enrichment_results = eval_obj.manual_pathway_enrichment(test_module[0])
    
    print(f"\nTesting with genes: {test_module[0]}")
    print(f"Found {len(enrichment_results)} pathway enrichments")
    
    # Show top 10 most significant results
    print("\nTop 10 most significant pathways:")
    for i, result in enumerate(enrichment_results[:10]):
        print(f"{i+1:2d}. {result['pathway']}")
        print(f"    P-value: {result['p_value']:.2e}")
        print(f"    Overlap: {result['overlap_size']}/{result['query_size']} genes")
        print(f"    Pathway size: {result['pathway_size']} genes")
        print(f"    Enrichment ratio: {result['enrichment_ratio']:.2f}")
        print(f"    Overlapping genes: {result['overlap_genes']}")
        print()
        
    # Test significant pathways (p < 0.05)
    significant = [r for r in enrichment_results if r['p_value'] <= 0.05]
    print(f"Significant pathways (p <= 0.05): {len(significant)}")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
