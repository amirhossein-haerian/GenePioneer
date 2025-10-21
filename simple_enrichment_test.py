#!/usr/bin/env python3
"""
Standalone test for pathway enrichment
"""
import pandas as pd
import os
from scipy.stats import hypergeom
import json

def load_hallmarks():
    """Load hallmarks data"""
    hallmarks_path = "genepioneer/Data/Hallmarks_genes_wide.xlsx"
    df = pd.read_excel(hallmarks_path)
    
    hallmarks_dict = {}
    for pathway in df.columns:
        genes = set(df[pathway].dropna().astype(str))
        hallmarks_dict[pathway] = genes
    
    return hallmarks_dict

def manual_enrichment(query_genes, hallmarks_data, background_size=20000):
    """Manual pathway enrichment"""
    query_genes = set(str(gene).upper() for gene in query_genes)
    results = []
    
    for pathway_name, pathway_genes in hallmarks_data.items():
        pathway_genes_upper = set(str(gene).upper() for gene in pathway_genes)
        
        overlap = query_genes.intersection(pathway_genes_upper)
        overlap_size = len(overlap)
        
        if overlap_size == 0:
            continue
            
        M = background_size
        n = len(pathway_genes_upper)
        N = len(query_genes)
        k = overlap_size
        
        p_value = hypergeom.sf(k - 1, M, n, N)
        
        results.append({
            'pathway': pathway_name,
            'p_value': p_value,
            'overlap_size': overlap_size,
            'overlap_genes': list(overlap)
        })
    
    results.sort(key=lambda x: x['p_value'])
    return results

def main():
    print("Loading hallmarks data...")
    hallmarks = load_hallmarks()
    print(f"Loaded {len(hallmarks)} pathways")
    
    # Load test module
    with open('tests/Prostate_filtered.json', 'r') as f:
        modules = json.load(f)
    
    test_genes = modules[0][0]  # First module genes
    print(f"Testing genes: {test_genes}")
    
    # Run enrichment
    results = manual_enrichment(test_genes, hallmarks)
    print(f"Found {len(results)} enrichment results")
    
    # Show top 10
    print("\\nTop 10 results:")
    for i, result in enumerate(results[:10]):
        print(f"{i+1:2d}. {result['pathway']}")
        print(f"     P-value: {result['p_value']:.2e}")
        print(f"     Overlap: {result['overlap_size']} genes: {result['overlap_genes']}")
    
    # Check significant
    significant = [r for r in results if r['p_value'] <= 0.05]
    print(f"\\nSignificant results (p <= 0.05): {len(significant)}")
    
    if significant:
        print("Significant pathways:")
        for result in significant:
            print(f"  - {result['pathway']}: p={result['p_value']:.2e}")

if __name__ == "__main__":
    main()
