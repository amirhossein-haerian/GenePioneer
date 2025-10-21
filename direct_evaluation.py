#!/usr/bin/env python3
"""
Direct module evaluation without full class initialization
"""
import pandas as pd
import json
from scipy.stats import hypergeom

def load_hallmarks():
    hallmarks_path = "genepioneer/Data/Hallmarks_genes_wide.xlsx"
    df = pd.read_excel(hallmarks_path)
    
    hallmarks_dict = {}
    for pathway in df.columns:
        genes = set(df[pathway].dropna().astype(str))
        hallmarks_dict[pathway] = genes
    
    return hallmarks_dict

def manual_pathway_enrichment(query_genes, hallmarks_data, background_size=20000):
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
            'pathway_size': n,
            'query_size': N,
            'overlap_size': overlap_size,
            'overlap_genes': list(overlap),
            'p_value': p_value,
            'enrichment_ratio': (overlap_size / N) / (n / M) if n > 0 else 0
        })
    
    results.sort(key=lambda x: x['p_value'])
    return results

def evaluate_module(module, hallmarks_data):
    genes, score1, score2 = module
    
    enrichment_results = manual_pathway_enrichment(genes, hallmarks_data)
    
    significant_pathways = [
        result for result in enrichment_results
        if result['p_value'] <= 0.05
    ]
    
    if len(significant_pathways) >= 1:
        pathway_info = [
            {
                'name': result['pathway'],
                'p_value': result['p_value'],
                'overlap_size': result['overlap_size'],
                'pathway_size': result['pathway_size'],
                'enrichment_ratio': result['enrichment_ratio'],
                'overlap_genes': result['overlap_genes']
            }
            for result in significant_pathways
        ]
        
        return {
            'module_genes': genes,
            'score1': score1,
            'score2': score2,
            'significant_pathways': pathway_info
        }
    
    return None

def main():
    print("Loading hallmarks data...")
    hallmarks_data = load_hallmarks()
    print(f"Loaded {len(hallmarks_data)} pathways")
    
    print("Loading modules...")
    with open('tests/Prostate_filtered.json', 'r') as f:
        modules = json.load(f)
    
    print(f"Evaluating {len(modules)} modules...")
    
    results = []
    for i, module in enumerate(modules):  # Process ALL modules
        result = evaluate_module(module, hallmarks_data)
        if result:
            results.append(result)
            print(f"Module {i+1}: ACCEPTED ({len(result['significant_pathways'])} pathways)")
        else:
            print(f"Module {i+1}: REJECTED")
        
        # Print progress every 10 modules
        if (i + 1) % 10 == 0:
            print(f"  ... processed {i+1}/{len(modules)} modules")
    
    print(f"\\nFinal results: {len(results)} modules passed evaluation")
    
    # Save results
    final_results = {"Prostate_filtered": results}
    with open('tests/evaluated_modules_result.json', 'w') as f:
        json.dump(final_results, f, indent=2)
    
    print("Results saved to evaluated_modules_result.json")

if __name__ == "__main__":
    main()
