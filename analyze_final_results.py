#!/usr/bin/env python3

import json
import sys
import os

def analyze_updated_module_results():
    """Analyze the updated evaluated_modules_result.json with corrected network size"""
    
    print("UPDATED GENEPIONEER EVALUATION RESULTS")
    print("="*60)
    print("Network Size: 3,812 genes (corrected from 20,000 estimate)")
    print("="*60)
    
    # Load the updated results
    try:
        with open('evaluated_modules_result.json', 'r') as f:
            results = json.load(f)
    except FileNotFoundError:
        print("Error: evaluated_modules_result.json not found")
        return
    
    if "Prostate_filtered" not in results:
        print("Error: No Prostate_filtered results found")
        return
    
    accepted_modules = results["Prostate_filtered"]
    total_modules = 57  # Based on your previous analysis
    
    print(f"\n📊 OVERALL RESULTS SUMMARY:")
    print(f"   • Total modules evaluated: {total_modules}")
    print(f"   • Modules with significant pathways: {len(accepted_modules)}")
    print(f"   • Success rate: {len(accepted_modules)/total_modules*100:.1f}%")
    
    # Count total significant enrichments
    total_enrichments = sum(len(module['significant_pathways']) for module in accepted_modules)
    print(f"   • Total significant pathway enrichments: {total_enrichments}")
    
    # Get unique pathways
    unique_pathways = set()
    for module in accepted_modules:
        for pathway in module['significant_pathways']:
            unique_pathways.add(pathway['name'])
    
    print(f"   • Unique cancer hallmark pathways identified: {len(unique_pathways)}")
    
    # Priority gene analysis
    chd1l_modules = [m for m in accepted_modules if 'CHD1L' in m['module_genes']]
    dpf2_modules = [m for m in accepted_modules if 'DPF2' in m['module_genes']]
    
    print(f"\n🎯 PRIORITY GENE ANALYSIS:")
    print(f"   • Modules containing CHD1L: {len(chd1l_modules)}")
    print(f"   • Modules containing DPF2: {len(dpf2_modules)}")
    
    if chd1l_modules:
        chd1l_pathways = sum(len(m['significant_pathways']) for m in chd1l_modules)
        print(f"   • CHD1L pathway enrichments: {chd1l_pathways}")
    
    if dpf2_modules:
        dpf2_pathways = sum(len(m['significant_pathways']) for m in dpf2_modules)
        print(f"   • DPF2 pathway enrichments: {dpf2_pathways}")
    
    # Most significant enrichments
    print(f"\n🏆 TOP 10 MOST SIGNIFICANT ENRICHMENTS (Updated p-values):")
    all_enrichments = []
    for i, module in enumerate(accepted_modules):
        for pathway in module['significant_pathways']:
            all_enrichments.append({
                'module_id': i + 1,
                'module_genes': module['module_genes'],
                'pathway': pathway['name'],
                'p_value': pathway['p_value'],
                'enrichment_ratio': pathway['enrichment_ratio'],
                'overlap_genes': pathway['overlap_genes']
            })
    
    all_enrichments.sort(key=lambda x: x['p_value'])
    
    for i, enrichment in enumerate(all_enrichments[:10]):
        genes_str = ', '.join(enrichment['module_genes'][:3])
        if len(enrichment['module_genes']) > 3:
            genes_str += "..."
        
        print(f"\n   {i+1:2d}. {enrichment['pathway']}")
        print(f"       Module: [{genes_str}]")
        print(f"       p-value: {enrichment['p_value']:.2e}")
        print(f"       Enrichment: {enrichment['enrichment_ratio']:.1f}x")
        print(f"       Genes: {', '.join(enrichment['overlap_genes'])}")
    
    # Pathway frequency analysis
    print(f"\n📈 MOST FREQUENT CANCER HALLMARKS:")
    pathway_counts = {}
    for module in accepted_modules:
        for pathway in module['significant_pathways']:
            pathway_name = pathway['name']
            if pathway_name not in pathway_counts:
                pathway_counts[pathway_name] = 0
            pathway_counts[pathway_name] += 1
    
    sorted_pathways = sorted(pathway_counts.items(), key=lambda x: x[1], reverse=True)
    
    for i, (pathway, count) in enumerate(sorted_pathways[:10]):
        print(f"   {i+1:2d}. {pathway}: {count} modules")
    
    # Statistical impact analysis
    print(f"\n📊 STATISTICAL IMPACT OF NETWORK SIZE CORRECTION:")
    print(f"   • Previous background: ~20,000 genes")
    print(f"   • Actual background: 3,812 genes (81% reduction)")
    print(f"   • Impact: More stringent p-values, higher biological accuracy")
    print(f"   • Result: {len(accepted_modules)} modules still meet significance criteria")
    
    # Priority gene detailed analysis
    if chd1l_modules or dpf2_modules:
        print(f"\n🔬 PRIORITY GENE DETAILED ANALYSIS:")
        
        if chd1l_modules:
            print(f"\n   CHD1L Modules ({len(chd1l_modules)} total):")
            for i, module in enumerate(chd1l_modules):
                top_pathway = min(module['significant_pathways'], key=lambda x: x['p_value'])
                print(f"   • Module {i+1}: {top_pathway['name']} (p={top_pathway['p_value']:.2e})")
        
        if dpf2_modules:
            print(f"\n   DPF2 Modules ({len(dpf2_modules)} total):")
            for i, module in enumerate(dpf2_modules):
                top_pathway = min(module['significant_pathways'], key=lambda x: x['p_value'])
                print(f"   • Module {i+1}: {top_pathway['name']} (p={top_pathway['p_value']:.2e})")

if __name__ == "__main__":
    analyze_updated_module_results()
