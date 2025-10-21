#!/usr/bin/env python3

import json

def analyze_priority_gene_modules():
    """Extract and analyze only CHD1L and DPF2 modules from updated results"""
    
    print("CHD1L AND DPF2 MODULE ANALYSIS")
    print("="*50)
    print("Network Size: 3,812 genes (corrected)")
    print("="*50)
    
    # Load results
    with open('evaluated_modules_result.json', 'r') as f:
        results = json.load(f)
    
    accepted_modules = results["Prostate_filtered"]
    
    # Filter for CHD1L and DPF2 modules only
    chd1l_modules = [m for m in accepted_modules if 'CHD1L' in m['module_genes']]
    dpf2_modules = [m for m in accepted_modules if 'DPF2' in m['module_genes']]
    
    print(f"🎯 PRIORITY GENE RESULTS:")
    print(f"   • CHD1L modules: {len(chd1l_modules)}")
    print(f"   • DPF2 modules: {len(dpf2_modules)}")
    print(f"   • Total priority modules: {len(chd1l_modules) + len(dpf2_modules)}")
    
    # Analyze CHD1L modules
    if chd1l_modules:
        print(f"\n🧬 CHD1L MODULES ({len(chd1l_modules)} modules):")
        print("-" * 40)
        
        chd1l_total_pathways = 0
        for i, module in enumerate(chd1l_modules):
            print(f"\nModule {i+1}: {module['module_genes']}")
            print(f"Score: {module['score1']:.1f}")
            print(f"Significant pathways: {len(module['significant_pathways'])}")
            chd1l_total_pathways += len(module['significant_pathways'])
            
            # Show ALL pathways for this module
            pathways = sorted(module['significant_pathways'], key=lambda x: x['p_value'])
            for j, pathway in enumerate(pathways):
                print(f"  {j+1}. {pathway['name']}")
                print(f"     p-value: {pathway['p_value']:.2e}")
                print(f"     Enrichment: {pathway['enrichment_ratio']:.1f}x")
                print(f"     Genes: {', '.join(pathway['overlap_genes'])}")
        
        print(f"\nCHD1L Summary: {chd1l_total_pathways} total pathway enrichments")
    
    # Analyze DPF2 modules
    if dpf2_modules:
        print(f"\n🧬 DPF2 MODULES ({len(dpf2_modules)} modules):")
        print("-" * 40)
        
        dpf2_total_pathways = 0
        for i, module in enumerate(dpf2_modules):
            print(f"\nModule {i+1}: {module['module_genes']}")
            print(f"Score: {module['score1']:.1f}")
            print(f"Significant pathways: {len(module['significant_pathways'])}")
            dpf2_total_pathways += len(module['significant_pathways'])
            
            # Show ALL pathways for this module
            pathways = sorted(module['significant_pathways'], key=lambda x: x['p_value'])
            for j, pathway in enumerate(pathways):
                print(f"  {j+1}. {pathway['name']}")
                print(f"     p-value: {pathway['p_value']:.2e}")
                print(f"     Enrichment: {pathway['enrichment_ratio']:.1f}x")
                print(f"     Genes: {', '.join(pathway['overlap_genes'])}")
        
        print(f"\nDPF2 Summary: {dpf2_total_pathways} total pathway enrichments")
    
    # Combined analysis
    all_priority_modules = chd1l_modules + dpf2_modules
    if all_priority_modules:
        print(f"\n📊 COMBINED PRIORITY GENE ANALYSIS:")
        print("-" * 40)
        
        # Get all pathways from priority modules
        all_priority_enrichments = []
        for module in all_priority_modules:
            for pathway in module['significant_pathways']:
                all_priority_enrichments.append({
                    'gene': 'CHD1L' if 'CHD1L' in module['module_genes'] else 'DPF2',
                    'module': module['module_genes'],
                    'pathway': pathway['name'],
                    'p_value': pathway['p_value'],
                    'enrichment_ratio': pathway['enrichment_ratio'],
                    'overlap_genes': pathway['overlap_genes']
                })
        
        # Sort by significance
        all_priority_enrichments.sort(key=lambda x: x['p_value'])
        
        print(f"Total priority gene enrichments: {len(all_priority_enrichments)}")
        
        # Top 10 most significant from priority genes
        print(f"\n🏆 TOP 10 PRIORITY GENE ENRICHMENTS:")
        for i, enrich in enumerate(all_priority_enrichments[:10]):
            print(f"\n{i+1:2d}. {enrich['pathway']} ({enrich['gene']})")
            print(f"    p-value: {enrich['p_value']:.2e}")
            print(f"    Enrichment: {enrich['enrichment_ratio']:.1f}x")
            print(f"    Module: {enrich['module']}")
            print(f"    Overlap genes: {', '.join(enrich['overlap_genes'])}")
        
        # Pathway frequency in priority modules
        priority_pathway_counts = {}
        for enrich in all_priority_enrichments:
            pathway = enrich['pathway']
            if pathway not in priority_pathway_counts:
                priority_pathway_counts[pathway] = 0
            priority_pathway_counts[pathway] += 1
        
        sorted_priority_pathways = sorted(priority_pathway_counts.items(), key=lambda x: x[1], reverse=True)
        
        print(f"\n📈 MOST FREQUENT PATHWAYS IN PRIORITY MODULES:")
        for i, (pathway, count) in enumerate(sorted_priority_pathways[:8]):
            print(f"   {i+1}. {pathway}: {count} enrichments")
    
    else:
        print("\n❌ NO MODULES WITH CHD1L OR DPF2 FOUND")
        print("This could indicate that the corrected network size made the")
        print("statistical criteria too stringent for these specific modules.")

if __name__ == "__main__":
    analyze_priority_gene_modules()
