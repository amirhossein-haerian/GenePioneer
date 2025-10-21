#!/usr/bin/env python3
"""
Create visual summaries of the module evaluation results
"""
import json
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import numpy as np

def create_visualizations():
    # Load results
    with open('evaluated_modules_result.json', 'r') as f:
        results = json.load(f)
    
    modules_data = results['Prostate_filtered']
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 16))
    
    # 1. Top pathways frequency
    pathway_counts = {}
    all_pvalues = []
    all_enrichments = []
    
    for module in modules_data:
        for pathway in module['significant_pathways']:
            pathway_name = pathway['name']
            if pathway_name not in pathway_counts:
                pathway_counts[pathway_name] = 0
            pathway_counts[pathway_name] += 1
            all_pvalues.append(-np.log10(pathway['p_value']))
            all_enrichments.append(pathway['enrichment_ratio'])
    
    # Top 15 pathways
    top_pathways = dict(sorted(pathway_counts.items(), key=lambda x: x[1], reverse=True)[:15])
    
    plt.subplot(2, 3, 1)
    pathways = list(top_pathways.keys())
    counts = list(top_pathways.values())
    
    bars = plt.barh(range(len(pathways)), counts)
    plt.yticks(range(len(pathways)), [p.replace('HALLMARK_', '') for p in pathways])
    plt.xlabel('Number of Modules')
    plt.title('Top 15 Most Enriched Pathways', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    
    # Color bars by frequency
    for i, bar in enumerate(bars):
        bar.set_color(plt.cm.viridis(counts[i] / max(counts)))
    
    # 2. P-value distribution
    plt.subplot(2, 3, 2)
    plt.hist(all_pvalues, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    plt.axvline(-np.log10(0.05), color='red', linestyle='--', label='p = 0.05')
    plt.axvline(-np.log10(0.01), color='orange', linestyle='--', label='p = 0.01')
    plt.axvline(-np.log10(0.001), color='darkred', linestyle='--', label='p = 0.001')
    plt.xlabel('-log10(P-value)')
    plt.ylabel('Frequency')
    plt.title('Distribution of P-values', fontsize=14, fontweight='bold')
    plt.legend()
    
    # 3. Enrichment ratio distribution
    plt.subplot(2, 3, 3)
    plt.hist(all_enrichments, bins=20, alpha=0.7, color='lightcoral', edgecolor='black')
    plt.axvline(2, color='green', linestyle='--', label='2x enrichment')
    plt.axvline(5, color='orange', linestyle='--', label='5x enrichment')
    plt.axvline(10, color='red', linestyle='--', label='10x enrichment')
    plt.xlabel('Enrichment Ratio')
    plt.ylabel('Frequency')
    plt.title('Distribution of Enrichment Ratios', fontsize=14, fontweight='bold')
    plt.legend()
    plt.xlim(0, 100)
    
    # 4. Module size vs number of pathways
    plt.subplot(2, 3, 4)
    module_sizes = [len(module['module_genes']) for module in modules_data]
    pathway_counts_per_module = [len(module['significant_pathways']) for module in modules_data]
    
    plt.scatter(module_sizes, pathway_counts_per_module, alpha=0.6, s=50)
    plt.xlabel('Module Size (number of genes)')
    plt.ylabel('Number of Significant Pathways')
    plt.title('Module Size vs Pathway Enrichment', fontsize=14, fontweight='bold')
    
    # Add correlation
    correlation = np.corrcoef(module_sizes, pathway_counts_per_module)[0, 1]
    plt.text(0.05, 0.95, f'Correlation: {correlation:.2f}', transform=plt.gca().transAxes)
    
    # 5. CHD1L vs DPF2 modules comparison
    plt.subplot(2, 3, 5)
    chd1l_modules = []
    dpf2_modules = []
    other_modules = []
    
    for module in modules_data:
        genes = module['module_genes']
        pathway_count = len(module['significant_pathways'])
        
        if 'CHD1L' in genes:
            chd1l_modules.append(pathway_count)
        elif 'DPF2' in genes:
            dpf2_modules.append(pathway_count)
        else:
            other_modules.append(pathway_count)
    
    data_to_plot = [chd1l_modules, dpf2_modules, other_modules]
    labels = [f'CHD1L\\n(n={len(chd1l_modules)})', f'DPF2\\n(n={len(dpf2_modules)})', f'Other\\n(n={len(other_modules)})']
    
    plt.boxplot(data_to_plot, labels=labels)
    plt.ylabel('Number of Significant Pathways')
    plt.title('Pathway Enrichment by Module Type', fontsize=14, fontweight='bold')
    
    # 6. Network and quality scores
    plt.subplot(2, 3, 6)
    network_scores = [module['score1'] for module in modules_data]
    quality_scores = [module['score2'] for module in modules_data]
    
    colors = ['red' if 'CHD1L' in module['module_genes'] or 'DPF2' in module['module_genes'] 
              else 'blue' for module in modules_data]
    
    plt.scatter(quality_scores, network_scores, c=colors, alpha=0.6, s=50)
    plt.xlabel('Quality Score')
    plt.ylabel('Network Score')
    plt.title('Module Scores (Red: CHD1L/DPF2 modules)', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('Module_Analysis_Visualizations.png', dpi=300, bbox_inches='tight')
    plt.savefig('Module_Analysis_Visualizations.pdf', bbox_inches='tight')
    # plt.show()  # Commented out to avoid GUI issues
    plt.close()
    
    # Create a summary table for key modules
    priority_modules = []
    for i, module in enumerate(modules_data):
        genes = module['module_genes']
        if 'CHD1L' in genes or 'DPF2' in genes:
            priority_modules.append({
                'Module_ID': f'Module_{i+1}',
                'Genes': ', '.join(genes),
                'Priority_Gene': 'CHD1L' if 'CHD1L' in genes else 'DPF2',
                'Network_Score': f"{module['score1']:.1f}",
                'Quality_Score': f"{module['score2']:.1f}",
                'Pathway_Count': len(module['significant_pathways']),
                'Top_Pathway': module['significant_pathways'][0]['name'].replace('HALLMARK_', ''),
                'Best_P_Value': f"{min(p['p_value'] for p in module['significant_pathways']):.2e}"
            })
    
    priority_df = pd.DataFrame(priority_modules)
    priority_df.to_csv('Priority_Modules_Summary.csv', index=False)
    
    print("\\nPRIORITY MODULES SUMMARY:")
    print("=" * 50)
    print(priority_df.to_string(index=False))
    
    return fig

def main():
    print("Creating visualizations...")
    fig = create_visualizations()
    print("\\nVisualizations saved as:")
    print("- Module_Analysis_Visualizations.png")
    print("- Module_Analysis_Visualizations.pdf")
    print("- Priority_Modules_Summary.csv")

if __name__ == "__main__":
    main()
