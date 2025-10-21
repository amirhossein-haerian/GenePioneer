#!/usr/bin/env python3
"""
Generate a comprehensive, human-readable report from module evaluation results
"""
import json
import pandas as pd
from datetime import datetime

def load_evaluation_results():
    """Load the evaluation results"""
    with open('evaluated_modules_result.json', 'r') as f:
        return json.load(f)

def get_pathway_descriptions():
    """Return descriptions for major cancer hallmark pathways"""
    descriptions = {
        'HALLMARK_APOPTOSIS': 'Programmed cell death - critical for eliminating damaged cells and preventing cancer',
        'HALLMARK_DNA_REPAIR': 'DNA damage response and repair mechanisms - defects lead to genomic instability',
        'HALLMARK_P53_PATHWAY': 'p53 tumor suppressor pathway - "guardian of the genome", prevents cancer formation',
        'HALLMARK_MYC_TARGETS_V1': 'MYC oncogene targets (set 1) - promotes cell proliferation and transformation',
        'HALLMARK_MYC_TARGETS_V2': 'MYC oncogene targets (set 2) - additional proliferation and metabolic targets',
        'HALLMARK_E2F_TARGETS': 'E2F transcription factor targets - cell cycle progression and DNA replication',
        'HALLMARK_G2M_CHECKPOINT': 'G2/M cell cycle checkpoint - ensures proper chromosome segregation',
        'HALLMARK_ANGIOGENESIS': 'Blood vessel formation - tumors need new blood vessels to grow',
        'HALLMARK_EPITHELIAL_MESENCHYMAL_TRANSITION': 'EMT - cancer cells gain ability to invade and metastasize',
        'HALLMARK_HYPOXIA': 'Low oxygen response - tumor microenvironment adaptation',
        'HALLMARK_INFLAMMATORY_RESPONSE': 'Inflammation - can promote or suppress tumor development',
        'HALLMARK_TNFA_SIGNALING_VIA_NFKB': 'TNF-alpha/NF-kB signaling - inflammation and cell survival',
        'HALLMARK_INTERFERON_GAMMA_RESPONSE': 'Interferon gamma response - immune system activation',
        'HALLMARK_INTERFERON_ALPHA_RESPONSE': 'Interferon alpha response - antiviral and immune response',
        'HALLMARK_IL6_JAK_STAT3_SIGNALING': 'IL-6/JAK/STAT3 pathway - inflammation and cell proliferation',
        'HALLMARK_IL2_STAT5_SIGNALING': 'IL-2/STAT5 pathway - T cell activation and proliferation',
        'HALLMARK_COMPLEMENT': 'Complement system - immune response and inflammation',
        'HALLMARK_KRAS_SIGNALING_UP': 'KRAS oncogene upregulated targets - promotes cell growth',
        'HALLMARK_KRAS_SIGNALING_DN': 'KRAS oncogene downregulated targets - growth inhibition',
        'HALLMARK_PI3K_AKT_MTOR_SIGNALING': 'PI3K/AKT/mTOR pathway - cell survival and growth',
        'HALLMARK_NOTCH_SIGNALING': 'Notch signaling - cell fate determination and differentiation',
        'HALLMARK_WNT_BETA_CATENIN_SIGNALING': 'Wnt signaling - cell proliferation and differentiation',
        'HALLMARK_TGF_BETA_SIGNALING': 'TGF-beta signaling - growth inhibition or promotion depending on context',
        'HALLMARK_HEDGEHOG_SIGNALING': 'Hedgehog signaling - development and stem cell maintenance',
        'HALLMARK_ANDROGEN_RESPONSE': 'Androgen receptor signaling - particularly relevant in prostate cancer',
        'HALLMARK_ESTROGEN_RESPONSE_EARLY': 'Early estrogen response genes - hormone-dependent growth',
        'HALLMARK_ESTROGEN_RESPONSE_LATE': 'Late estrogen response genes - sustained hormone effects',
        'HALLMARK_OXIDATIVE_PHOSPHORYLATION': 'Mitochondrial energy production - metabolic reprogramming',
        'HALLMARK_GLYCOLYSIS': 'Glucose metabolism - cancer cells often rely on glycolysis',
        'HALLMARK_FATTY_ACID_METABOLISM': 'Fat metabolism - alternative energy source for cancer cells',
        'HALLMARK_CHOLESTEROL_HOMEOSTASIS': 'Cholesterol metabolism - membrane synthesis and signaling',
        'HALLMARK_MTORC1_SIGNALING': 'mTORC1 complex signaling - protein synthesis and cell growth',
        'HALLMARK_UNFOLDED_PROTEIN_RESPONSE': 'ER stress response - protein folding quality control',
        'HALLMARK_UV_RESPONSE_DN': 'UV radiation response (downregulated) - DNA damage response',
        'HALLMARK_UV_RESPONSE_UP': 'UV radiation response (upregulated) - DNA damage response',
        'HALLMARK_REACTIVE_OXYGEN_SPECIES_PATHWAY': 'ROS pathway - oxidative stress and DNA damage',
        'HALLMARK_MITOTIC_SPINDLE': 'Mitotic spindle assembly - chromosome segregation',
        'HALLMARK_COAGULATION': 'Blood coagulation - thrombosis and metastasis',
        'HALLMARK_APICAL_JUNCTION': 'Cell-cell junctions - tissue integrity and polarity',
        'HALLMARK_APICAL_SURFACE': 'Apical cell surface - epithelial cell polarity',
        'HALLMARK_MYOGENESIS': 'Muscle development - differentiation programs',
        'HALLMARK_ADIPOGENESIS': 'Fat cell development - metabolic reprogramming',
        'HALLMARK_PEROXISOME': 'Peroxisome function - fatty acid oxidation',
        'HALLMARK_HEME_METABOLISM': 'Heme biosynthesis - oxygen transport and signaling',
        'HALLMARK_BILE_ACID_METABOLISM': 'Bile acid metabolism - lipid digestion and signaling',
        'HALLMARK_XENOBIOTIC_METABOLISM': 'Drug and toxin metabolism - cellular detoxification',
        'HALLMARK_PROTEIN_SECRETION': 'Protein secretion - cellular communication',
        'HALLMARK_SPERMATOGENESIS': 'Sperm development - male reproductive function',
        'HALLMARK_PANCREAS_BETA_CELLS': 'Pancreatic beta cell function - insulin production',
        'HALLMARK_ALLOGRAFT_REJECTION': 'Immune rejection - tissue transplantation response'
    }
    return descriptions

def interpret_significance(p_value, enrichment_ratio, overlap_size):
    """Interpret the biological significance of enrichment results"""
    significance_level = ""
    if p_value < 0.001:
        significance_level = "Highly Significant (p < 0.001)"
    elif p_value < 0.01:
        significance_level = "Very Significant (p < 0.01)"
    elif p_value < 0.05:
        significance_level = "Significant (p < 0.05)"
    else:
        significance_level = "Not Significant (p ≥ 0.05)"
    
    enrichment_strength = ""
    if enrichment_ratio > 10:
        enrichment_strength = "Very Strong Enrichment (>10x expected)"
    elif enrichment_ratio > 5:
        enrichment_strength = "Strong Enrichment (5-10x expected)"
    elif enrichment_ratio > 2:
        enrichment_strength = "Moderate Enrichment (2-5x expected)"
    elif enrichment_ratio > 1:
        enrichment_strength = "Weak Enrichment (1-2x expected)"
    else:
        enrichment_strength = "No Enrichment (≤1x expected)"
    
    confidence = ""
    if overlap_size >= 3:
        confidence = "High Confidence (≥3 overlapping genes)"
    elif overlap_size == 2:
        confidence = "Moderate Confidence (2 overlapping genes)"
    else:
        confidence = "Low Confidence (1 overlapping gene)"
    
    return significance_level, enrichment_strength, confidence

def generate_module_report(modules_data, output_file="Module_Evaluation_Report.md"):
    """Generate a comprehensive markdown report"""
    pathway_descriptions = get_pathway_descriptions()
    
    report = []
    report.append("# Cancer Module Pathway Enrichment Analysis Report")
    report.append(f"**Generated on:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"**Dataset:** Prostate Cancer Filtered Network")
    report.append(f"**Total Modules Evaluated:** {len(modules_data)}")
    report.append("")
    
    # Summary statistics
    report.append("## Summary Statistics")
    report.append("")
    
    all_pathways = set()
    total_significant_pathways = 0
    highly_significant_count = 0
    
    for module in modules_data:
        for pathway in module['significant_pathways']:
            all_pathways.add(pathway['name'])
            total_significant_pathways += 1
            if pathway['p_value'] < 0.001:
                highly_significant_count += 1
    
    report.append(f"- **Modules with Significant Pathways:** {len(modules_data)}")
    report.append(f"- **Unique Pathways Enriched:** {len(all_pathways)}")
    report.append(f"- **Total Significant Enrichments:** {total_significant_pathways}")
    report.append(f"- **Highly Significant Enrichments (p < 0.001):** {highly_significant_count}")
    report.append("")
    
    # Key findings
    report.append("## Key Findings")
    report.append("")
    
    # Count pathway frequencies
    pathway_counts = {}
    for module in modules_data:
        for pathway in module['significant_pathways']:
            pathway_name = pathway['name']
            if pathway_name not in pathway_counts:
                pathway_counts[pathway_name] = 0
            pathway_counts[pathway_name] += 1
    
    # Top pathways
    top_pathways = sorted(pathway_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    
    report.append("### Most Frequently Enriched Pathways")
    report.append("")
    for pathway, count in top_pathways:
        description = pathway_descriptions.get(pathway, "No description available")
        report.append(f"1. **{pathway}** (enriched in {count} modules)")
        report.append(f"   - *{description}*")
        report.append("")
    
    # Detailed module analysis
    report.append("## Detailed Module Analysis")
    report.append("")
    
    for i, module in enumerate(modules_data, 1):
        genes = module['module_genes']
        score1 = module['score1']
        score2 = module['score2']
        pathways = module['significant_pathways']
        
        report.append(f"### Module {i}: {', '.join(genes)}")
        report.append("")
        
        # Check if it contains priority genes
        priority_genes = [g for g in genes if g in ['CHD1L', 'DPF2']]
        if priority_genes:
            report.append(f"🎯 **Priority Module** - Contains: {', '.join(priority_genes)}")
            report.append("")
        
        report.append(f"**Module Composition:**")
        report.append(f"- **Genes:** {', '.join(genes)} ({len(genes)} genes)")
        report.append(f"- **Network Score:** {score1:.2f}")
        report.append(f"- **Quality Score:** {score2:.2f}")
        report.append(f"- **Significant Pathways:** {len(pathways)}")
        report.append("")
        
        # Sort pathways by p-value
        sorted_pathways = sorted(pathways, key=lambda x: x['p_value'])
        
        report.append("**Pathway Enrichment Results:**")
        report.append("")
        
        for j, pathway in enumerate(sorted_pathways, 1):
            pathway_name = pathway['name']
            p_value = pathway['p_value']
            enrichment_ratio = pathway['enrichment_ratio']
            overlap_size = pathway['overlap_size']
            overlap_genes = pathway['overlap_genes']
            pathway_size = pathway['pathway_size']
            
            significance_level, enrichment_strength, confidence = interpret_significance(
                p_value, enrichment_ratio, overlap_size
            )
            
            description = pathway_descriptions.get(pathway_name, "No description available")
            
            report.append(f"{j}. **{pathway_name}**")
            report.append(f"   - **Description:** {description}")
            report.append(f"   - **P-value:** {p_value:.2e} ({significance_level})")
            report.append(f"   - **Enrichment Ratio:** {enrichment_ratio:.1f}x ({enrichment_strength})")
            report.append(f"   - **Overlap:** {overlap_size}/{len(genes)} module genes in pathway of {pathway_size} genes ({confidence})")
            report.append(f"   - **Overlapping Genes:** {', '.join(overlap_genes)}")
            
            # Add biological interpretation
            if p_value < 0.01 and enrichment_ratio > 5:
                report.append(f"   - **⭐ Biological Significance:** This pathway is strongly enriched in this module, suggesting these genes work together in this cancer-related process.")
            elif p_value < 0.05 and enrichment_ratio > 2:
                report.append(f"   - **✓ Biological Relevance:** This pathway shows meaningful enrichment, indicating potential functional relationships.")
            
            report.append("")
        
        report.append("---")
        report.append("")
    
    # Methodology explanation
    report.append("## Methodology & Interpretation Guide")
    report.append("")
    report.append("### Statistical Methods")
    report.append("- **Enrichment Analysis:** Hypergeometric test comparing module genes to cancer hallmark pathways")
    report.append("- **P-value:** Probability that the observed overlap occurred by chance")
    report.append("- **Enrichment Ratio:** Fold-enrichment compared to random expectation")
    report.append("- **Background:** 20,000 genes used as genomic background")
    report.append("")
    
    report.append("### Significance Levels")
    report.append("- **p < 0.001:** Highly significant (very strong evidence)")
    report.append("- **p < 0.01:** Very significant (strong evidence)")
    report.append("- **p < 0.05:** Significant (meaningful evidence)")
    report.append("")
    
    report.append("### Enrichment Interpretation")
    report.append("- **>10x enrichment:** Very strong functional relationship")
    report.append("- **5-10x enrichment:** Strong functional relationship")
    report.append("- **2-5x enrichment:** Moderate functional relationship")
    report.append("- **1-2x enrichment:** Weak functional relationship")
    report.append("")
    
    report.append("### Cancer Hallmark Pathways")
    report.append("These represent fundamental cancer processes identified by Hanahan & Weinberg:")
    report.append("- Sustaining proliferative signaling")
    report.append("- Evading growth suppressors")
    report.append("- Resisting cell death")
    report.append("- Enabling replicative immortality")
    report.append("- Inducing angiogenesis")
    report.append("- Activating invasion and metastasis")
    report.append("- Reprogramming energy metabolism")
    report.append("- Evading immune destruction")
    report.append("")
    
    # Write to file
    with open(output_file, 'w') as f:
        f.write('\n'.join(report))
    
    return output_file

def main():
    print("Loading evaluation results...")
    results = load_evaluation_results()
    
    if 'Prostate_filtered' not in results or not results['Prostate_filtered']:
        print("No results found for Prostate_filtered")
        return
    
    modules_data = results['Prostate_filtered']
    print(f"Found {len(modules_data)} modules with significant pathway enrichments")
    
    print("Generating comprehensive report...")
    output_file = generate_module_report(modules_data)
    print(f"Report saved to: {output_file}")
    
    # Also generate a CSV summary for easy analysis
    csv_data = []
    for i, module in enumerate(modules_data, 1):
        for pathway in module['significant_pathways']:
            csv_data.append({
                'Module_ID': f"Module_{i}",
                'Module_Genes': ', '.join(module['module_genes']),
                'Contains_CHD1L': 'Yes' if 'CHD1L' in module['module_genes'] else 'No',
                'Contains_DPF2': 'Yes' if 'DPF2' in module['module_genes'] else 'No',
                'Network_Score': module['score1'],
                'Quality_Score': module['score2'],
                'Pathway_Name': pathway['name'],
                'P_Value': pathway['p_value'],
                'Enrichment_Ratio': pathway['enrichment_ratio'],
                'Overlap_Size': pathway['overlap_size'],
                'Pathway_Size': pathway['pathway_size'],
                'Overlapping_Genes': ', '.join(pathway['overlap_genes'])
            })
    
    df = pd.DataFrame(csv_data)
    csv_file = "Module_Pathway_Enrichment_Summary.csv"
    df.to_csv(csv_file, index=False)
    print(f"CSV summary saved to: {csv_file}")

if __name__ == "__main__":
    main()
