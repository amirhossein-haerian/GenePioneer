#!/usr/bin/env python3
"""
Validation Analysis on New Cohort
This script validates the CHD1L biomarker findings on a new cohort.
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from collections import defaultdict

# Add project to path
sys.path.append('.')

from genepioneer.network_builder import NetworkBuilder
from genepioneer.network_analysis import NetworkAnalysis
from genepioneer.evaluation import Evaluation

def load_mutation_data(file_path):
    """
    Load mutation data from MAF-like file
    Returns a set of mutated genes
    """
    print(f"Loading mutation data from: {file_path}")
    
    try:
        df = pd.read_csv(file_path, sep='\t', comment='#')
        
        # Find Hugo_Symbol column (case-insensitive)
        symbol_col = None
        for col in df.columns:
            if col.upper() == 'HUGO_SYMBOL':
                symbol_col = col
                break
        
        if symbol_col is None:
            raise ValueError("Could not find Hugo_Symbol column in mutation file")
        
        mutated_genes = set(df[symbol_col].dropna().unique())
        print(f"Found {len(mutated_genes)} mutated genes")
        return mutated_genes
    
    except Exception as e:
        print(f"Error loading mutation data: {e}")
        return set()

def load_cna_data(file_path, threshold=0):
    """
    Load copy number alteration data
    Returns a set of genes with CNAs
    """
    print(f"Loading CNA data from: {file_path}")
    
    try:
        df = pd.read_csv(file_path, sep='\t')
        
        # First column should be Hugo_Symbol
        gene_col = df.columns[0]
        
        # Get genes with any CNA (absolute value > threshold)
        cna_genes = set()
        for idx, row in df.iterrows():
            gene = row[gene_col]
            if pd.isna(gene) or gene == '':
                continue
            
            # Check if any sample has CNA
            numeric_values = pd.to_numeric(row[1:], errors='coerce')
            if (numeric_values.abs() > threshold).any():
                cna_genes.add(gene)
        
        print(f"Found {len(cna_genes)} genes with CNAs")
        return cna_genes
    
    except Exception as e:
        print(f"Error loading CNA data: {e}")
        return set()

def check_chd1l_presence(mutated_genes, cna_genes):
    """
    Check if CHD1L is in the dataset
    """
    in_mutations = 'CHD1L' in mutated_genes
    in_cna = 'CHD1L' in cna_genes
    
    print(f"\nCHD1L presence check:")
    print(f"  In mutations: {in_mutations}")
    print(f"  In CNAs: {in_cna}")
    print(f"  Overall: {in_mutations or in_cna}")
    
    return in_mutations or in_cna

def create_validation_network(mutated_genes, cna_genes, output_dir="./validation_cohort"):
    """
    Create network for validation cohort
    Similar to the original TCGA cohort
    """
    print("\n" + "="*60)
    print("CREATING VALIDATION NETWORK")
    print("="*60)
    
    # Combine mutations and CNAs
    all_altered_genes = mutated_genes.union(cna_genes)
    print(f"\nTotal altered genes (mutations + CNAs): {len(all_altered_genes)}")
    
    # Check CHD1L presence
    chd1l_present = 'CHD1L' in all_altered_genes
    print(f"CHD1L in network: {chd1l_present}")
    
    if not chd1l_present:
        print("\n⚠️  WARNING: CHD1L is not among the altered genes!")
        print("The biomarker cannot be validated without CHD1L in the network.")
        print("\nOptions:")
        print("1. Add CHD1L manually if it has clinical relevance")
        print("2. Expand the gene list with pathway/interaction partners")
        return None, None
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save gene list
    gene_list_file = os.path.join(output_dir, "altered_genes.txt")
    with open(gene_list_file, 'w') as f:
        for gene in sorted(all_altered_genes):
            f.write(f"{gene}\n")
    print(f"\nSaved {len(all_altered_genes)} genes to: {gene_list_file}")
    
    # Build network
    print("\nBuilding network...")
    network_builder = NetworkBuilder(
        cancer_type="ValidationCohort",
        genes_list=list(all_altered_genes),
        data_path="./GenesData"
    )
    
    # Build STRING network
    network_builder.build_string_network()
    
    # Get feature dictionary
    features = network_builder.feature_dict
    print(f"Network created with {len(features)} nodes")
    
    # Verify CHD1L is in the network
    if 'CHD1L' in features:
        print("✓ CHD1L successfully included in the network")
    else:
        print("✗ CHD1L not in the final network (possibly isolated)")
    
    return network_builder, features

def detect_modules(network_builder, prioritized_genes=["CHD1L"], 
                   min_size=3, max_size=10, output_dir="./validation_cohort"):
    """
    Detect modules prioritizing CHD1L
    Uses the same algorithm as the original TCGA analysis
    """
    print("\n" + "="*60)
    print("MODULE DETECTION")
    print("="*60)
    
    print(f"\nPrioritized genes: {prioritized_genes}")
    print(f"Module size range: {min_size} - {max_size}")
    
    # Create NetworkAnalysis object
    network_analysis = NetworkAnalysis(
        cancer_type="ValidationCohort",
        features=network_builder.feature_dict
    )
    
    # Save network features
    network_file = os.path.join(output_dir, "ValidationCohort_network_features.gml")
    network_builder.GNX.write_gml(network_file)
    print(f"\nNetwork saved to: {network_file}")
    
    # Detect modules
    print("\nDetecting modules...")
    modules = network_analysis.module_detection(
        min_comm_size=min_size,
        max_comm_size=max_size,
        prioritized_genes=prioritized_genes
    )
    
    print(f"\nDetected {len(modules)} modules")
    
    # Find modules containing CHD1L
    chd1l_modules = []
    for module, score, quality in modules:
        if 'CHD1L' in module:
            chd1l_modules.append((module, score, quality))
    
    print(f"Modules containing CHD1L: {len(chd1l_modules)}")
    
    # Save modules
    modules_file = os.path.join(output_dir, "ValidationCohort_modules.json")
    with open(modules_file, 'w') as f:
        json.dump(modules, f, indent=2)
    print(f"Modules saved to: {modules_file}")
    
    # Save CHD1L-specific modules
    if chd1l_modules:
        chd1l_modules_file = os.path.join(output_dir, "CHD1L_modules.json")
        with open(chd1l_modules_file, 'w') as f:
            json.dump(chd1l_modules, f, indent=2)
        print(f"CHD1L modules saved to: {chd1l_modules_file}")
        
        # Print CHD1L modules
        print("\n" + "-"*60)
        print("CHD1L MODULES:")
        print("-"*60)
        for i, (module, score, quality) in enumerate(chd1l_modules, 1):
            print(f"\nModule {i} (Score: {score:.4f}, Quality: {quality:.4f}):")
            print(f"  Genes: {', '.join(sorted(module))}")
            print(f"  Size: {len(module)}")
    
    return modules, chd1l_modules

def evaluate_modules(modules, output_dir="./validation_cohort"):
    """
    Evaluate modules using pathway enrichment
    """
    print("\n" + "="*60)
    print("MODULE EVALUATION")
    print("="*60)
    
    # Prepare modules in the format expected by Evaluation
    module_dict = {"ValidationCohort": modules}
    
    # Create evaluation object
    print("\nInitializing evaluation...")
    evaluator = Evaluation(data_path="./GenesData")
    
    print(f"Loaded {len(evaluator.hallmarks_data)} hallmark pathways")
    
    # Evaluate modules
    print("\nEvaluating modules...")
    results = evaluator.evaluate_modules(module_dict)
    
    # Save results
    results_file = os.path.join(output_dir, "module_evaluation_results.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_file}")
    
    # Print summary
    if "ValidationCohort" in results:
        validated_modules = results["ValidationCohort"]
        print(f"\nValidated modules: {len(validated_modules)}")
        
        # Print CHD1L module results
        for i, module_result in enumerate(validated_modules, 1):
            if 'CHD1L' in module_result['module_genes']:
                print(f"\n{'='*60}")
                print(f"CHD1L Module {i}")
                print(f"{'='*60}")
                print(f"Genes: {', '.join(sorted(module_result['module_genes']))}")
                print(f"Score1 (Coverage): {module_result['score1']:.4f}")
                print(f"Score2 (Pathway enrichment): {module_result['score2']:.4f}")
                print(f"\nSignificant pathways ({len(module_result['significant_pathways'])}):")
                for pathway in module_result['significant_pathways']:
                    print(f"  - {pathway['name']}: p={pathway['p_value']:.2e}, "
                          f"overlap={pathway['overlap']}/{pathway['pathway_size']}")
    
    return results

def compare_with_tcga(validation_results, tcga_results_file="evaluated_modules_result.json"):
    """
    Compare validation cohort results with original TCGA results
    """
    print("\n" + "="*60)
    print("COMPARISON WITH TCGA COHORT")
    print("="*60)
    
    if not os.path.exists(tcga_results_file):
        print(f"⚠️  TCGA results file not found: {tcga_results_file}")
        return
    
    # Load TCGA results
    with open(tcga_results_file, 'r') as f:
        tcga_results = json.load(f)
    
    # Extract CHD1L modules from both cohorts
    validation_chd1l_modules = []
    if "ValidationCohort" in validation_results:
        for module in validation_results["ValidationCohort"]:
            if 'CHD1L' in module['module_genes']:
                validation_chd1l_modules.append(module)
    
    tcga_chd1l_modules = []
    for cancer_type, modules in tcga_results.items():
        for module in modules:
            if 'CHD1L' in module['module_genes']:
                tcga_chd1l_modules.append(module)
    
    print(f"\nTCGA CHD1L modules: {len(tcga_chd1l_modules)}")
    print(f"Validation CHD1L modules: {len(validation_chd1l_modules)}")
    
    if not validation_chd1l_modules:
        print("\n⚠️  No CHD1L modules found in validation cohort")
        return
    
    if not tcga_chd1l_modules:
        print("\n⚠️  No CHD1L modules found in TCGA cohort")
        return
    
    # Compare pathways
    print("\n" + "-"*60)
    print("PATHWAY COMPARISON")
    print("-"*60)
    
    # Get all pathways from TCGA CHD1L modules
    tcga_pathways = set()
    for module in tcga_chd1l_modules:
        for pathway in module.get('significant_pathways', []):
            tcga_pathways.add(pathway['name'])
    
    # Get all pathways from validation CHD1L modules
    validation_pathways = set()
    for module in validation_chd1l_modules:
        for pathway in module.get('significant_pathways', []):
            validation_pathways.add(pathway['name'])
    
    # Find overlapping pathways
    common_pathways = tcga_pathways.intersection(validation_pathways)
    
    print(f"\nTCGA pathways: {len(tcga_pathways)}")
    print(f"Validation pathways: {len(validation_pathways)}")
    print(f"Common pathways: {len(common_pathways)}")
    
    if common_pathways:
        print("\nCommon pathways:")
        for pathway in sorted(common_pathways):
            print(f"  ✓ {pathway}")
    
    # Calculate Jaccard similarity
    if tcga_pathways or validation_pathways:
        jaccard = len(common_pathways) / len(tcga_pathways.union(validation_pathways))
        print(f"\nJaccard similarity: {jaccard:.3f}")
    
    # Compare gene overlap
    print("\n" + "-"*60)
    print("GENE OVERLAP")
    print("-"*60)
    
    for v_idx, v_module in enumerate(validation_chd1l_modules, 1):
        v_genes = set(v_module['module_genes'])
        
        print(f"\nValidation Module {v_idx} ({len(v_genes)} genes):")
        print(f"  Genes: {', '.join(sorted(v_genes))}")
        
        best_overlap = 0
        best_tcga_module = None
        
        for t_idx, t_module in enumerate(tcga_chd1l_modules):
            t_genes = set(t_module['module_genes'])
            overlap = len(v_genes.intersection(t_genes))
            
            if overlap > best_overlap:
                best_overlap = overlap
                best_tcga_module = (t_idx, t_module)
        
        if best_tcga_module:
            t_idx, t_module = best_tcga_module
            t_genes = set(t_module['module_genes'])
            jaccard_genes = best_overlap / len(v_genes.union(t_genes))
            
            print(f"  Best TCGA match: Module {t_idx + 1}")
            print(f"    Overlap: {best_overlap} genes")
            print(f"    Jaccard: {jaccard_genes:.3f}")
            print(f"    Common genes: {', '.join(sorted(v_genes.intersection(t_genes)))}")

def main():
    """
    Main validation pipeline
    """
    print("="*70)
    print("VALIDATION COHORT ANALYSIS")
    print("Validating CHD1L biomarker on new cohort")
    print("="*70)
    
    # File paths
    mutation_file = "./data_mutations.txt"
    cna_file = "./data_cna.txt"
    output_dir = "./validation_cohort"
    
    # Step 1: Load data
    print("\n" + "="*60)
    print("STEP 1: LOADING DATA")
    print("="*60)
    
    mutated_genes = load_mutation_data(mutation_file)
    cna_genes = load_cna_data(cna_file)
    
    # Check CHD1L presence
    chd1l_present = check_chd1l_presence(mutated_genes, cna_genes)
    
    if not chd1l_present:
        print("\n" + "!"*60)
        print("CHD1L NOT FOUND IN DATASET")
        print("!"*60)
        print("\nCannot proceed with validation.")
        print("Consider:")
        print("1. Manually adding CHD1L if it has clinical significance")
        print("2. Expanding gene set with CHD1L pathway partners")
        return
    
    # Step 2: Create network
    print("\n" + "="*60)
    print("STEP 2: NETWORK CONSTRUCTION")
    print("="*60)
    
    network_builder, features = create_validation_network(
        mutated_genes, cna_genes, output_dir
    )
    
    if network_builder is None:
        print("\nNetwork creation failed. Exiting.")
        return
    
    # Step 3: Detect modules (prioritizing CHD1L)
    print("\n" + "="*60)
    print("STEP 3: MODULE DETECTION")
    print("="*60)
    
    all_modules, chd1l_modules = detect_modules(
        network_builder,
        prioritized_genes=["CHD1L"],
        min_size=3,
        max_size=10,
        output_dir=output_dir
    )
    
    # Step 4: Evaluate modules
    print("\n" + "="*60)
    print("STEP 4: MODULE EVALUATION")
    print("="*60)
    
    validation_results = evaluate_modules(all_modules, output_dir)
    
    # Step 5: Compare with TCGA
    print("\n" + "="*60)
    print("STEP 5: COMPARISON WITH TCGA")
    print("="*60)
    
    compare_with_tcga(validation_results)
    
    print("\n" + "="*70)
    print("VALIDATION ANALYSIS COMPLETE")
    print("="*70)
    print(f"\nResults saved in: {output_dir}/")
    print("\nGenerated files:")
    print("  - altered_genes.txt: List of altered genes")
    print("  - ValidationCohort_network_features.gml: Network file")
    print("  - ValidationCohort_modules.json: All detected modules")
    print("  - CHD1L_modules.json: CHD1L-containing modules")
    print("  - module_evaluation_results.json: Pathway enrichment results")

if __name__ == "__main__":
    main()
