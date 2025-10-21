#!/usr/bin/env python3
"""
Validation Analysis Pipeline - Flexible Execution
This script allows you to run specific steps of the validation pipeline.

Usage:
  python run_validation_from_step3.py              # Run steps 3-5 (default)
  python run_validation_from_step3.py 3            # Run only step 3
  python run_validation_from_step3.py 3 4          # Run steps 3-4
  python run_validation_from_step3.py 4 5          # Run steps 4-5
  python run_validation_from_step3.py 5            # Run only step 5

Steps:
  3. Detect Modules (using NetworkAnalysis)
  4. Evaluate Modules (using Evaluation)
  5. Compare with TCGA
"""

import os
import sys
import json
import networkx as nx
import argparse

# Add project to path
sys.path.append('.')

from genepioneer.network_analysis import NetworkAnalysis
from genepioneer.evaluation import Evaluation


def load_features_from_network(network_file):
    """
    Load network and extract features from nodes
    """
    print(f"\nLoading network from: {network_file}")
    G = nx.read_gml(network_file)
    
    print(f"✓ Loaded network with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    
    # Extract features from network nodes
    features = {}
    for node in G.nodes():
        node_data = G.nodes[node]
        features[node] = {}
        
        # Copy all numerical attributes as features
        for key, value in node_data.items():
            if isinstance(value, (int, float)):
                features[node][key] = value
    
    print(f"✓ Extracted features for {len(features)} genes")
    
    return features


def step3_detect_modules(features, prioritized_genes=None, 
                         min_size=5, max_size=15, output_dir="./validation_cohort"):
    """
    STEP 3: Detect modules using NetworkAnalysis
    """
    print("\n" + "="*70)
    print("STEP 3: DETECTING MODULES")
    print("="*70)
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nInput:")
    print(f"  Genes: {len(features)}")
    print(f"  Prioritized genes: {prioritized_genes}")
    print(f"  Module size range: {min_size}-{max_size}")
    
    # Filter out metadata keys (keep only gene features)
    gene_features = {k: v for k, v in features.items() 
                     if k not in ['graph_entropy', 'ls_for_features']}
    
    print(f"\n✓ Filtered to {len(gene_features)} gene features")
    
    # Create NetworkAnalysis object
    network_analysis = NetworkAnalysis(
        cancer_type="ValidationCohort",
        features=gene_features
    )
    
    # Detect modules (prioritizing CHD1L)
    # NetworkAnalysis.module_detection() expects the network file to be in the current directory
    # So we need to temporarily change to the output directory
    print("\nRunning module detection algorithm...")
    original_dir = os.getcwd()
    try:
        os.chdir(output_dir)
        modules = network_analysis.module_detection(
            min_comm_size=min_size,
            max_comm_size=max_size,
            prioritized_genes=prioritized_genes
        )
    finally:
        os.chdir(original_dir)
    
    print(f"\n✓ Detected {len(modules)} total modules")
    
    # Find modules containing CHD1L
    chd1l_modules = []
    for module, score, quality in modules:
        if 'CHD1L' in module:
            chd1l_modules.append((module, score, quality))
    
    print(f"✓ Modules containing CHD1L: {len(chd1l_modules)}")
    
    # Save all modules
    modules_file = os.path.join(output_dir, "all_modules.json")
    with open(modules_file, 'w') as f:
        json.dump(modules, f, indent=2)
    print(f"\n✓ All modules saved to: {modules_file}")
    
    # Save CHD1L modules
    if chd1l_modules:
        chd1l_file = os.path.join(output_dir, "CHD1L_modules.json")
        with open(chd1l_file, 'w') as f:
            json.dump(chd1l_modules, f, indent=2)
        print(f"✓ CHD1L modules saved to: {chd1l_file}")
        
        # Print CHD1L modules
        print("\n" + "-"*70)
        print("CHD1L MODULES:")
        print("-"*70)
        for i, (module, score, quality) in enumerate(chd1l_modules, 1):
            print(f"\nModule {i}:")
            print(f"  Size: {len(module)} genes")
            print(f"  Score: {score:.4f}")
            print(f"  Quality: {quality:.4f}")
            print(f"  Genes: {', '.join(sorted(module))}")
    else:
        print("\n⚠️  WARNING: No modules containing CHD1L were detected")
        print("This may indicate:")
        print("  1. CHD1L is isolated in the network")
        print("  2. Module size parameters are too restrictive")
        print("  3. CHD1L connections are weak")
    
    return modules, chd1l_modules


def step4_evaluate_modules(modules, output_dir="./validation_cohort"):
    """
    STEP 4: Evaluate modules using pathway enrichment
    """
    print("\n" + "="*70)
    print("STEP 4: EVALUATING MODULES")
    print("="*70)
    
    if not modules:
        print("\n⚠️  No modules to evaluate")
        return None
    
    # Prepare modules for evaluation
    module_dict = {"ValidationCohort": modules}
    
    # Create Evaluation object
    print("\nInitializing evaluation...")
    evaluator = Evaluation(data_path="./GenesData")
    
    if not evaluator.hallmarks_data:
        print("⚠️  Failed to load Hallmark pathways")
        return None
    
    print(f"✓ Loaded {len(evaluator.hallmarks_data)} Hallmark pathways")
    
    # Evaluate modules
    print("\nPerforming pathway enrichment analysis...")
    results = evaluator.evaluate_modules(module_dict)
    
    # Save results
    results_file = os.path.join(output_dir, "evaluation_results.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Evaluation results saved to: {results_file}")
    
    # Print summary
    if "ValidationCohort" in results:
        validated_modules = results["ValidationCohort"]
        print(f"✓ Validated modules: {len(validated_modules)}")
        
        # Find and print CHD1L module results
        chd1l_evaluated = []
        for module_result in validated_modules:
            if 'CHD1L' in module_result['module_genes']:
                chd1l_evaluated.append(module_result)
        
        if chd1l_evaluated:
            print(f"✓ CHD1L modules with enrichment: {len(chd1l_evaluated)}")
            
            print("\n" + "-"*70)
            print("CHD1L MODULE ENRICHMENT RESULTS:")
            print("-"*70)
            
            for i, module_result in enumerate(chd1l_evaluated, 1):
                print(f"\nModule {i}:")
                print(f"  Genes ({len(module_result['module_genes'])}): {', '.join(sorted(module_result['module_genes']))}")
                print(f"  Score1 (Coverage): {module_result['score1']:.4f}")
                print(f"  Score2 (Pathway): {module_result['score2']:.4f}")
                print(f"  Significant pathways: {len(module_result['significant_pathways'])}")
                
                if module_result['significant_pathways']:
                    print("\n  Top pathways:")
                    for pathway in module_result['significant_pathways'][:5]:
                        print(f"    - {pathway['name']}")
                        print(f"      p-value: {pathway['p_value']:.2e}")
                        print(f"      Overlap: {pathway['overlap']}/{pathway['pathway_size']}")
        else:
            print("\n⚠️  No CHD1L modules passed enrichment criteria")
    
    return results


def step5_compare_with_tcga(validation_results, tcga_file="./evaluated_modules_result.json", 
                            output_dir="./validation_cohort"):
    """
    STEP 5: Compare validation results with TCGA cohort
    """
    print("\n" + "="*70)
    print("STEP 5: COMPARING WITH TCGA COHORT")
    print("="*70)
    
    if not validation_results or "ValidationCohort" not in validation_results:
        print("\n⚠️  No validation results to compare")
        return
    
    if not os.path.exists(tcga_file):
        print(f"\n⚠️  TCGA results file not found: {tcga_file}")
        return
    
    # Load TCGA results
    print(f"\nLoading TCGA results from: {tcga_file}")
    with open(tcga_file, 'r') as f:
        tcga_results = json.load(f)
    
    # Extract CHD1L modules from both cohorts
    validation_chd1l = []
    for module in validation_results["ValidationCohort"]:
        if 'CHD1L' in module['module_genes']:
            validation_chd1l.append(module)
    
    tcga_chd1l = []
    for cancer_type, modules in tcga_results.items():
        for module in modules:
            if 'CHD1L' in module['module_genes']:
                tcga_chd1l.append(module)
    
    print(f"\n✓ TCGA CHD1L modules: {len(tcga_chd1l)}")
    print(f"✓ Validation CHD1L modules: {len(validation_chd1l)}")
    
    if not validation_chd1l:
        print("\n⚠️  No CHD1L modules in validation cohort to compare")
        return
    
    if not tcga_chd1l:
        print("\n⚠️  No CHD1L modules in TCGA cohort to compare")
        return
    
    # Compare pathways
    print("\n" + "-"*70)
    print("PATHWAY COMPARISON:")
    print("-"*70)
    
    # Get all pathways
    tcga_pathways = set()
    for module in tcga_chd1l:
        for pathway in module.get('significant_pathways', []):
            tcga_pathways.add(pathway['name'])
    
    validation_pathways = set()
    for module in validation_chd1l:
        for pathway in module.get('significant_pathways', []):
            validation_pathways.add(pathway['name'])
    
    common_pathways = tcga_pathways.intersection(validation_pathways)
    
    print(f"\nTCGA pathways: {len(tcga_pathways)}")
    print(f"Validation pathways: {len(validation_pathways)}")
    print(f"Common pathways: {len(common_pathways)}")
    
    jaccard = 0
    if tcga_pathways or validation_pathways:
        jaccard = len(common_pathways) / len(tcga_pathways.union(validation_pathways))
        print(f"Jaccard similarity: {jaccard:.3f}")
        
        if jaccard > 0.5:
            print("✓ STRONG pathway overlap (>0.5)")
        elif jaccard > 0.3:
            print("~ MODERATE pathway overlap (0.3-0.5)")
        else:
            print("✗ WEAK pathway overlap (<0.3)")
    
    if common_pathways:
        print("\nCommon pathways:")
        for pathway in sorted(common_pathways)[:10]:
            print(f"  ✓ {pathway}")
    
    # Compare genes
    print("\n" + "-"*70)
    print("GENE OVERLAP:")
    print("-"*70)
    
    for v_idx, v_module in enumerate(validation_chd1l, 1):
        v_genes = set(v_module['module_genes'])
        
        print(f"\nValidation Module {v_idx} ({len(v_genes)} genes):")
        
        best_overlap = 0
        best_tcga = None
        
        for t_idx, t_module in enumerate(tcga_chd1l):
            t_genes = set(t_module['module_genes'])
            overlap = len(v_genes.intersection(t_genes))
            
            if overlap > best_overlap:
                best_overlap = overlap
                best_tcga = (t_idx, t_module, t_genes)
        
        if best_tcga:
            t_idx, t_module, t_genes = best_tcga
            jaccard_genes = best_overlap / len(v_genes.union(t_genes))
            
            print(f"  Best TCGA match: Module {t_idx + 1}")
            print(f"  Overlap: {best_overlap} genes")
            print(f"  Jaccard: {jaccard_genes:.3f}")
            print(f"  Common genes: {', '.join(sorted(v_genes.intersection(t_genes)))}")
    
    # Save comparison report
    comparison = {
        "tcga_modules": len(tcga_chd1l),
        "validation_modules": len(validation_chd1l),
        "tcga_pathways": list(tcga_pathways),
        "validation_pathways": list(validation_pathways),
        "common_pathways": list(common_pathways),
        "pathway_jaccard": jaccard
    }
    
    comparison_file = os.path.join(output_dir, "tcga_comparison.json")
    with open(comparison_file, 'w') as f:
        json.dump(comparison, f, indent=2)
    print(f"\n✓ Comparison report saved to: {comparison_file}")


def main():
    """
    Main validation pipeline - flexible step execution
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Run validation pipeline from specific steps',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s              # Run steps 3-5 (default)
  %(prog)s 3            # Run only step 3
  %(prog)s 3 4          # Run steps 3-4
  %(prog)s 4 5          # Run steps 4-5
  %(prog)s 5            # Run only step 5
        """
    )
    parser.add_argument('steps', nargs='*', type=int, default=[3, 4, 5],
                        help='Steps to run (3=modules, 4=evaluate, 5=compare). Default: 3 4 5')
    parser.add_argument('--output-dir', default='./validation_cohort',
                        help='Output directory (default: ./validation_cohort)')
    parser.add_argument('--prioritize', nargs='+', default=['CHD1L'],
                        help='Genes to prioritize (default: CHD1L)')
    parser.add_argument('--min-size', type=int, default=5,
                        help='Minimum module size (default: 5)')
    parser.add_argument('--max-size', type=int, default=15,
                        help='Maximum module size (default: 15)')
    
    args = parser.parse_args()
    
    # Validate steps
    valid_steps = [3, 4, 5]
    steps_to_run = sorted(set(args.steps))
    
    for step in steps_to_run:
        if step not in valid_steps:
            print(f"❌ ERROR: Invalid step {step}. Valid steps are: 3, 4, 5")
            return
    
    if not steps_to_run:
        print("❌ ERROR: No steps specified")
        return
    
    # Determine step range
    min_step = min(steps_to_run)
    max_step = max(steps_to_run)
    steps_to_run = list(range(min_step, max_step + 1))
    
    print("="*70)
    print("VALIDATION COHORT ANALYSIS PIPELINE")
    print("CHD1L Biomarker Validation - Flexible Execution")
    print("="*70)
    print(f"\nSteps to run: {steps_to_run}")
    print(f"Output directory: {args.output_dir}")
    print(f"Prioritized genes: {args.prioritize}")
    
    # Configuration
    output_dir = args.output_dir
    network_file = os.path.join(output_dir, "ValidationCohort_network_features.gml")
    prioritized_genes = args.prioritize
    
    # Variables to store results between steps
    features = None
    all_modules = None
    chd1l_modules = None
    validation_results = None
    
    # STEP 3: Module Detection
    if 3 in steps_to_run:
        # Check if network file exists
        if not os.path.exists(network_file):
            print(f"\n❌ ERROR: Network file not found: {network_file}")
            print("\nPlease run steps 1-2 first to generate the network.")
            print("You can run the full pipeline with: python run_validation_pipeline.py")
            return
        
        # Load features from saved network
        print("\n" + "="*70)
        print("LOADING SAVED NETWORK FEATURES")
        print("="*70)
        features = load_features_from_network(network_file)
        
        # Check if prioritized genes are in the features
        for gene in prioritized_genes:
            if gene not in features:
                print(f"\n⚠️  WARNING: {gene} not found in network features")
            else:
                print(f"✓ {gene} found in network with {len(features[gene])} features")
        
        # Run module detection
        all_modules, chd1l_modules = step3_detect_modules(
            features, 
            prioritized_genes=prioritized_genes,
            min_size=args.min_size,
            max_size=args.max_size,
            output_dir=output_dir
        )
        
        if not all_modules:
            print("\n⚠️  WARNING: No modules detected")
            if 4 in steps_to_run or 5 in steps_to_run:
                print("Cannot proceed to subsequent steps without modules")
                return
    
    # STEP 4: Module Evaluation
    if 4 in steps_to_run:
        # Load CHD1L modules if not from step 3
        if chd1l_modules is None:
            chd1l_file = os.path.join(output_dir, "CHD1L_modules.json")
            if not os.path.exists(chd1l_file):
                print(f"\n❌ ERROR: CHD1L modules file not found: {chd1l_file}")
                print("Please run step 3 first to detect modules")
                return
            
            print("\n" + "="*70)
            print("LOADING SAVED CHD1L MODULES")
            print("="*70)
            print(f"Loading from: {chd1l_file}")
            with open(chd1l_file, 'r') as f:
                chd1l_modules = json.load(f)
            print(f"✓ Loaded {len(chd1l_modules)} CHD1L modules")
        
        if not chd1l_modules:
            print("\n⚠️  No CHD1L modules to evaluate")
            if 5 in steps_to_run:
                print("Cannot proceed to step 5 without CHD1L modules")
                return
        
        # Evaluate ONLY CHD1L modules (faster and more focused)
        print(f"\n→ Evaluating {len(chd1l_modules)} CHD1L modules only")
        validation_results = step4_evaluate_modules(chd1l_modules, output_dir)
        
        if not validation_results:
            print("\n⚠️  WARNING: Evaluation failed or no results")
            if 5 in steps_to_run:
                print("Cannot proceed to step 5 without evaluation results")
                return
    
    # STEP 5: Compare with TCGA
    if 5 in steps_to_run:
        # Load evaluation results if not from step 4
        if validation_results is None:
            results_file = os.path.join(output_dir, "evaluation_results.json")
            if not os.path.exists(results_file):
                print(f"\n❌ ERROR: Evaluation results not found: {results_file}")
                print("Please run step 4 first to evaluate modules")
                return
            
            print("\n" + "="*70)
            print("LOADING SAVED EVALUATION RESULTS")
            print("="*70)
            print(f"Loading from: {results_file}")
            with open(results_file, 'r') as f:
                validation_results = json.load(f)
            print(f"✓ Loaded evaluation results")
        
        # Run comparison
        step5_compare_with_tcga(validation_results, output_dir=output_dir)
    
    # Final summary
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print(f"\nResults saved in: {output_dir}/")
    print("\nAvailable files:")
    
    files_info = {
        "all_modules.json": "All detected modules",
        "CHD1L_modules.json": "CHD1L-containing modules",
        "evaluation_results.json": "Pathway enrichment results",
        "tcga_comparison.json": "TCGA comparison report"
    }
    
    for filename, description in files_info.items():
        filepath = os.path.join(output_dir, filename)
        if os.path.exists(filepath):
            print(f"  ✓ {filename}: {description}")
        else:
            print(f"  - {filename}: {description} (not generated)")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    main()
