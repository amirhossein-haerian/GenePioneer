import pandas as pd
import glob
import os

import concurrent.futures


import numpy as np

import json
from gprofiler import GProfiler

from sklearn.metrics import precision_score, recall_score, roc_curve, auc
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu, hypergeom


from genepioneer import DataLoader

class Evaluation:
    def __init__(self, data_path, cancer_gene_path=None, module_data_path=None, benchmark_data_path=None, auto_load_modules=False):
        self.gp = GProfiler(return_dataframe=True)
        
        print(os.getcwd())
        
        self.cancer_gene_path = cancer_gene_path or "genepioneer/Data/benchmark-data"
        self.module_data_path = module_data_path or "genepioneer/Data/module-data"
        self.benchmark_data_path = benchmark_data_path or "genepioneer/Data/benchmark-data"

        # Load hallmarks data for manual enrichment analysis
        self.hallmarks_data = self.load_hallmarks_data()
        
        # Only load other data if paths exist and we're doing full evaluation
        if os.path.exists(self.benchmark_data_path):
            self.benchmark_genes = self.read_benchmark_genes(self.benchmark_data_path)
        else:
            self.benchmark_genes = {}
            
        if os.path.exists(self.cancer_gene_path):
            self.network_genes = self.read_network_genes(self.cancer_gene_path)
        else:
            self.network_genes = {}
        
        # Only auto-load modules if explicitly requested (for backward compatibility)
        if auto_load_modules and os.path.exists(self.module_data_path):
            self.modules = self.read_modules(self.module_data_path)
        else:
            self.modules = {}
        
        # Only run evaluations if we have the necessary data
        if self.network_genes and self.benchmark_genes:
            self.result = self.eval(self.network_genes, self.benchmark_genes)
        else:
            self.result = {}
            
        # Only auto-evaluate if explicitly requested (for backward compatibility)
        if auto_load_modules and self.modules:
            self.module_results = self.evaluate_modules(self.modules)
        else:
            self.module_results = {}
            
        self.data_path = data_path
        print(self.data_path)
    
    def load_hallmarks_data(self):
        """Load hallmarks gene sets from Excel file"""
        try:
            hallmarks_path = os.path.join("genepioneer", "Data", "Hallmarks_genes_wide.xlsx")
            df = pd.read_excel(hallmarks_path)
            
            # Convert to dictionary where keys are pathway names and values are gene sets
            hallmarks_dict = {}
            for pathway in df.columns:
                # Remove NaN values and convert to set
                genes = set(df[pathway].dropna().astype(str))
                hallmarks_dict[pathway] = genes
                
            print(f"Loaded {len(hallmarks_dict)} hallmark pathways")
            return hallmarks_dict
        except Exception as e:
            print(f"Error loading hallmarks data: {e}")
            return {}
    
    def get_network_size(self):
        """Get the actual number of genes in the Prostate_filtered network"""
        try:
            import networkx as nx
            # Try different possible paths for the network file
            possible_paths = [
                "tests/Prostate_filtered_network_features.gml",
                "Prostate_filtered_network_features.gml",
                os.path.join("tests", "Prostate_filtered_network_features.gml")
            ]
            
            for network_path in possible_paths:
                if os.path.exists(network_path):
                    G = nx.read_gml(network_path)
                    network_size = G.number_of_nodes()
                    print(f"Loaded network from {network_path} with {network_size} genes")
                    return network_size
            
            # If no network file found, fallback to default
            print("Warning: Could not find Prostate_filtered network file, using default background size")
            return 20000
            
        except Exception as e:
            print(f"Error loading network: {e}, using default background size")
            return 20000
    
    def read_modules(self, benchmark_folder):
        benchmark_genes = {}
        for filepath in glob.glob(os.path.join(benchmark_folder, '*.json')):
            module_name = os.path.basename(filepath).split('.')[0]
            with open(filepath, 'r') as file:
                modules = json.load(file)
            benchmark_genes[module_name] = modules
        return benchmark_genes
                
    def read_benchmark_genes(self, benchmark_folder):
        benchmark_genes = {}
        for filepath in glob.glob(os.path.join(benchmark_folder, '*.txt')):
            print(filepath)
            benchmark_name = os.path.basename(filepath).split('.')[0]
            with open(filepath, 'r') as file:
                genes = set(file.read().strip().split('\n'))
            benchmark_genes[benchmark_name] = genes
        return benchmark_genes

    def read_network_genes(self, network_folder):
        network_genes = {}
        for filepath in glob.glob(os.path.join(network_folder, '*.csv')):
            cancer_type = os.path.basename(filepath).split('.')[0]
            df = pd.read_csv(filepath)
            df = df.sort_values(by='ls_score', ascending=False)
            genes = df['node'].tolist()
            network_genes[cancer_type] = genes
        return network_genes
    def read_mutated_genes(self, network_folder):
        network_genes = {}
        for filepath in glob.glob(os.path.join(network_folder, '*.csv')):
            cancer_type = os.path.basename(filepath).split('.')[0]
            cancer_type = cancer_type.replace('_network_features', '')
            data_loader = DataLoader(cancer_type, self.data_path)
            print(data_loader)
            genes = data_loader.load_TCGA()
            network_genes[cancer_type] = genes
        return network_genes

    def get_top_n_genes(self, genes, n):
        return set(genes[:n])

    def calculate_metrics(self, predicted_genes, benchmark_genes):
        TP = len(predicted_genes.intersection(benchmark_genes))
        B_size = len(benchmark_genes)
                
        return TP, B_size

    def eval(self, network_genes, benchmark_genes):
        results = {}
        for cancer_type, genes in network_genes.items():
            results[cancer_type] = {}
            total_genes = set(genes)
            for benchmark_name, benchmark in benchmark_genes.items():
                top_n_genes = self.get_top_n_genes(genes, len(benchmark))
                metrics = self.calculate_metrics(top_n_genes, benchmark)
                auc_roc, precision, recall = self.calculate_auc_roc(top_n_genes, benchmark, total_genes)
                driver_ranks, other_ranks = self.evaluate_driver_genes(genes, benchmark)
                results[cancer_type][benchmark_name] = {
                    'metrics': metrics,
                    'auc_roc': auc_roc,
                    'precision': precision,
                    'recall': recall,
                    'driver_ranks': driver_ranks,
                    'other_ranks': other_ranks
                }
        return results
    
    def evaluate_driver_genes(self, genes, benchmark):
        driver_ranks = [genes.index(gene) for gene in benchmark if gene in genes]
        other_ranks = [rank for rank, gene in enumerate(genes) if gene not in benchmark]
        return driver_ranks, other_ranks

    def calculate_auc_roc(self, predicted_genes, benchmark_genes, total_genes):
        y_true = [1 if gene in benchmark_genes else 0 for gene in total_genes]
        y_scores = [1 if gene in predicted_genes else 0 for gene in total_genes]
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        precision = precision_score(y_true, y_scores)
        recall = recall_score(y_true, y_scores)
        auc_roc = auc(fpr, tpr)
        return auc_roc, precision, recall
    
    def print_driver_ranking_stats(self, cancer_type, benchmark_name, driver_ranks, other_ranks):
        if driver_ranks:
            mean_rank = np.mean(driver_ranks)
            median_rank = np.median(driver_ranks)
            rank_percentiles = np.percentile(driver_ranks, [25, 50, 75])
            mannwhitney_p = mannwhitneyu(driver_ranks, other_ranks, alternative='less').pvalue
            print(f'Cancer Type: {cancer_type} - Benchmark: {benchmark_name}')
            print(f'  Median Rank: {median_rank}')
            print(f'  Rank Percentiles (25th, 50th, 75th): {rank_percentiles}')
            print(f'  Mann-Whitney U Test p-value: {mannwhitney_p}')
        else:
            print(f'Cancer Type: {cancer_type} - Benchmark: {benchmark_name}')
            print('  No driver genes found in the ranked list.')
    
    def print_result(self):
        for cancer_type, benchmarks in self.result.items():
            print(f'Cancer Type: {cancer_type}')
            for benchmark_name, result in benchmarks.items():
                metrics = result['metrics']
                TP, B_size = metrics
                auc_roc = result['auc_roc']
                precision = result['precision']
                recall = result['recall']
                driver_ranks = result['driver_ranks']
                other_ranks = result['other_ranks']
                print(f'Benchmark: {benchmark_name}')
                print(f'TP: {TP}, size of benchmark: {B_size}')
                print(f'AUC-ROC: {auc_roc:.3f}')
                print(f'precision: {precision:.3f}')
                print(f'recall: {recall:.3f}')
    def manual_pathway_enrichment(self, query_genes, background_size=None):
        """
        Perform manual pathway enrichment analysis using hypergeometric test
        Uses actual network size as background if not specified
        """
        query_genes = set(str(gene).upper() for gene in query_genes)  # Convert to uppercase strings
        
        # Use actual network size if background_size not provided
        if background_size is None:
            background_size = self.get_network_size()
        
        results = []
        
        for pathway_name, pathway_genes in self.hallmarks_data.items():
            # Convert pathway genes to uppercase strings for comparison
            pathway_genes_upper = set(str(gene).upper() for gene in pathway_genes)
            
            # Calculate overlap
            overlap = query_genes.intersection(pathway_genes_upper)
            overlap_size = len(overlap)
            
            # Skip if no overlap
            if overlap_size == 0:
                continue
                
            # Hypergeometric test parameters
            M = background_size  # Total number of genes in background (actual network size)
            n = len(pathway_genes_upper)  # Number of genes in this pathway
            N = len(query_genes)  # Number of genes in query
            k = overlap_size  # Number of overlapping genes
            
            # Calculate p-value using hypergeometric distribution
            # P(X >= k) where X ~ Hypergeometric(M, n, N)
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
        
        # Sort by p-value
        results.sort(key=lambda x: x['p_value'])
        return results
                
    def evaluate_modules(self, modules):
        
        # We'll accept any significant hallmark pathway, not just specific ones
        results = {}        
        def evaluate_single_module(module):
            genes, score1, score2 = module
            
            # Use manual enrichment analysis instead of gprofiler
            enrichment_results = self.manual_pathway_enrichment(genes)
            
            # Filter for significant pathways (p-value <= 0.05)
            # Remove the restriction to pathways_of_interest to be more inclusive
            significant_pathways = [
                result for result in enrichment_results
                if result['p_value'] <= 0.05
            ]
            
            print(f"Module {genes[:3]}...: Found {len(enrichment_results)} total enrichments, {len(significant_pathways)} significant")
            
            # Be permissive: accept modules with 1+ significant pathways
            if len(significant_pathways) >= 1:
                # Convert to format similar to gprofiler output
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
                
                print(f"  -> Module ACCEPTED with {len(significant_pathways)} significant pathways")
                return {
                    'module_genes': genes,
                    'score1': score1,
                    'score2': score2,
                    'significant_pathways': pathway_info
                }
            else:
                print(f"  -> Module REJECTED (no significant pathways)")
                return None
        # Process modules sequentially instead of with threading to avoid issues
        count = 0
        for cancer_type, module_list in modules.items():
            if cancer_type not in results:
                results[cancer_type] = []
            
            print(f"Processing {len(module_list)} modules for {cancer_type}")
            
            for module in module_list:
                try:
                    evaluation = evaluate_single_module(module)
                    if evaluation:  # Only add if not None
                        results[cancer_type].append(evaluation)
                    count += 1
                    print(f"Processed {count} modules")
                except Exception as e:
                    print(f"Module evaluation failed for {cancer_type}: {e}")
                    import traceback
                    traceback.print_exc()
    
        return results
    
    def print_module_evaluation(self): 
        with open('evaluated_modules_result.json', 'w') as f:
            json.dump(self.module_results, f, indent=2)
