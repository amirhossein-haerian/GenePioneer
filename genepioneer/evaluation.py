import pandas as pd
import glob
import os

from genepioneer import DataLoader

class Evaluation:

    def __init__(self):
                
        self.benchmark_genes = self.read_benchmark_genes("../Data/benchmark-data")
        self.network_genes = self.read_mutated_genes("../Data/cancer-gene-data")
        self.result = self.eval2(self.network_genes, self.benchmark_genes)
                
    def read_benchmark_genes(self, benchmark_folder):
        benchmark_genes = {}
        for filepath in glob.glob(os.path.join(benchmark_folder, '*.txt')):
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
            data_loader = DataLoader(cancer_type)
            genes_with_cases, cases_with_genes, total_cases = data_loader.load_TCGA()
            genes = list(genes_with_cases.keys())
            network_genes[cancer_type] = genes
        return network_genes

    def get_top_n_genes(self, genes, n):
        return set(genes[:n])

    def calculate_metrics(self, predicted_genes, benchmark_genes):
        TP = len(predicted_genes.intersection(benchmark_genes))
        FP = len(predicted_genes.difference(benchmark_genes))
        FN = len(benchmark_genes.difference(predicted_genes))
        
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f_measure = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return TP, FP, FN, precision, recall, f_measure
    
    def calculate_metrics2(self, predicted_genes, benchmark_genes):
        TP = len(benchmark_genes.intersection(predicted_genes))
        FP = len(predicted_genes)
        FN = len(benchmark_genes)
        
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f_measure = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return TP, FP, FN, precision, recall, f_measure

    def eval(self, network_genes, benchmark_genes):
        results = {}
        for cancer_type, genes in network_genes.items():
            results[cancer_type] = {}
            for benchmark_name, benchmark in benchmark_genes.items():
                intersected_genes = set(genes).intersection(benchmark)
                top_n_genes = self.get_top_n_genes(genes, len(intersected_genes))
                metrics = self.calculate_metrics(top_n_genes, intersected_genes)
                results[cancer_type][benchmark_name] = metrics
        return results
    
    def eval2(self, network_genes, benchmark_genes):
        results = {}
        for cancer_type, genes in network_genes.items():
            results[cancer_type] = {}
            for benchmark_name, benchmark in benchmark_genes.items():
                intersected_genes = set(benchmark)
                metrics = self.calculate_metrics2(set(genes), intersected_genes)
                results[cancer_type][benchmark_name] = metrics
        return results
    
    def print_result(self):
        for cancer_type, benchmarks in self.result.items():
            print(f'Cancer Type: {cancer_type}')
            for benchmark_name, metrics in benchmarks.items():
                TP, FP, FN, precision, recall, f_measure = metrics
                print(f'  Benchmark: {benchmark_name}')
                print(f'    TP: {TP}, FP: {FP}, FN: {FN}')
                print(f'    Precision: {precision:.3f}, Recall: {recall:.3f}, F-measure: {f_measure:.3f}')