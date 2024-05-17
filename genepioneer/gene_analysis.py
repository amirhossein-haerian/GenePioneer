import numpy as np
import pandas as pd
import os
import json

import networkx as nx

class GeneAnalysis:

    def __init__(self, cancer_type, input_path):
        self.cancer_type = cancer_type
        
        self.network = self.load_network()
        self.modules = self.load_modules()
        self.rankings = self.rank_nodes_by_ls_score()
        self.genes = self.load_genes(input_path)
                
    def load_network(self):
        G = nx.read_gml(f"../Data/{self.cancer_type}_network_features.gml")
        return G
    
    def load_modules(self):
        # Load the JSON data from the given file path
        with open(f"{self.cancer_type}.json", 'r') as file:
            data = json.load(file)
        
        # Create a dictionary to store module data by gene
        modules = {}
        for module_index, (gene_list, score) in enumerate(data):
            for gene in gene_list:
                if gene not in modules:
                    modules[gene] = []
                # Store the entire module (gene list and value) for each gene
                modules[gene].append({'genes': gene_list, 'score': score})
        return modules
    
    def load_genes(self, input_path):
        # Read genes from the specified input file
        with open(input_path, 'r') as file:
            genes = [line.strip() for line in file if line.strip()]
        return genes
    
    def rank_nodes_by_ls_score(self):
        return {node: rank for rank, (node, data) in enumerate(sorted(self.network.nodes(data=True), key=lambda x: x[1].get('ls_score', 0), reverse=True))}
    
    def find_gene_modules(self, gene):
        return self.modules.get(gene, [])
    
    def analyze_genes(self):
        results = {}
        for gene in self.genes:
            gene_info = {}
            gene_info['rank'] = self.rankings.get(gene, 'Not in network')
            gene_info['modules'] = self.find_gene_modules(gene)
            results[gene] = gene_info
        
        with open("../Data/output.json", 'w') as f:
            json.dump(results, f, indent=4)