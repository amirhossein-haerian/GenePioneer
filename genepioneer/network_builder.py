import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse
import scipy.sparse.csgraph
import csv

from collections import defaultdict
from itertools import combinations

from concurrent.futures import ThreadPoolExecutor, as_completed

import networkx as nx
from .data_loader import DataLoader
from .network_analysis import NetworkAnalysis

class NetworkBuilder:
    def __init__(self, cancer_type):
        self.cancer_type = cancer_type
        self.graph = nx.Graph()
        data_loader = DataLoader(self.cancer_type)

        self.genes_with_cases ,self.cases_with_genes, self.total_cases = data_loader.load_TCGA()
        self.genes_with_processes, self.processes_with_genes, self.total_processes = data_loader.load_IBM()
        
        self.extended_genes = set()
        
        self.all_features = defaultdict(dict)
        
        self.calculated = 0
        self.calculated1 = 1

    def build_network(self):
        self.edge_adder("base")
        self.edge_adder("extend")
        self.edge_adder("extend_connections")
        
        return self.graph


    def edge_adder(self, type):
        if type == "base":
            for genes in self.cases_with_genes.values():
                for gene1, gene2 in combinations(genes, 2):
                    if not self.graph.has_edge(gene1, gene2):
                        attributes = {
                            'process_weight': self.shared_processes(gene1, gene2),
                            'case_weight': self.shared_cases(gene1, gene2),
                        }
                        self.graph.add_edge(gene1, gene2, **attributes)

        elif type == "extend":
            list_of_nodes = list(self.graph.nodes())
            for gene in list_of_nodes:
                # Find all processes that include this gene
                for process in self.genes_with_processes[gene]:
                    # Connect this gene to all others in the same process
                    for gene_to_connect in self.processes_with_genes[process]:
                        if gene != gene_to_connect:
                            self.extended_genes.add(gene_to_connect)
                            # Check if the edge exists to avoid adding a duplicate
                            if not self.graph.has_edge(gene, gene_to_connect):
                                attributes = {
                                    'process_weight': self.shared_processes(gene, gene_to_connect),
                                    'case_weight': self.shared_cases(gene, gene_to_connect),
                                }   
                                self.graph.add_edge(gene, gene_to_connect, **attributes)
                                
        elif type == "extend_connections":
            for gene in self.extended_genes:
                for gene_to_connect in self.extended_genes:
                    if gene != gene_to_connect:
                        for process in self.genes_with_processes[gene]:
                        # Connect this gene to all others in the same process
                            if gene_to_connect in self.processes_with_genes[process]:
                                if not self.graph.has_edge(gene, gene_to_connect):
                                    attributes = {
                                        'process_weight': self.shared_processes(gene, gene_to_connect),
                                        'case_weight': self.shared_cases(gene, gene_to_connect),
                                    }   
                                    self.graph.add_edge(gene, gene_to_connect, **attributes)
                                    break
            
                                
            self.calculate_edge_weight()


    def shared_cases(self, gene1, gene2):
        cases_num = len(self.genes_with_cases[gene1] & self.genes_with_cases[gene2]) if gene1 in self.genes_with_cases and gene2 in self.genes_with_cases else 0
        return cases_num

    def shared_processes(self, gene1, gene2):
        processes_num = sum(gene2 in self.processes_with_genes[process] for process in self.genes_with_processes[gene1])
        return processes_num
    
    def calculate_edge_weight(self):
        print("here")
        degrees = self.graph.degree()
        with open('node_degrees.txt', 'w') as file:
            for node, degree in degrees:
                file.write(f'{node}: {degree}\n')
        
        process_weights = []
        case_weights = []

        for u, v, data in self.graph.edges(data=True):
            process_weights.append(data['process_weight'])
            case_weights.append(data['case_weight'])

        # Convert to numpy arrays
        process_weights = np.array(process_weights)
        case_weights = np.array(case_weights)

        # Calculate min and max
        process_min, process_max = process_weights.min(), process_weights.max()
        case_min, case_max = case_weights.min(), case_weights.max()
        
        normalized_process_weights = (process_weights - process_min) / (process_max - process_min)
        normalized_case_weights = (case_weights - case_min) / (case_max - case_min)
        
        for i, (u, v, data) in enumerate(self.graph.edges(data=True)):
            data['weight'] = normalized_process_weights[i] + normalized_case_weights[i]
    

    def weight_node(self, gi):
        weights = [self.graph[gi][gj]['weight'] for gj in self.graph.neighbors(gi)]
        return sum(weights) #/ len(weights)

    def graph_entropy(self, weights):
        probabilities = weights / weights.sum()
        return -np.sum(probabilities * np.log(probabilities))

    def node_effect_on_entropy(self, node, entropy):   
        adjusted_weights = np.array(list(self.node_weights_cache.values()))
        node_index = list(self.graph.nodes()).index(node)
        
        adjusted_weights[node_index] = 0
        
        for neighbor in self.graph.neighbors(node):
            neighbor_index = list(self.graph.nodes()).index(neighbor)
            edge_weight = self.graph[node][neighbor].get('weight')  # Default weight of 1
            adjusted_weights[neighbor_index] -= edge_weight
            adjusted_weights[neighbor_index] = max(adjusted_weights[neighbor_index], 0) 
            
        adjusted_weights = adjusted_weights[adjusted_weights > 0]
        
        new_entropy = self.graph_entropy(adjusted_weights)

        return abs(entropy - new_entropy)
    
    def calculate_all_features(self):
        # Precompute all centralities and entropy
        # closeness_centrality = nx.closeness_centrality(self.graph, distance='weight')
        A = nx.adjacency_matrix(self.graph).tolil()
        D = scipy.sparse.csgraph.floyd_warshall(A, directed=False, unweighted=False)
        
        n = D.shape[0]
        closeness_centrality = {}
        
        for r, node in enumerate(self.graph.nodes()):
            cc = 0.0
            
            possible_paths = list(enumerate(D[r, :]))
            shortest_paths = dict(filter( \
                lambda x: not x[1] == np.inf, possible_paths))
            
            total = sum(shortest_paths.values())
            n_shortest_paths = len(shortest_paths) - 1.0
            if total > 0.0 and n > 1:
                s = n_shortest_paths / (n - 1)
                cc = (n_shortest_paths / total) * s
            closeness_centrality[node] = cc
        
        # max_closeness = max(closeness_centrality.values())
        # min_closeness = min(closeness_centrality.values())
        normalized_closeness = closeness_centrality # {node: (value - min_closeness) / (max_closeness - min_closeness) for node, value in closeness_centrality.items()}
        print("closensess")
        # betweenness_centrality = nx.betweenness_centrality(self.graph, normalized=True, weight='weight')
        betweenness_centrality = nx.betweenness_centrality(self.graph, normalized=True, weight='weight') # k=int(len(self.graph.nodes())*.1)
        print("betweenness")
        eigenvector_centrality = nx.eigenvector_centrality(self.graph, weight='weight')
        print("eigenvector")
        
        # Get weights for all nodes
        self.node_weights_cache = {node: self.weight_node(node) for node in self.graph.nodes()}
        weights_array = np.array(list(self.node_weights_cache.values()))
        entropy = self.graph_entropy(weights_array)
        print("entropy")

        # Use a ThreadPoolExecutor to parallelize the node effect on entropy calculation

        for node in self.graph.nodes():
            effect_on_entropy = self.node_effect_on_entropy(node, entropy)
            self.all_features[node] = {
                'weight': self.node_weights_cache[node],
                'closeness_centrality': normalized_closeness[node],
                'betweenness_centrality': betweenness_centrality[node],
                'eigenvector_centrality': eigenvector_centrality[node],
                'effect_on_entropy': effect_on_entropy,
            }
            self.graph.nodes[node].update({
                'weight': self.node_weights_cache[node],
                'closeness_centrality': normalized_closeness[node],
                'betweenness_centrality': betweenness_centrality[node],
                'eigenvector_centrality': eigenvector_centrality[node],
                'effect_on_entropy': effect_on_entropy,
                'graph_entropy': entropy
            })
        
        print("goes to ls")
        self.add_LS_to_network()
        
        self.all_features['graph_entropy'] = entropy
        
        return self.all_features
    
    def save_features_to_csv(self, all_features, filename):
        nx.write_gml(self.graph, f"{filename}.gml")
        with open(f"{filename}.csv", 'w', newline='') as csvfile:
            fieldnames = ['node', 'weight', 'closeness_centrality', 'betweenness_centrality',
                        'eigenvector_centrality', 'effect_on_entropy', "ls_score"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for node, features in all_features.items():
                if node != 'graph_entropy' and node != "ls_for_features":
                    features['node'] = node  # Add node ID to the row
                    writer.writerow(features)
                    
            writer.writerow({'node': 'graph_entropy', 'weight': all_features['graph_entropy']})
            writer.writerow({'node': 'ls_for_features', 'weight': all_features['ls_for_features']})
                
            
    def add_LS_to_network(self):
        network_analysis = NetworkAnalysis(self.cancer_type, self.all_features);

        nodes, features = network_analysis.compute_laplacian_scores()
        for node in self.graph.nodes():
                        
            self.all_features[node]["ls_score"] = nodes.loc[node].LaplacianScore
            
            self.graph.nodes[node].update({
                'ls_score': nodes.loc[node].LaplacianScore,
                'ls_for_features': np.array_str(features),
            })
            
        self.all_features["ls_for_features"] = np.array_str(features)
        return self.all_features
    