import matplotlib.pyplot as plt
import numpy as np
import csv

from collections import defaultdict
from itertools import combinations
from functools import lru_cache

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
        
        self.all_features = defaultdict(dict)

    def build_network(self):
        self.edge_adder("base")
        self.edge_adder("extend")


    def edge_adder(self, type):
        if type == "base":
            for genes in self.cases_with_genes.values():
                for gene1, gene2 in combinations(genes, 2):
                    if not self.graph.has_edge(gene1, gene2):
                        self.graph.add_edge(gene1, gene2, weight=self.calculate_weight(gene1, gene2))

        elif type == "extend":
            list_of_nodes = list(self.graph.nodes())
            for gene in list_of_nodes:
                # Find all processes that include this gene
                for process in self.genes_with_processes[gene]:
                    # Connect this gene to all others in the same process
                    for gene_to_connect in self.processes_with_genes[process]:
                        if gene != gene_to_connect:
                            # Check if the edge exists to avoid adding a duplicate
                            if not self.graph.has_edge(gene, gene_to_connect):
                                self.graph.add_edge(gene, gene_to_connect, weight=self.calculate_weight(gene, gene_to_connect))


    @lru_cache(maxsize=None)
    def shared_cases(self, gene1, gene2):
        return len(self.genes_with_cases[gene1] & self.genes_with_cases[gene2]) if gene1 in self.genes_with_cases and gene2 in self.genes_with_cases else 0

    @lru_cache(maxsize=None)
    def shared_processes(self, gene1, gene2):
        return sum(gene2 in self.processes_with_genes[process] for process in self.genes_with_processes[gene1])
    
    def calculate_weight(self, gene1, gene2):

        weight_cases = self.shared_cases(gene1, gene2) / self.total_cases
        weight_processes = self.shared_processes(gene1, gene2) / self.total_processes

        return weight_cases + weight_processes
    

    def weight_node(self, gi):
        return sum([self.graph[gi][gj]['weight'] for gj in self.graph.neighbors(gi)])

    def graph_entropy(self, weights):
        probabilities = weights / weights.sum()
        return -np.sum(probabilities * np.log(probabilities))

    def node_effect_on_entropy(self, node, entropy):
        graph_clone = self.graph.copy()
        graph_clone.remove_node(node)

        node_weights = {node: self.weight_node(node) for node in graph_clone.nodes()}
        weights_array = np.array(list(node_weights.values()))
        new_entropy = self.graph_entropy(weights_array)
        return abs(entropy - new_entropy)
    
    def calculate_all_features(self):
        # Precompute all centralities and entropy
        closeness_centrality = nx.closeness_centrality(self.graph, distance='weight')
        max_closeness = max(closeness_centrality.values())
        min_closeness = min(closeness_centrality.values())
        normalized_closeness = {node: (value - min_closeness) / (max_closeness - min_closeness) for node, value in closeness_centrality.items()}
        betweenness_centrality = nx.betweenness_centrality(self.graph, normalized=True, weight='weight')
        eigenvector_centrality = nx.eigenvector_centrality(self.graph, weight='weight')

        # Get weights for all nodes
        node_weights = {node: self.weight_node(node) for node in self.graph.nodes()}
        weights_array = np.array(list(node_weights.values()))
        entropy = self.graph_entropy(weights_array)

        # Use a ThreadPoolExecutor to parallelize the node effect on entropy calculation
        with ThreadPoolExecutor() as executor:
            futures = {}
            for node in self.graph.nodes():
                futures[executor.submit(self.node_effect_on_entropy, node, entropy)] = node

            for future in as_completed(futures):
                node = futures[future]
                effect_on_entropy = future.result()
                self.all_features[node] = {
                    'weight': node_weights[node],
                    'closeness_centrality': normalized_closeness[node],
                    'betweenness_centrality': betweenness_centrality[node],
                    'eigenvector_centrality': eigenvector_centrality[node],
                    'effect_on_entropy': effect_on_entropy,
                }
                self.graph.nodes[node].update({
                    'weight': node_weights[node],
                    'closeness_centrality': normalized_closeness[node],
                    'betweenness_centrality': betweenness_centrality[node],
                    'eigenvector_centrality': eigenvector_centrality[node],
                    'effect_on_entropy': effect_on_entropy,
                    'graph_entropy': entropy
                })
            
            self.add_LS_to_network()
            
            self.all_features['graph_entropy'] = entropy
        
        return self.all_features
    
    def save_features_to_csv(self, all_features, filename):
        nx.write_gml(self.graph, "test.gml")
        with open(filename, 'w', newline='') as csvfile:
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
        
    def print_network_summary(self):
        print(f"Number of nodes: {self.graph.number_of_nodes()}")
        print(f"Number of edges: {self.graph.number_of_edges()}")
        # Print the density of the graph
        print(f"Density of the graph: {nx.density(self.graph)}")
        # Check if the graph is connected
        print(f"Is the graph connected? {nx.is_connected(self.graph)}")
        # Print the number of connected components
        print(f"Number of connected components: {nx.number_connected_components(self.graph)}")

    def analyze_nodes_edges(self):
        # Degree of each node
        degrees = dict(self.graph.degree())
        sorted_degrees = sorted(degrees.items(), key=lambda x: x[1], reverse=True)
        print("Top 5 nodes by degree:", sorted_degrees[:5])

        # Weights of each edge
        edges_weights = nx.get_edge_attributes(self.graph, 'weight')
        sorted_weights = sorted(edges_weights.items(), key=lambda x: x[1], reverse=True)
        print("Top 5 edges by weight:", sorted_weights[:5])

        connected_components = list(nx.connected_components(self.graph))

        print(f"Total connected components: {len(connected_components)}")

        for i, component in enumerate(connected_components):
            subgraph = self.graph.subgraph(component)
            print(f"Subgraph {i+1} has {subgraph.number_of_nodes()} nodes.")
    