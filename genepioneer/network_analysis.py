import numpy as np
import pandas as pd

from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler

import igraph as ig
import leidenalg as la
from collections import defaultdict

import networkx as nx
import random

class NetworkAnalysis:

    def __init__(self, cancer_type, features={}):
        self.cancer_type = cancer_type
        self.feature_dict = features
        self.feature_matrix = None
        self.node_names = None
        self.counter = 1
        if features:
            self.load_data()
        
    def load_data(self):
        try:
            # Load the data
            df = pd.DataFrame.from_dict(self.feature_dict, orient='index')
            df.reset_index(inplace=True)
            df.rename(columns={'index': 'node'}, inplace=True)
            # Drop the last row and the first column containing node names
            self.node_names = df.iloc[:, 0]
            self.feature_matrix = df.iloc[:, 1:].values.astype(np.float64)
            scaler = StandardScaler()
            self.feature_matrix = scaler.fit_transform(self.feature_matrix)
        except Exception as e:
            raise ValueError(f"An error occurred while reading the file: {e}")
       
    def euclidean_distance_vec(self):
        try:
            # Using scipy's cdist to compute all pairwise Euclidean distances efficiently
            return cdist(self.feature_matrix, self.feature_matrix, 'euclidean')
        except Exception as e:
            raise ValueError(f"An error occurred while euclidean_distance_vec: {e}")

    def compute_similarity_matrix(self, delta=5, t=100):
        try:
            dist_matrix = self.euclidean_distance_vec()
            # Apply the exponential similarity function and the threshold
            # This operation is vectorized for efficiency
            S = np.exp(-(dist_matrix**2) / t) * (dist_matrix < delta)
            return S
        except Exception as e:
            raise ValueError(f"An error occurred while compute_similarity_matrix: {e}")
    
    def compute_laplacian_scores(self):
        try:
            S = self.compute_similarity_matrix()
            
            D = np.diag(np.sum(S, axis=1))
            L = D - S

            # Compute the Laplacian Score for each feature
            m, n = self.feature_matrix.shape
            J = np.ones((m, 1))
            F = self.feature_matrix.T
            L_scores = np.zeros(n)
            for j in range(n):
                F_j = F[j].reshape(-1, 1)
                
                weighted_mean = (np.matmul(np.matmul(np.transpose(F_j),D),J).item() / np.matmul(np.matmul(np.transpose(J),D),J).item()) * J
                # Subtract the weighted mean from each element of F_j to get F_j_tilde
                F_j_tilde = F_j - weighted_mean
                numerator = np.matmul(np.matmul(np.transpose(F_j_tilde),L),F_j_tilde).item()
                denominator = np.matmul(np.matmul(np.transpose(F_j_tilde),D),F_j_tilde).item()
                L_scores[j] = numerator / denominator if denominator != 0 else 0
            # Compute the LS for each gene
            LS = self.feature_matrix @ L_scores
            results_df = pd.DataFrame({
                'LaplacianScore': LS
            }, index=self.node_names)
            
            return results_df, L_scores
        except Exception as e:
            raise ValueError(f"An error occurred while compute_laplacian_scores: {e}")
        
    def find_neighborhood(self, G, S):
        neighborhood = set()
        for node in S:
            neighborhood.update(set(G.neighbors(node)) - set(S))
        return list(neighborhood)
    
    def module_quality(self, module, G):
        subgraph = G.subgraph(module)
        density = nx.density(subgraph)
        edge_weights = nx.get_edge_attributes(subgraph, 'weight').values()
        if len(edge_weights) == 0:
            return 0
        average_weight = sum(edge_weights) / len(edge_weights)
        return density * average_weight

    def modularity(self, module, G):
        subgraph = G.subgraph(module)
        
        # Internal Density
        internal_density = nx.density(subgraph)
        
        # Conductance
        cut_edges = nx.cut_size(G, module)
        conductance = cut_edges / (2 * len(module))
        
        # Expansion
        expansion = cut_edges / len(module)
        
        # Cut Ratio
        cut_ratio = cut_edges / (len(module) * (len(G.nodes()) - len(module)))
        
        # Combining all metrics into a single quality score
        # Adjust the weights as needed based on the importance of each metric
        quality_score = (internal_density + (1 - conductance) + (1 - expansion) + (1 - cut_ratio)) / 4
        
        return quality_score  
      
    def MG_algorithm(self, G, T=10, T_low=1, min_comm_size=4, max_comm_size=10, threshold=0.9):                
        modules = []
        nodes_to_process = set(G.nodes())
        node_participation = defaultdict(int)
        
        if min_comm_size <= len(list(nodes_to_process)) <= max_comm_size: 
            avg_weight = np.mean([G.nodes[node]['ls_score'] for node in nodes_to_process])
            modules.append((list(nodes_to_process), avg_weight))
            return modules
        
        self.counter = 0
        while nodes_to_process:
            seed = max(nodes_to_process, key=lambda node: G.nodes[node]['ls_score'])
            module = [seed]
            nodes_to_process.remove(seed)

            print(len(nodes_to_process))
            
            current_T = T
            improvement = True
            current_modularity = self.module_quality(module, G)

            # Grow the module
            while current_T > T_low and improvement:
                x = random.uniform(0, 1)
                avg_weight = np.mean([G.nodes[node]['ls_score'] for node in module])
                adjacent_nodes = self.find_neighborhood(G, module)
                next_node = ""
                    
                if len(adjacent_nodes) == 0:
                    break
                
                eligible_nodes = [node for node in adjacent_nodes if G.nodes[node]['ls_score'] > avg_weight]
                
                if eligible_nodes:
                    participation_counts = np.array([node_participation[node] for node in eligible_nodes])
                    weights = np.exp(-participation_counts)
                    probabilities = weights / weights.sum()
                    next_node = np.random.choice(eligible_nodes, p=probabilities)
                else:
                    node = max(adjacent_nodes, key=lambda node: G.nodes[node]['ls_score'])
                    if x < np.exp((G.nodes[node]['ls_score'] - avg_weight) / current_T):
                        next_node = node
                    else:
                        break
                    
                if next_node:
                    new_module = module + [next_node]
                    new_modularity = self.module_quality(new_module, G)
                    if new_modularity >= current_modularity:
                        avg_weight = np.mean([G.nodes[node]['ls_score'] for node in module + [next_node]])
                        module.append(next_node)
                        current_modularity = new_modularity
                    else:
                        improvement = False
                    
                current_T *= threshold
                
            if min_comm_size <= len(list(module)) <= max_comm_size: 
                avg_weight = np.mean([G.nodes[node]['ls_score'] for node in module])
                modules.append((list(module), avg_weight))
                nodes_to_process.difference_update(module)
                for node in module:
                    node_participation[node] += 1
        
        return modules
        
    def get_labeled_modules_with_scores(self, modules, graph):
        labeled_modules = []
        for module, ls_score in modules:
            # Retrieve labels for each node ID in the module
            labels = [graph.vs[node_id]['label'] for node_id in module]  # Ensure 'label' matches the attribute name
            # Append the list of labels with the average ls_score for the module
            labeled_modules.append((list(labels), ls_score))
            
        return labeled_modules
    
    def new2_algorithm(self, min_comm_size=4, max_comm_size=10):
        GNX = nx.read_gml(f"{self.cancer_type}_network_features.gml")
        G = ig.Graph.Read_GML(f"{self.cancer_type}_network_features.gml")

        # Find partitions
        partition = la.find_partition(G, la.RBConfigurationVertexPartition, weights="weight")
        modules = [list(community) for community in partition]
        
        community_sizes = [len(community) for community in modules]
        for i, size in enumerate(community_sizes):
            print(f"Community {i+1} has size {size}")
        
        labeled_modules = []
        for module in modules:
            # Retrieve labels for each node ID in the module
            labels = [G.vs[node_id]['label'] for node_id in module]  # Ensure 'label' matches the attribute name
            # Append the list of labels with the average ls_score for the module
            labeled_modules.append(list(labels))
            
        modules = []
        for partition in labeled_modules:
            subgraph = GNX.subgraph(partition)
            m = self.MG_algorithm(subgraph)
            for (module, score) in m:
                modules.append((module, score))
        new_modules = []
        for (module, ls_score) in modules:
            module_set = set(module)
            is_subset = False
            modules_to_remove = []
            for (existing_module, score, quality) in new_modules:
                existing_module_set = set(existing_module)
                if module_set.issubset(existing_module_set):
                    is_subset = True
                    break
                elif existing_module_set.issubset(module_set):
                    modules_to_remove.append(existing_module)
                        
            if not is_subset:
                quality = self.module_quality(module, GNX)
                if (quality + ls_score) / 2 > 20:
                    new_modules.append((module, ls_score, quality))
                    for module_to_remove in modules_to_remove:
                        new_modules = [module for module in new_modules if sorted(module[0]) != sorted(module_to_remove)]
        
        to_remove = set()

        for i in range(len(new_modules)):
            for j in range(i + 1, len(new_modules)):
                module_i, score_i, quality_i = new_modules[i]
                module_j, score_j, quality_j = new_modules[j]
                set_i = set(module_i)
                set_j = set(module_j)
                value_i = (quality_i + score_i) / 2
                value_i = (quality_j + score_j) / 2
                # Check if they share three or more genes
                if len(set_i.intersection(set_j)) >= 3:
                    if value_i < value_i:  # Compare quality
                        to_remove.add(i)
                    else:
                        to_remove.add(j)

        # Remove marked modules by index
        filtered_modules = [module for idx, module in enumerate(new_modules) if idx not in to_remove]

        print("len", len(new_modules))
        
        max_score = max(filtered_modules, key=lambda x: x[1])[1]
        min_score = min(filtered_modules, key=lambda x: x[1])[1]

        # To get the max and min qualities
        max_quality = max(filtered_modules, key=lambda x: x[2])[2]
        min_quality = min(filtered_modules, key=lambda x: x[2])[2]
        def composite_score(module):
            normalized_score = (module[1] - min_score) / (max_score - min_score)
            normalized_quality = (module[2] - min_quality) / (max_quality - min_quality)
            return (normalized_score + normalized_quality) /2
        
        filtered_modules.sort(key=composite_score, reverse=True)
        
        return filtered_modules