import numpy as np
import pandas as pd
import os
from scipy.spatial.distance import cdist

import igraph as ig
from igraph import Graph
import leidenalg as la

import networkx as nx
import random

class NetworkAnalysis:

    def __init__(self, cancer_type, features={}):
        self.cancer_type = cancer_type
        
        self.feature_dict = features
        self.feature_matrix = None
        self.node_names = None
        
        self.load_data()
        
    def load_data(self):
        try:
            # Load the data
            #df = pd.read_csv(os.path.join("../tests/", f"{self.cancer_type}_network_features.csv"))
            df = pd.DataFrame.from_dict(self.feature_dict, orient='index')
            df.reset_index(inplace=True)
            df.rename(columns={'index': 'node'}, inplace=True)
                        
            # Drop the last row and the first column containing node names
            
            self.node_names = df.iloc[:, 0]
            self.feature_matrix = df.iloc[:, 1:].values.astype(np.float64)
            
            
        except Exception as e:
            raise ValueError(f"An error occurred while reading the file: {e}")
       
    def euclidean_distance_vec(self):
        # Using scipy's cdist to compute all pairwise Euclidean distances efficiently
        return cdist(self.feature_matrix, self.feature_matrix, 'euclidean')

    def compute_similarity_matrix(self, delta=5, t=100):
        dist_matrix = self.euclidean_distance_vec()
                
        # Apply the exponential similarity function and the threshold
        # This operation is vectorized for efficiency
        S = np.exp(-(dist_matrix**2) / t) * (dist_matrix < delta)
        return S
    
    def compute_laplacian_scores(self):
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
            numerator = np.matmul(np.matmul(np.transpose(F_j_tilde),L),F_j_tilde)
            denominator = np.matmul(np.matmul(np.transpose(F_j_tilde),D),F_j_tilde)
            L_scores[j] = numerator / denominator if denominator != 0 else 0
        # Compute the LS for each gene
        LS = self.feature_matrix @ L_scores
        results_df = pd.DataFrame({
            'LaplacianScore': LS
        }, index=self.node_names)
        
        return results_df, L_scores
    

    def calculate_average_weight(self, G, S):
        weights = [G.nodes[node]['ls_score'] for node in S]
        return sum(weights) / len(weights) if weights else 0

    def find_neighborhood(self, G, S):
        return set(nx.node_boundary(G, S))
        
    def MG_algorithm(self, T=10, T_low=0.01):
        G = nx.read_gml(f"{self.cancer_type}_network_features.gml")
        
        modules = []
        unvisited_nodes = set(G.nodes())
        
        while unvisited_nodes:
            # Select a random seed node
            seed = random.choice(tuple(unvisited_nodes))
            S = {seed}
            if seed in unvisited_nodes: 
                unvisited_nodes.remove(seed)
            T_current = T
            
            # Grow the module
            while T_current > T_low:
                print(T_current)
                C_S = self.find_neighborhood(G, S)
                avg_weight = self.calculate_average_weight(G, S)
                candidates = [(node, G.nodes[node]['ls_score']) for node in C_S]
                candidates.sort(key=lambda x: x[1], reverse=True)
                print(candidates[0])
                # Decide whether to add the highest-weighted node
                if candidates and candidates[0][1] > avg_weight:
                    S.add(candidates[0][0])
                    if candidates[0][0] in unvisited_nodes: 
                        unvisited_nodes.remove(candidates[0][0])
                    print("here")
                else:
                    x = random.uniform(0, 1)
                    if candidates and x < np.exp((candidates[0][1] - avg_weight) / T_current):
                        S.add(candidates[0][0])
                        if candidates[0][0] in unvisited_nodes: 
                            unvisited_nodes.remove(candidates[0][0])
                
                T_current *= 0.9
            
            modules.append((list(S), avg_weight))
        
        return modules
    
    def calculate_intra_module_metrics(self, G, module):
        avg_ls_score = sum(G.nodes[node]['ls_score'] for node in module) / len(module)
        # Additional metrics can be included here
        return avg_ls_score


    def load_and_prepare_graph(self, file_name):
        G = ig.Graph.Read_GML(f"{file_name}.gml")
        G.es['ls_score'] = [(G.vs[edge.source]['ls_score'] + G.vs[edge.target]['ls_score']) / 2 for edge in G.es]
        return G

    def find_partitions(self, graph, min_size, max_size):
        partitions = []
        for size in range(min_size, max_size + 1):
            partition = la.find_partition(graph, la.ModularityVertexPartition, weights='ls_score', max_comm_size=size, n_iterations=-1, seed=82)
            avg_ls_scores = [sum(graph.vs[comm]['ls_score']) / len(comm) for comm in partition]
            partitions.append((partition, avg_ls_scores))
        return partitions

    def filter_and_select_partitions(self, partitions, min_comm_size, max_comm_size):
        filtered_partitions = []
        total_communities = 0
        for partition, scores in partitions:
            total_communities += len(partition)
            filtered_comm = [(comm, score) for comm, score in zip(partition, scores) if min_comm_size <= len(comm) <= max_comm_size]
            # Remove nested communities
            unique_communities = []
            seen_communities = set()
            for comm, score in filtered_comm:
                sorted_comm = tuple(sorted(comm))  # Convert list to sorted tuple for consistency and comparison
                # Check if this community is subset of another or if it's already seen
                if not any(set(sorted_comm).issubset(set(other)) for other in seen_communities) and sorted_comm not in seen_communities:
                    seen_communities.add(sorted_comm)
                    unique_communities.append((comm, score))

            filtered_partitions.extend(unique_communities)
        print(total_communities, len(filtered_partitions))
        # Sort by average 'ls_score' and select top 20
        top_modules = sorted(filtered_partitions, key=lambda x: x[1], reverse=True)[:20]
        return top_modules
    
    def get_labeled_modules_with_scores(self, modules, graph):
        labeled_modules = []
        for module, score in modules:
            # Retrieve labels for each node ID in the module
            labels = [graph.vs[node_id]['label'] for node_id in module]  # Ensure 'label' matches the attribute name
            # Append the list of labels with the average ls_score for the module
            labeled_modules.append((list(labels), score))
            
        return labeled_modules
    
    def new_algorithm(self):
        
        G = self.load_and_prepare_graph(f"{self.cancer_type}_network_features")

        # Find partitions
        partitions = self.find_partitions(G, 5, 30)

        # Filter and select top partitions
        top_modules = self.filter_and_select_partitions(partitions, 5, 15)  
        labeled_modules_with_scores = self.get_labeled_modules_with_scores(top_modules, G)    
        
        return labeled_modules_with_scores
        
    
