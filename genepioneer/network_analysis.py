import numpy as np
import pandas as pd
import os
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MaxAbsScaler

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
        self.counter = 0
        if features:
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

    def load_and_prepare_graph(self, file_name):
        G = ig.Graph.Read_GML(f"{file_name}.gml")
        
        return G
    
    def find_partitions(self, graph, min_size, max_size):
        partitions = []
        for size in range(min_size, max_size + 1):
            # partition = la.find_partition(graph, la.ModularityVertexPartition, weights='process_weight', max_comm_size=size, n_iterations=-1)
            partition = la.find_partition(graph, la.CPMVertexPartition, weights='process_weight', resolution_parameter=0.05, max_comm_size=size, n_iterations=-1)

            avg_ls_scores = [sum(graph.vs[community]['ls_score']) / len(community) for community in partition]
            avg_edge_weights = [self.calculate_avg_edge_weight(graph, community) for community in partition]
                    
            self.counter += 1
            print(self.counter)
            
            partitions.append((partition, avg_ls_scores, avg_edge_weights)) 
        return partitions
    
    def calculate_avg_edge_weight(self, graph, community):
        edge_weights = []
        processed_edges = set()

        for v in community:
            for neighbor in graph.neighbors(v, mode="all"):
                if neighbor in community:
                    edge_id = graph.get_eid(v, neighbor)
                    # Ensure the edge is only added once
                    if edge_id not in processed_edges:
                        edge_weights.append(graph.es[edge_id]['process_weight'])
                        processed_edges.add(edge_id)
                        
        return sum(edge_weights) / len(edge_weights) if edge_weights else 0
    
    def filter_partitions(self, partitions, min_comm_size, max_comm_size):
        filtered_partitions = []
        total_communities = 0

        for partition, ls_scores, edge_weights in partitions:
            total_communities += len(partition)
            filtered_comm = [
                (comm, ls_score, edge_weight)
                for comm, ls_score, edge_weight in zip(partition, ls_scores, edge_weights)
                if min_comm_size <= len(comm) <= max_comm_size
            ]
            unique_communities = self.remove_nested_communities(filtered_comm)
            filtered_partitions.extend(unique_communities)
            
        print(total_communities, len(filtered_partitions))
        return filtered_partitions
            
    def remove_nested_communities(self, communities):
        unique_communities = []
        seen_communities = set()

        for comm, ls_score, edge_weight in communities:
            if not any(set(tuple(comm)) <= set(other) for other in seen_communities):
                if any(set(other) <= set(comm) for other in seen_communities):
                    unique_communities = [
                        community for community in unique_communities
                        if community[0] != comm
                    ]

                seen_communities.add(tuple(comm))
                unique_communities.append((comm, ls_score, edge_weight))

        return unique_communities

    def filter_and_select_partitions(self, partitions, min_comm_size, max_comm_size):
        filtered_partitions = self.filter_partitions(partitions, min_comm_size, max_comm_size)
        partitions_by_length = defaultdict(list)
        
        for partition, ls_score, edge_weight in filtered_partitions:
            length = len(partition)
            partitions_by_length[length].append((partition, ls_score, edge_weight))
            
        top_modules_by_length = []
        for length, partition_list in partitions_by_length.items():
            top_partitions = self.select_top_partitions(partition_list)
            top_modules_by_length.extend(top_partitions)
            
        return top_modules_by_length
    
    def select_top_partitions(self, partitions):
        ls_scores = [item[1] for item in partitions]
        edge_weights = [item[2] for item in partitions]
        
        print("s", np.max(edge_weights))

        # Normalize both metrics
        ls_scores_normalized = (ls_scores - np.min(ls_scores)) / (np.max(ls_scores) - np.min(ls_scores) + 1e-9)
        edge_weights_normalized = (edge_weights - np.min(edge_weights)) / (np.max(edge_weights) - np.min(edge_weights) + 1e-9)

        # Combine the normalized metrics (weighted sum, equal weight here but can be adjusted)
        combined_scores = 0.5 * ls_scores_normalized + 0.5 * edge_weights_normalized

        # Sort based on combined scores
        sorted_indices = np.argsort(combined_scores)[::-1]
        top_indices = sorted_indices[:5]

        # Select top partitions based on combined scores
        top_partitions = [partitions[i] for i in top_indices]

        return top_partitions


    def get_labeled_modules_with_scores(self, modules, graph):
        labeled_modules = []
        for module, ls_score, weight_score in modules:
            # Retrieve labels for each node ID in the module
            labels = [graph.vs[node_id]['label'] for node_id in module]  # Ensure 'label' matches the attribute name
            # Append the list of labels with the average ls_score for the module
            labeled_modules.append((list(labels), ls_score, weight_score))
            
        return labeled_modules
    
    def new_algorithm(self):
        
        G = self.load_and_prepare_graph(f"{self.cancer_type}_network_features")

        # Find partitions
        partitions = self.find_partitions(G, 4, 100)

        # Filter and select top partitions
        top_modules = self.filter_and_select_partitions(partitions, 4, 10)  
        labeled_modules_with_scores = self.get_labeled_modules_with_scores(top_modules, G)    
        
        return labeled_modules_with_scores
        
    
