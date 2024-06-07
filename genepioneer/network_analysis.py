import numpy as np
import pandas as pd
import os
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MaxAbsScaler

import heapq
import statistics

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
        
    def calculate_average_weight(self, G, S):
        weights = [G.nodes[node]['ls_score'] for node in S]
        return sum(weights) / len(weights) if weights else 0
    
    def calculate_average_edge_weight(self, G, S):
        if len(S) < 2:
            return 0
        edge_weights = [G[u][v]['process_weight'] for u in S for v in S if u != v and G.has_edge(u, v)]
        return sum(edge_weights) / len(edge_weights) if edge_weights else 0

    def find_neighborhood(self, G, S):
        neighborhood = set()
        for node in S:
            neighborhood.update(set(G.neighbors(node)) - set(S))
        return list(neighborhood)
    
    def module_quality(self, module, G):
        subgraph = G.subgraph(module)
        density = nx.density(subgraph)
        connectivity = nx.average_node_connectivity(subgraph)
        return density + connectivity
        
    def MG_algorithm(self, T=10, T_low=0.01, min_comm_size=4, max_comm_size=10, threshold=0.8):
        G = nx.read_gml(f"{self.cancer_type}_network_features.gml")
        
        modules = []
        unvisited_nodes = set(G.nodes())
        
        while unvisited_nodes:
            # Select a random seed node
            seed = random.choice(tuple(unvisited_nodes))
            S = {seed}
            
            print(self.counter)
            self.counter += 1
            
            T_current = T            
            unvisited_nodes.remove(seed)
                
            checked_nodes = set()            
            # Grow the module
            while T_current > T_low:
                candidate_nodes = [(G.nodes[node]['ls_score'], node) for node in self.find_neighborhood(G, S)]
                avg_weight = self.calculate_average_weight(G, S)
                if candidate_nodes:
                    ls_score, random_candidate = random.choice(candidate_nodes)
                    while random_candidate in checked_nodes:
                        ls_score, random_candidate = random.choice(candidate_nodes)
                    if G.nodes[random_candidate]['ls_score'] > avg_weight:
                        S.add(random_candidate)
                        # if random_candidate in unvisited_nodes:
                        #     unvisited_nodes.remove(random_candidate)
                    else:
                        x = random.uniform(0, 1)
                        if x < np.exp((G.nodes[random_candidate]['ls_score'] - avg_weight) / T_current):
                            S.add(random_candidate)
                            # if random_candidate in unvisited_nodes:
                            #     unvisited_nodes.remove(random_candidate)
                        else:
                            checked_nodes.add(random_candidate)
                            
                else:
                    break
                    
                T_current *= threshold
                
            avg_edge_weight = self.calculate_average_edge_weight(G, S)
            if min_comm_size <= len(list(S)) <= max_comm_size: 
                modules.append((list(S), avg_weight, avg_edge_weight))
        
        
        modules.sort(key=lambda x: x[1]+x[2]/2, reverse=True)
                                
        unique_communities = []
        seen_communities = set()

        for comm, ls_score, edge_weight in modules:
            comm_set = set(comm)
            if not any(comm_set <= set(other) for other in seen_communities):
                subsets_to_remove = [other for other in seen_communities if set(other) <= comm_set]
                unique_communities = [
                    community for community in unique_communities
                    if set(community[0]) not in subsets_to_remove
                ]

                seen_communities.add(tuple(comm))
                unique_communities.append((comm, ls_score, edge_weight))
                        
        return unique_communities
        
    def calculate_weight_threshold(self, G):
        ls_scores = [G.nodes[node]['ls_score'] for node in G.nodes()]
        std_ls_score = np.std(ls_scores)
        weight_diff_threshold = std_ls_score * 4
        return weight_diff_threshold
        
        
    def new_MG_algorithm(self, T=10, T_low=0.01, min_comm_size=4, max_comm_size=10, threshold=0.9):
        G = nx.read_gml(f"{self.cancer_type}_network_features.gml")
        
        modules = []
        nodes_to_process = set(G.nodes())
        
        while nodes_to_process:
            seed = random.choice(list(nodes_to_process))
            module = [seed]
            nodes_to_process.remove(seed)
            
            print(self.counter)
            self.counter += 1   
            
            current_T = T
            # Grow the module
            while current_T > T_low:
                avg_weight = np.mean([G.nodes[node]['ls_score'] for node in module])
                adjacent_nodes = self.find_neighborhood(G, module)
                next_node = ""
                    
                if len(adjacent_nodes) == 0:
                    break
                
                eligible_nodes = [node for node in adjacent_nodes if G.nodes[node]['ls_score'] > avg_weight]
                
                if eligible_nodes:
                    next_node = np.random.choice(eligible_nodes)
                else:
                    x = random.uniform(0, 1)
                    node = max(adjacent_nodes, key=lambda node: G.nodes[node]['ls_score'])
                    if x < np.exp((G.nodes[node]['ls_score'] - avg_weight) / current_T):
                        next_node = node
                    else:
                        break
                    
                if next_node:    
                    module.append(next_node)
                    
                current_T *= threshold
            print(len(module))
            if min_comm_size <= len(list(module)) <= max_comm_size: 
                modules.append((list(module), avg_weight))
                nodes_to_process.difference_update(module)       
        
        
        max_avg_weight = max(modules, key=lambda x: x[1])[1]
        max_module_quality = max(modules, key=lambda x: x[2])[2]
        
        def composite_score(module):
            normalized_weight = module[1] / max_avg_weight if max_avg_weight else 0
            normalized_quality = module[2] / max_module_quality if max_module_quality else 0
            return normalized_weight + normalized_quality
        
        modules.sort(key=composite_score, reverse=True)
                                
        unique_communities = []
        seen_communities = set()

        for comm, ls_score, quality_score in modules:
            comm_set = set(comm)
            if not any(comm_set <= set(other) for other in seen_communities):
                subsets_to_remove = [other for other in seen_communities if set(other) <= comm_set]
                unique_communities = [
                    community for community in unique_communities
                    if set(community[0]) not in subsets_to_remove
                ]

                seen_communities.add(tuple(comm))
                unique_communities.append((comm, ls_score, quality_score))
                        
        return unique_communities
        


    def load_and_prepare_graph(self, file_name):
        G = ig.Graph.Read_GML(f"{file_name}.gml")
        
        return G
    
    def find_partitions(self, graph, min_size, max_size):
        partitions = []
        for size in range(min_size, max_size + 1):
            partition = la.find_partition(graph, la.ModularityVertexPartition, max_comm_size=size)
            # partition = la.find_partition(graph, la.CPMVertexPartition, weights='process_weight', resolution_parameter=0.1, max_comm_size=size, n_iterations=-1)

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
            
        top_partitions = self.select_top_partitions(filtered_partitions)
            
        return top_partitions
    
    def select_top_partitions(self, partitions):
        ls_scores = [item[1] for item in partitions]
        edge_weights = [item[2] for item in partitions]
        
        # Normalize both metrics
        ls_scores_normalized = (ls_scores - np.min(ls_scores)) / (np.max(ls_scores) - np.min(ls_scores))
        edge_weights_normalized = (edge_weights - np.min(edge_weights)) / (np.max(edge_weights) - np.min(edge_weights))

        # Combine the normalized metrics (weighted sum, equal weight here but can be adjusted)
        combined_scores = 0.5 * ls_scores_normalized + 0.5 * edge_weights_normalized

        # Sort based on combined scores
        sorted_indices = np.argsort(combined_scores)[::-1]
        top_indices = sorted_indices[:20]

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
        
    
