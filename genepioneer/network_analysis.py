import numpy as np
import pandas as pd
import os
from scipy.spatial.distance import cdist

import networkx as nx
import random

class NetworkAnalysis:

    def __init__(self, cancer_type, features):
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
            unvisited_nodes.remove(seed)
            T_current = T
            
            # Grow the module
            while T_current > T_low:
                C_S = self.find_neighborhood(G, S)
                avg_weight = self.calculate_average_weight(G, S)
                candidates = [(node, G.nodes[node]['ls_score']) for node in C_S]
                candidates.sort(key=lambda x: x[1], reverse=True)
                
                # Decide whether to add the highest-weighted node
                if candidates and candidates[0][1] > avg_weight:
                    S.add(candidates[0][0])
                    unvisited_nodes.remove(candidates[0][0])
                else:
                    x = random.uniform(0, 1)
                    if candidates and x < (candidates[0][1] - avg_weight) / (T_current - avg_weight):
                        S.add(candidates[0][0])
                        unvisited_nodes.remove(candidates[0][0])
                
                T_current *= 0.9
            
            modules.append(S)
        
        return modules