import numpy as np
import pandas as pd
import os
from scipy.spatial.distance import cdist

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
            
            print(self.feature_matrix)
            
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
        print(m,n)
        J = np.ones((m, 1))
        F = self.feature_matrix.T
        L_scores = np.zeros(n)
        for j in range(n):
            F_j = F[j].reshape(-1, 1)
            weighted_mean = (F_j.T @ D @ J) / (J.T @ D @ J)
            # Subtract the weighted mean from each element of F_j to get F_j_tilde
            F_j_tilde = F_j - weighted_mean
            numerator = F_j_tilde.T @ L @ F_j_tilde
            denominator = F_j_tilde.T @ D @ F_j_tilde
            L_scores[j] = numerator / denominator if denominator != 0 else 0
        # Compute the LS for each gene
        print(L_scores)
        LS = self.feature_matrix @ L_scores
        results_df = pd.DataFrame({
            'LaplacianScore': LS
        }, index=self.node_names)
        
        return results_df, L_scores