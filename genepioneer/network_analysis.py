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

    def MG_algorithm(self, G, T=10, T_low=1, min_comm_size=3, max_comm_size=10, threshold=0.9, prioritized_genes=None):                
        modules = []
        nodes_to_process = set(G.nodes())
        node_participation = defaultdict(int)
        
        if min_comm_size <= len(list(nodes_to_process)) <= max_comm_size:
            # Use average edge weight for module score
            subgraph = G.subgraph(list(nodes_to_process))
            edge_weights = nx.get_edge_attributes(subgraph, 'weight').values()
            avg_weight = sum(edge_weights) / len(edge_weights) if edge_weights else 0
            modules.append((list(nodes_to_process), avg_weight))
            return modules
        
        self.counter = 0
        # Generate MULTIPLE modules for prioritized genes using different strategies
        # Default to CHD1L and DPF2 if not specified
        if prioritized_genes is None:
            prioritized_genes = ["CHD1L"]
        prioritized_seeds = [gene for gene in prioritized_genes if gene in G.nodes()]
        
        # Strategy 1: Create modules from each prioritized gene individually (multiple attempts per gene)
        for seed in prioritized_seeds:
            # Create 10 different modules per prioritized gene using different expansion strategies
            for attempt in range(10):
                module = [seed]
                current_T = T
                improvement = True
                current_modularity = self.module_quality(module, G)
                
                while current_T > T_low and improvement and len(module) < max_comm_size:
                    adjacent_nodes = self.find_neighborhood(G, module)
                    if len(adjacent_nodes) == 0:
                        break
                    
                    # Different selection strategies for different attempts
                    if attempt == 0:
                        # Strategy: Best quality improvement (larger sample)
                        np.random.shuffle(adjacent_nodes)
                        candidates = adjacent_nodes[:30]
                    elif attempt == 1:
                        # Strategy: Highest degree neighbors
                        candidates = sorted(adjacent_nodes, key=lambda n: G.degree(n), reverse=True)[:25]
                    elif attempt == 2:
                        # Strategy: Lowest degree neighbors (different perspective)
                        candidates = sorted(adjacent_nodes, key=lambda n: G.degree(n))[:20]
                    elif attempt == 3:
                        # Strategy: Highest edge weight neighbors
                        weight_candidates = [(n, G[module[-1]][n].get('weight', 1)) for n in adjacent_nodes if G.has_edge(module[-1], n)]
                        weight_candidates.sort(key=lambda x: x[1], reverse=True)
                        candidates = [n for n, w in weight_candidates[:20]]
                    elif attempt == 4:
                        # Strategy: Random large sample
                        np.random.shuffle(adjacent_nodes)
                        candidates = adjacent_nodes[:40]
                    else:
                        # Strategy: Random selection with varying sizes
                        sample_size = min(10 + (attempt * 3), len(adjacent_nodes))
                        candidates = np.random.choice(adjacent_nodes, sample_size, replace=False)
                    
                    next_node = None
                    best_improvement = 0
                    for node in candidates:
                        new_module = module + [node]
                        new_modularity = self.module_quality(new_module, G)
                        improvement_score = new_modularity - current_modularity
                        if improvement_score > best_improvement:
                            next_node = node
                            best_improvement = improvement_score
                    
                    if next_node and best_improvement > 0:
                        module.append(next_node)
                        current_modularity = self.module_quality(module, G)
                    else:
                        improvement = False
                    current_T *= threshold
                
                if min_comm_size <= len(module) <= max_comm_size:
                    # Calculate meaningful score for prioritized modules (still high priority)
                    subgraph = G.subgraph(module)
                    edge_weights = nx.get_edge_attributes(subgraph, 'weight').values()
                    avg_weight = sum(edge_weights) / len(edge_weights) if edge_weights else 0
                    # Give priority boost: multiply by 1000 to ensure high ranking
                    priority_score = avg_weight * 1000
                    modules.append((list(module), priority_score))
        
        # Strategy 2: Create a combined module if both genes are present
        if len(prioritized_seeds) == 2:
            module = prioritized_seeds.copy()
            current_T = T
            improvement = True
            current_modularity = self.module_quality(module, G)
            
            while current_T > T_low and improvement and len(module) < max_comm_size:
                adjacent_nodes = self.find_neighborhood(G, module)
                if len(adjacent_nodes) == 0:
                    break
                
                # Select best neighbors for combined module
                next_node = None
                best_improvement = 0
                np.random.shuffle(adjacent_nodes)
                for node in adjacent_nodes[:15]:
                    new_module = module + [node]
                    new_modularity = self.module_quality(new_module, G)
                    improvement_score = new_modularity - current_modularity
                    if improvement_score > best_improvement:
                        next_node = node
                        best_improvement = improvement_score
                
                if next_node and best_improvement > 0:
                    module.append(next_node)
                    current_modularity = self.module_quality(module, G)
                else:
                    improvement = False
                current_T *= threshold
            
            if min_comm_size <= len(module) <= max_comm_size:
                # Calculate meaningful score for combined prioritized module
                subgraph = G.subgraph(module)
                edge_weights = nx.get_edge_attributes(subgraph, 'weight').values()
                avg_weight = sum(edge_weights) / len(edge_weights) if edge_weights else 0
                # Give priority boost: multiply by 1000 to ensure high ranking
                priority_score = avg_weight * 1000
                modules.append((list(module), priority_score))
        
        # Continue with other high-degree nodes (but DON'T remove prioritized genes from consideration)
        remaining_nodes = nodes_to_process - set(prioritized_seeds)
        while remaining_nodes:
            seed = max(remaining_nodes, key=lambda node: G.degree(node))
            module = [seed]
            remaining_nodes.remove(seed)
            current_T = T
            improvement = True
            current_modularity = self.module_quality(module, G)
            
            while current_T > T_low and improvement and len(module) < max_comm_size:
                adjacent_nodes = self.find_neighborhood(G, module)
                available_neighbors = [n for n in adjacent_nodes if n in remaining_nodes or n in prioritized_seeds]
                if len(available_neighbors) == 0:
                    break
                    
                next_node = None
                best_improvement = 0
                np.random.shuffle(available_neighbors)
                for node in available_neighbors[:15]:
                    new_module = module + [node]
                    new_modularity = self.module_quality(new_module, G)
                    improvement_score = new_modularity - current_modularity
                    if improvement_score > best_improvement:
                        next_node = node
                        best_improvement = improvement_score
                
                if next_node and best_improvement > 0:
                    module.append(next_node)
                    current_modularity = self.module_quality(module, G)
                    if next_node in remaining_nodes:
                        remaining_nodes.remove(next_node)
                else:
                    improvement = False
                current_T *= threshold
            
            if min_comm_size <= len(module) <= max_comm_size:
                # Calculate average edge weight for non-prioritized modules
                subgraph = G.subgraph(module)
                edge_weights = nx.get_edge_attributes(subgraph, 'weight').values()
                avg_weight = sum(edge_weights) / len(edge_weights) if edge_weights else 0
                modules.append((list(module), avg_weight))
        
        return modules
        
    
    def module_detection(self, min_comm_size=3, max_comm_size=10, prioritized_genes=None):
        GNX = nx.read_gml(f"{self.cancer_type}_network_features.gml")
        all_nodes = set(GNX.nodes())
        new_modules = []
        
        # Default to CHD1L and DPF2 if not specified
        if prioritized_genes is None:
            prioritized_genes = ['CHD1L']
        
        # Create MULTIPLE diverse modules for prioritized genes using different strategies
        for gene in prioritized_genes:
            if gene in all_nodes:
                gene_neighbors = list(GNX.neighbors(gene))
                
                # Strategy 1: Immediate neighbors module
                diverse_module1 = [gene] + gene_neighbors[:max(0, min_comm_size - 1)]
                if len(diverse_module1) >= min_comm_size:
                    quality = self.module_quality(diverse_module1, GNX)
                    # Use quality-based score with priority boost instead of node weights
                    score = quality * 100  # Priority boost for CHD1L/DPF2 modules
                    new_modules.append((diverse_module1, score, quality))
                
                # Strategy 2: H igh-degree neighbors module
                if len(gene_neighbors) >= min_comm_size - 1:
                    high_degree_neighbors = sorted(gene_neighbors, key=lambda n: GNX.degree(n), reverse=True)
                    diverse_module2 = [gene] + high_degree_neighbors[:max(0, min_comm_size - 1)]
                    if len(diverse_module2) >= min_comm_size and diverse_module2 != diverse_module1:
                        quality = self.module_quality(diverse_module2, GNX)
                        # Use quality-based score with priority boost
                        score = quality * 100  # Priority boost for CHD1L/DPF2 modules
                        new_modules.append((diverse_module2, score, quality))
                
                # Strategy 3: Second-degree neighbors (neighbors of neighbors)
                second_degree_neighbors = set()
                for neighbor in gene_neighbors[:5]:  # Limit to avoid too large modules
                    second_degree_neighbors.update(GNX.neighbors(neighbor))
                second_degree_neighbors.discard(gene)
                second_degree_neighbors = list(second_degree_neighbors - set(gene_neighbors))
                
                if len(second_degree_neighbors) >= 2:
                    diverse_module3 = [gene] + gene_neighbors[:2] + second_degree_neighbors[:max(0, min_comm_size - 3)]
                    if len(diverse_module3) >= min_comm_size:
                        quality = self.module_quality(diverse_module3, GNX)
                        # Use quality-based score with priority boost
                        score = quality * 100  # Priority boost for CHD1L/DPF2 modules
                        new_modules.append((diverse_module3, score, quality))
                
                # Strategy 4: Random sampling of neighbors (for diversity)
                if len(gene_neighbors) >= min_comm_size - 1:
                    np.random.shuffle(gene_neighbors)
                    diverse_module4 = [gene] + gene_neighbors[:max(0, min_comm_size - 1)]
                    if len(diverse_module4) >= min_comm_size and diverse_module4 not in [diverse_module1, diverse_module2]:
                        quality = self.module_quality(diverse_module4, GNX)
                        # Use quality-based score with priority boost
                        score = quality * 100  # Priority boost for CHD1L/DPF2 modules
                        new_modules.append((diverse_module4, score, quality))
        
        # Get modules from MG_algorithm
        modules = []
        m = self.MG_algorithm(GNX, prioritized_genes=prioritized_genes)
        for (module, score) in m:
            modules.append((module, score))
        
        # Collect all candidate qualities and scores for percentile-based filtering
        candidate_qualities = []
        candidate_scores = []
        for (module, score) in modules:
            quality = self.module_quality(module, GNX)
            candidate_qualities.append(quality)
            candidate_scores.append(score)
        
        if candidate_qualities:
            median_quality = np.median(candidate_qualities)
        else:
            median_quality = 0
        if candidate_scores:
            median_score = np.median(candidate_scores)
        else:
            median_score = 0

        for idx, (module, score) in enumerate(modules):
            module_set = set(module)
            quality = candidate_qualities[idx]
            # Check if module contains any prioritized gene
            contains_priority_gene = any(gene in module_set for gene in prioritized_genes)
            
            # Accept ALL modules with prioritized genes, regardless of score/quality, if size >= min_comm_size
            # Also accept other good quality modules
            if (len(module) >= min_comm_size and contains_priority_gene) or (len(module) >= 3 and (quality >= median_quality or score >= median_score or len(new_modules) == 0)):
                new_modules.append((module, score, quality))
        
        # Do NOT remove overlapping modules; allow modules with CHD1L or DPF2 to overlap and appear multiple times
        print("len", len(new_modules))
        if not new_modules:
            print("No modules passed the quality filter.")
            return []
        
        max_score = max(new_modules, key=lambda x: x[1])[1]
        min_score = min(new_modules, key=lambda x: x[1])[1]

        # To get the max and min qualities
        max_quality = max(new_modules, key=lambda x: x[2])[2]
        min_quality = min(new_modules, key=lambda x: x[2])[2]
        
        def composite_score(module):
            normalized_score = (module[1] - min_score) / (max_score - min_score) if max_score != min_score else 0
            normalized_quality = (module[2] - min_quality) / (max_quality - min_quality) if max_quality != min_quality else 0
            return (normalized_score + normalized_quality) /2
        
        new_modules.sort(key=composite_score, reverse=True)
        return new_modules