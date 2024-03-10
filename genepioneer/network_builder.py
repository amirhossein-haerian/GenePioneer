import matplotlib.pyplot as plt

import networkx as nx
from .data_loader import DataLoader

class NetworkBuilder:
    def __init__(self, cancer_type):
        self.cancer_type = cancer_type
        self.graph = nx.Graph()
        data_loader = DataLoader(self.cancer_type)

        self.tcga_connected_genes ,self.tcga_genes, self.tcga_cases = data_loader.load_tcga_connected_genes()
        self.go_connected_genes, self.go_cases = data_loader.load_go_connected_genes()


    def build_network(self):
        self.edge_adder(self.tcga_connected_genes)
        self.edge_adder(self.go_connected_genes)

    def edge_adder(self, connected_genes):
        for genes_tuple, cases_list in connected_genes.items():
            gene1, gene2 = genes_tuple
            
            # if not self.graph.has_node(gene1):
            #     self.graph.add_node(gene1)
            # if not self.graph.has_node(gene2):
            #     self.graph.add_node(gene2)
            
            weight = self.weight_calculator(genes_tuple)
            self.graph.add_edge(gene1, gene2, weight=weight)

    def weight_calculator(self, genes_tuple):
        w1 = 0
        w2 = 0
        if(genes_tuple in self.tcga_connected_genes):
            w1 = len(self.tcga_connected_genes[genes_tuple])/len(self.tcga_cases)
        if(genes_tuple in self.go_connected_genes):
            w2 = len(self.go_connected_genes[genes_tuple])/len(self.go_cases)

        weight = w1 + w2

        return weight
    
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
    
    def draw_network(self, with_edge_weights=False):
        pos = nx.spring_layout(self.graph)  # For consistent positioning
        nx.draw(self.graph, pos, with_labels=True, node_color='skyblue', node_size=700, font_size=10)
        
        if with_edge_weights:
            edge_labels = nx.get_edge_attributes(self.graph, 'weight')
            nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=edge_labels, font_size=8)
        
        plt.show()
