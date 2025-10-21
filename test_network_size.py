#!/usr/bin/env python3

import networkx as nx
import os

def test_network_size():
    try:
        # Load the network
        network_path = "tests/Prostate_filtered_network_features.gml"
        if os.path.exists(network_path):
            G = nx.read_gml(network_path)
            print(f"Network file found: {network_path}")
            print(f"Network has {G.number_of_nodes()} nodes")
            print(f"Network has {G.number_of_edges()} edges")
            print(f"Sample nodes: {list(G.nodes())[:5]}")
            return G.number_of_nodes()
        else:
            print(f"Network file not found: {network_path}")
            return None
    except Exception as e:
        print(f"Error loading network: {e}")
        return None

if __name__ == "__main__":
    test_network_size()
