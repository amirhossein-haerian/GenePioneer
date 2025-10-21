from genepioneer import NetworkBuilder

from concurrent.futures import ProcessPoolExecutor, as_completed
import pandas as pd
import networkx as nx

# "Bladder", "Brain", "Cervix", "Colon", "Corpus uteri", "Kidney", "Liver", "Ovary", "Prostate", "Skin", "Thyroid"
cancer = "Prostate_filtered"

print("Working on: ", cancer)
try:
    print("try")
    network_builder = NetworkBuilder(cancer, '../GenesData')
    graph = network_builder.build_network()
    print("graph has built")
    print(nx.is_connected(graph))
    print("nodes: ", len(graph.nodes()))
    print("edges: ", len(graph.edges()))
    features = network_builder.calculate_all_features()
    network_builder.save_features_to_csv(
        features, f"{cancer}_network_features")
    print("done")
except Exception as exc:
    print("catch")
    print(exc)


def process_cancer(cancer):
    print("Working on: ", cancer)
    network_builder = NetworkBuilder(cancer)
    graph = network_builder.build_network()

    features = network_builder.calculate_all_features()
    network_builder.save_features_to_csv(
        features, f"{cancer}_network_features")
