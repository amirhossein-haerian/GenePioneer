from genepioneer import NetworkBuilder

network_builder = NetworkBuilder("Kidney")

network_builder.build_network()
network_builder.print_network_summary()
network_builder.analyze_nodes_edges()
# network_builder.compute_all_features()
# print(network_builder.graph.nodes("TP53"))
features = network_builder.calculate_all_features()
network_builder.save_features_to_csv(features, 'network_features.csv')