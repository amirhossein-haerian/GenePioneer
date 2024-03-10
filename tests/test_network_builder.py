from genepioneer import NetworkBuilder

network_builder = NetworkBuilder("Kidney")

network_builder.build_network()
network_builder.print_network_summary()
network_builder.analyze_nodes_edges()
# network_builder.draw_network()