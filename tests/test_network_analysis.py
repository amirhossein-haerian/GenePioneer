from genepioneer import NetworkAnalysis

network_analysis = NetworkAnalysis("Skin");

x = network_analysis.compute_laplacian_scores()

print(x)