from genepioneer import DataLoader

data_loader = DataLoader("Skin");
data = data_loader.load_tcga_genes()

print(data[0]["name"])
