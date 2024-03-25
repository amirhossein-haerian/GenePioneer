import json 

from genepioneer import DataLoader

data_loader = DataLoader("Skin");

genes, processes = data_loader.load_TCGA()

print(genes["TP53"])