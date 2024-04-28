import json 

from genepioneer import DataLoader

data_loader = DataLoader("Skin");

genes, processes, total_cases = data_loader.load_TCGA()

print(total_cases)