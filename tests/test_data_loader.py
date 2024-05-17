import json 

from genepioneer import DataLoader

data_loader = DataLoader("Skin");

genes_with_cases ,cases_with_genes, total_cases = data_loader.load_TCGA()
genes_with_processes, processes_with_genes, total_processes = data_loader.load_IBM()
print(processes_with_genes)