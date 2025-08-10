import os
import pandas as pd
from itertools import combinations
from collections import defaultdict

class DataLoader:
    def __init__(self, cancer_type, file_path):
        self.cancer_type = cancer_type
        self.TCGA_data_path = os.path.join(file_path, self.cancer_type)
        self.IBM_data_path = os.path.join(f"{file_path}/IBP_GO_Terms.xlsx")


    def load_TCGA(self):
        genes_list_file_path = os.path.join(self.TCGA_data_path, f"{self.cancer_type}.tsv")
        genes_df = pd.read_csv(genes_list_file_path, sep='\t')
        # Support different capitalizations of the column containing gene names
        # The original dataset used "Symbol" but newer TSV files might use
        # lowercase "symbol".  Try to find the correct column in a
        # case-insensitive manner.
        symbol_column = None
        for col in genes_df.columns:
            if col.lower() == "symbol":
                symbol_column = col
                break

        if symbol_column is None:
            raise KeyError(
                "The genes list file must contain a 'symbol' column"
            )

        genes = genes_df[symbol_column].dropna().tolist()
        return genes
    
    def load_IBM(self):
        processes_with_genes = defaultdict(set)

        processes_df = pd.read_excel(self.IBM_data_path)
        for index, row in processes_df.iterrows():
            process_name = row.iloc[0]
            genes_list = row.iloc[1:].dropna().tolist()
            processes_with_genes[process_name] = set(genes_list)

        genes_with_processes = defaultdict(set)

        for process, genes in processes_with_genes.items():
            for gene in genes:
                genes_with_processes[gene].add(process)               
  
        total_processes = len(processes_with_genes)
        return genes_with_processes, processes_with_genes, total_processes
        

