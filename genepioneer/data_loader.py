import os
import pandas as pd
from itertools import combinations

class DataLoader:
    def __init__(self, cancer_type):
        self.cancer_type = cancer_type
        self.TCGA_data_path = os.path.join("../GenesData/", self.cancer_type)
        self.GO_data_path = os.path.join("../GenesData/IBP_GO_Terms.xlsx")

    def load_tcga_genes(self):
        genes = {}
        cases = []

        gene_list_file_path = os.path.join(self.TCGA_data_path, f"{self.cancer_type}.tsv")
        genes_df = pd.read_csv(gene_list_file_path, sep='\t')
        genes_list = genes_df["Symbol"].tolist()
        for gene in genes_list:
            genes[gene] = []
            gene_path = os.path.join(self.TCGA_data_path, gene)
            case_list_file_path = os.path.join(gene_path, f"{gene}.tsv")
            cases_df = pd.read_csv(case_list_file_path, sep='\t')
            cases_list = cases_df["Case ID"].tolist()
            for case in cases_list:
                if "TCGA" in case: 
                    genes[gene].append(case)
                    if case not in cases:
                        cases.append(case)

        return (genes, cases)
    def load_go_cases(self):
        cases = {}

        cases_df = pd.read_excel(self.GO_data_path)
        for index, row in cases_df.iterrows():
            case_name = row.iloc[0]
            genes = row.iloc[1:].dropna().tolist()
            
            cases[case_name] = sorted(genes)
    
        return cases
    
    def load_tcga_connected_genes(self):
        genes, cases = self.load_tcga_genes()
        shared_cases_dict = {}
        for gene1, gene2 in combinations(genes.keys(), 2):
            shared_cases = list(set(genes[gene1]) & set(genes[gene2]))
            if shared_cases:
                shared_cases_dict[(gene1, gene2)] = shared_cases

        return (shared_cases_dict, genes ,cases)

    def load_go_connected_genes(self):
        cases = self.load_go_cases()
        shared_cases_dict = {}
        for case, genes_list in cases.items():
            for gene1, gene2 in combinations(genes_list, 2):
                if (gene1, gene2) not in shared_cases_dict:
                    shared_cases_dict[(gene1, gene2)] = [case]
                else:
                    shared_cases_dict[(gene1, gene2)].append(case)  
        return (shared_cases_dict, cases)
        

