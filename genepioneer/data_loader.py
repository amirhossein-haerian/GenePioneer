import os
import pandas as pd
from itertools import combinations
from collections import defaultdict

class DataLoader:
    def __init__(self, cancer_type=None, file_path="./GenesData"):
        self.cancer_type = cancer_type
        self.file_path = file_path
        
        # Only create TCGA path if cancer_type is provided
        if cancer_type:
            self.TCGA_data_path = os.path.join(file_path, self.cancer_type)
        else:
            self.TCGA_data_path = None
            
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
    
    def load_maf_mutations(self, maf_file_path):
        """
        Load genes from MAF (Mutation Annotation Format) file
        
        Args:
            maf_file_path: Path to MAF file (e.g., data_mutations.txt)
            
        Returns:
            list: List of unique mutated genes
        """
        print(f"Loading mutations from: {maf_file_path}")
        
        try:
            # Read MAF file, skip comment lines
            df = pd.read_csv(maf_file_path, sep='\t', comment='#')
            
            # Find Hugo_Symbol column (case-insensitive)
            symbol_col = None
            for col in df.columns:
                if col.lower() == 'hugo_symbol':
                    symbol_col = col
                    break
            
            if symbol_col is None:
                raise KeyError("MAF file must contain 'Hugo_Symbol' column")
            
            # Get unique mutated genes
            genes = df[symbol_col].dropna().unique().tolist()
            print(f"  Found {len(genes)} unique mutated genes")
            
            return genes
            
        except Exception as e:
            raise ValueError(f"Error loading MAF file: {e}")
    
    def load_cna_matrix(self, cna_file_path, threshold=0):
        """
        Load genes from CNA (Copy Number Alteration) matrix file
        
        Args:
            cna_file_path: Path to CNA file (e.g., data_cna.txt)
            threshold: Minimum absolute CNA value to consider (default: 0)
            
        Returns:
            list: List of unique genes with CNAs
        """
        print(f"Loading CNAs from: {cna_file_path}")
        
        try:
            # Read CNA file
            df = pd.read_csv(cna_file_path, sep='\t')
            
            # First column should be gene names
            gene_col = df.columns[0]
            
            # Find genes with any CNA above threshold
            genes_with_cna = []
            
            for idx, row in df.iterrows():
                gene = row[gene_col]
                
                # Skip if gene name is missing
                if pd.isna(gene) or gene == '':
                    continue
                
                # Check if any sample has CNA above threshold
                numeric_values = pd.to_numeric(row[1:], errors='coerce')
                if (numeric_values.abs() > threshold).any():
                    genes_with_cna.append(gene)
            
            print(f"  Found {len(genes_with_cna)} genes with CNAs")
            
            return genes_with_cna
            
        except Exception as e:
            raise ValueError(f"Error loading CNA file: {e}")
    
    def load_validation_cohort(self, mutations_file, cna_file, cna_threshold=0):
        """
        Load genes from validation cohort (mutations + CNAs)
        
        Args:
            mutations_file: Path to mutations file (MAF format)
            cna_file: Path to CNA matrix file
            cna_threshold: Minimum absolute CNA value (default: 0)
            
        Returns:
            list: Combined list of unique genes from mutations and CNAs
        """
        print("\n" + "="*60)
        print("LOADING VALIDATION COHORT DATA")
        print("="*60)
        
        # Load mutations
        mutated_genes = set(self.load_maf_mutations(mutations_file))
        
        # Load CNAs
        cna_genes = set(self.load_cna_matrix(cna_file, cna_threshold))
        
        # Combine and get unique genes
        all_genes = mutated_genes.union(cna_genes)
        
        print(f"\n{'='*60}")
        print(f"SUMMARY:")
        print(f"  Mutated genes: {len(mutated_genes)}")
        print(f"  CNA genes: {len(cna_genes)}")
        print(f"  Total unique genes: {len(all_genes)}")
        print(f"  Overlap: {len(mutated_genes.intersection(cna_genes))}")
        print(f"{'='*60}\n")
        
        return sorted(list(all_genes))
    
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
        

