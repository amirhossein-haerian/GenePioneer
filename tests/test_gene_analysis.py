from genepioneer import GeneAnalysis

from concurrent.futures import ProcessPoolExecutor, as_completed
import pandas as pd
import json


# "Adrenal", "Bladder", "Brain", "Cervix", "Colon", "Corpus uteri", "Kidney", "Liver", "Ovary", "Prostate", "Skin", "Thyroid"
cancers = ["Prostate"]

gene_analysis = GeneAnalysis("Prostate", "../genepioneer/Data/")
gene_analysis.analyze_genes()