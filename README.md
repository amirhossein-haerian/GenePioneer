First, the user needs to install the package; this can be done easily by this command on python environment: 

`pip install genepioneer`

Then you need to create a python file somewhere in your local environment, we assume that you have created a file called `gene_analyze.py`
Inside the `gene_analyze.py` you need to import the Python class that is designed for analyzing genes, this can be done like this:

`from genepioneer import GeneAnalysis`

After importing the class you need to initialize and instance of the class with two properties for the constructor. The first one is the type of the cancer, and the second one is path of the gene list that you want to analyze. The list of genes need to be a .txt file that each line of that contains one gene name and the format of the gene names should be OFFICIAL_GENE_SYMBOL.

![alt text](image.png)

For example, if we create a file called `gene_list.txt` as shown in above image. Then, we can initialize one instance from the Python class imported from the gene pioneer package by using this line of code: 

`gene_analysis = GeneAnalysis("Ovary", "./benchmark-data/gene_list.txt")`

Where "Ovary" is the name of the chosen cancer type and `./benchmark-data/gene_list.txt` is the path to the genes list file that we want to analyze.
It needs to be mentioned that, the cancer type can be one of the following names: "Adrenal", "Bladder", "Brain", "Cervix", "Colon", "Corpus uteri", "Kidney", "Liver", "Ovary", "Prostate", "Skin", "Thyroid".
After initializing one instance for the imported class from genepioner, `analyze_genes()` functionality can be called for that instance:

`gene_analysis.analyze_genes()`

By using this functionality on output.json file will be created in current path of your environment. 

The image below shows a sample of output file.

![alt text](image-1.png)
