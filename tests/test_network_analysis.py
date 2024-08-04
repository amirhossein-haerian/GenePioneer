from genepioneer import NetworkAnalysis

from concurrent.futures import ProcessPoolExecutor, as_completed
import pandas as pd
import json


# "Adrenal", "Bladder", "Brain", "Cervix", "Colon", "Corpus uteri", "Kidney", "Liver", "Ovary", "Prostate", "Skin", "Thyroid"
cancers = ["Brain", "Colon", "Skin"]

def analysis(cancer):
    print("Working on: ",cancer)
    network_analysis = NetworkAnalysis(cancer)
    
    # m = network_analysis.MG_algorithm()
    
    # with open(f'{cancer}2.json', 'w') as file:
    #     json.dump(m, file, indent=4)
    m = network_analysis.new2_algorithm()
    # m = network_analysis.new_MG_algorithm()
    with open(f'{cancer}8.json', 'w') as file:
        json.dump(m, file, indent=4)

# Using ProcessPoolExecutor to run tasks in parallel
def main():
    with ProcessPoolExecutor(max_workers=7) as executor:
        futures = [executor.submit(analysis, cancer) for cancer in cancers]
        for future in futures:
            # Handle results or exceptions if necessary
            try:
                result = future.result()
                print("try")
                # Process result if necessary
            except Exception as exc:
                print("catch")
                print(f'Generated an exception: {exc}')

if __name__ == '__main__':
    main()