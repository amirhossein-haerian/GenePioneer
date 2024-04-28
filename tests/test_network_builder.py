from genepioneer import NetworkBuilder

from concurrent.futures import ProcessPoolExecutor, as_completed
import pandas as pd

# "Bladder", "Brain", "Cervix", "Colon", "Corpus uteri", "Kidney", "Liver", "Ovary", "Prostate", "Skin", "Thyroid"
cancers = ["Adrenal"]

def process_cancer(cancer):
    print("Working on: ",cancer)
    network_builder = NetworkBuilder(cancer)
    network_builder.build_network()

    features = network_builder.calculate_all_features()
    network_builder.save_features_to_csv(features, f"{cancer}_network_features.csv")
    print("after csv")

# Using ProcessPoolExecutor to run tasks in parallel
def main():
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_cancer, cancer) for cancer in cancers]
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