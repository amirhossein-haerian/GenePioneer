#!/usr/bin/env python3

import sys
import os
sys.path.append('/Users/amirho3in/Documents/Stockholm University/Thesis/Project/GenePioneer')

from genepioneer.evaluation import Evaluation

def test_evaluation_network_size():
    try:
        # Create an Evaluation instance
        eval_instance = Evaluation(data_path=".")
        
        # Test the network size loading
        network_size = eval_instance.get_network_size()
        print(f"Network size from evaluation class: {network_size}")
        
        # Test manual pathway enrichment with actual network size
        test_genes = ['CHD1L', 'GATA3', 'PTEN']
        print(f"Testing enrichment for genes: {test_genes}")
        
        enrichment_results = eval_instance.manual_pathway_enrichment(test_genes)
        print(f"Found {len(enrichment_results)} enrichment results")
        
        if enrichment_results:
            print("Top 3 results:")
            for i, result in enumerate(enrichment_results[:3]):
                print(f"  {i+1}. {result['pathway']}: p={result['p_value']:.6f}, enrichment={result['enrichment_ratio']:.2f}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_evaluation_network_size()
