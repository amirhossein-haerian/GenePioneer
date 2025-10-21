# Validation Cohort Analysis for CHD1L Biomarker

## Overview

This directory contains the validation analysis for the CHD1L biomarker discovered in the TCGA cohort. The analysis follows the exact same pipeline used in the original discovery cohort to ensure consistency and comparability.

## Pipeline Steps

### 1. Data Loading
- **Input Files:**
  - `data_mutations.txt`: Mutation data in MAF format
  - `data_cna.txt`: Copy number alteration data
  - `patients.included.txt`: Patient metadata

- **Output:**
  - List of genes with mutations
  - List of genes with CNAs
  - Combined altered gene list

### 2. CHD1L Presence Check
The script first verifies if CHD1L is present in the validation cohort:
- ✓ Present in CNAs or mutations → proceed with analysis
- ✗ Not present → analysis cannot proceed (biomarker not detectable)

### 3. Network Construction
Using the same methodology as TCGA:
- Create a protein-protein interaction network using STRING database
- Include all genes with mutations or CNAs
- Build network features for each gene

### 4. Module Detection
Using the MG (Module Growth) algorithm with CHD1L prioritization:
- **Prioritized seed:** CHD1L (instead of CHD1L + DPF2 in TCGA)
- **Module size:** 3-10 genes
- **Detection strategies:**
  - Immediate neighbors
  - High-degree neighbors  
  - Second-degree neighbors
  - Random sampling for diversity

### 5. Module Evaluation
Pathway enrichment analysis using Hallmark pathways:
- Fisher's exact test for enrichment
- FDR correction for multiple testing
- Scores calculated for module quality

### 6. Comparison with TCGA
Compare validation results with original TCGA findings:
- Pathway overlap (Jaccard similarity)
- Gene overlap between modules
- Statistical significance of similarities

## Running the Analysis

```bash
# Make sure you're in the GenePioneer directory
cd /Users/amirho3in/Documents/Stockholm\ University/Thesis/Project/GenePioneer

# Run the validation script
python validate_new_cohort.py
```

## Expected Outputs

### Files Generated

1. **`altered_genes.txt`**
   - List of all genes with alterations (mutations + CNAs)
   - Used as input for network construction

2. **`ValidationCohort_network_features.gml`**
   - Network file in GML format
   - Contains nodes (genes) and edges (interactions)

3. **`ValidationCohort_modules.json`**
   - All detected modules with scores
   - Format: `[[genes], score, quality]`

4. **`CHD1L_modules.json`**
   - Subset of modules containing CHD1L
   - These are the key modules for validation

5. **`module_evaluation_results.json`**
   - Pathway enrichment results
   - Contains significant pathways for each module
   - Includes p-values and overlap statistics

### Console Output

The script provides detailed progress information:
- Number of mutated genes found
- Number of genes with CNAs
- CHD1L presence confirmation
- Network statistics
- Module detection progress
- Pathway enrichment results
- Comparison with TCGA cohort

## Interpretation

### Success Criteria

The validation is considered successful if:

1. **CHD1L is present** in the validation cohort data
2. **Modules containing CHD1L are detected** (ideally 3-10 modules)
3. **Pathways are enriched** in CHD1L modules (p < 0.05 after FDR correction)
4. **Pathway overlap with TCGA** shows significant similarity (Jaccard > 0.3)
5. **Gene overlap with TCGA** modules shows meaningful similarity

### Possible Outcomes

#### Scenario 1: Strong Validation ✓
- CHD1L present in validation cohort
- Multiple CHD1L modules detected
- High pathway overlap with TCGA (Jaccard > 0.5)
- Similar biological processes enriched
- **Conclusion:** CHD1L biomarker validated

#### Scenario 2: Partial Validation ~
- CHD1L present but fewer modules detected
- Moderate pathway overlap (Jaccard 0.3-0.5)
- Some pathways overlap with TCGA
- **Conclusion:** CHD1L shows promise but needs larger cohort

#### Scenario 3: No Validation ✗
- CHD1L absent from validation data
- OR CHD1L present but no modules detected
- OR No pathway enrichment
- OR No overlap with TCGA
- **Conclusion:** CHD1L biomarker not validated in this cohort

## Troubleshooting

### Issue: CHD1L Not Found

**Problem:** Script reports "CHD1L NOT FOUND IN DATASET"

**Possible solutions:**
1. Check if CHD1L is spelled correctly in data files
2. Verify data files are complete and properly formatted
3. Consider lowering CNA threshold (currently set to 0)
4. Manually add CHD1L if it has clinical relevance

### Issue: No Modules Detected

**Problem:** Network created but no modules containing CHD1L

**Possible solutions:**
1. CHD1L may be isolated in the network (no interactions)
2. Adjust module size parameters (try larger max_size)
3. Check network connectivity
4. Verify STRING database interactions are loaded

### Issue: No Pathway Enrichment

**Problem:** Modules detected but no significant pathways

**Possible solutions:**
1. Check if Hallmark pathway data is loaded correctly
2. Adjust p-value threshold
3. Module genes may not be functionally coherent
4. Try larger module sizes to capture more genes

## Technical Notes

### Data Format Requirements

**Mutation File (data_mutations.txt):**
- Tab-separated format
- Must contain `Hugo_Symbol` column (case-insensitive)
- Standard MAF format preferred

**CNA File (data_cna.txt):**
- Tab-separated format
- First column: Hugo_Symbol
- Subsequent columns: Samples
- Values: -2 (deletion), -1, 0 (neutral), +1, +2 (amplification)

**Patients File (patients.included.txt):**
- Tab-separated format
- Contains SAMPLE_ID, PATIENT_ID, and Risk_group columns
- Used for metadata (not required for analysis)

### Dependencies

The analysis requires:
- genepioneer package
- NetworkBuilder, NetworkAnalysis, Evaluation classes
- STRING database access
- Hallmark pathway data (in GenesData/)

### Performance

- Small cohorts (<50 genes): ~1-2 minutes
- Medium cohorts (50-200 genes): ~5-10 minutes  
- Large cohorts (>200 genes): ~20-30 minutes

Network construction with STRING is the slowest step.

## Contact & Support

For issues or questions about the validation analysis:
1. Check this README for troubleshooting
2. Review console output for error messages
3. Verify data file formats
4. Check that all dependencies are installed

## Version History

- **v1.0** (2024): Initial validation script
  - Automated CHD1L validation pipeline
  - TCGA comparison functionality
  - Comprehensive reporting

## References

- Original TCGA analysis: See main project README
- STRING database: https://string-db.org/
- Hallmark pathways: MSigDB
