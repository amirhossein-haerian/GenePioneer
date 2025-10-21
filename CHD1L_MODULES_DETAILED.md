# CHD1L Modules - Validation Cohort Analysis

**Analysis Date:** October 21, 2025  
**Total CHD1L Modules Detected:** 6  
**Module Detection Method:** Module Growth (MG) Algorithm with CHD1L prioritization  
**Module Size Range:** 3-10 genes  

---

## Module 1 - Immune Response & Signaling
**Size:** 10 genes  
**Connectivity Score:** 1,018,500  
**Quality Score:** 814.8  

**Genes:**
- **CHD1L** (Chromatin remodeler)
- SOX9 (Transcription factor)
- SFRP1 (WNT signaling inhibitor)
- WNT5A (WNT signaling)
- TNF (Tumor necrosis factor)
- TLR4, TLR3, TLR9 (Toll-like receptors)
- IL1B (Interleukin-1 beta)
- APP (Amyloid precursor protein)

**Biological Theme:** Immune signaling, inflammation, WNT pathway regulation

---

## Module 2 - WNT Signaling & Development
**Size:** 5 genes  
**Connectivity Score:** 1,038,000  
**Quality Score:** 726.6  

**Genes:**
- **CHD1L** (Chromatin remodeler)
- SOX9 (Transcription factor)
- WNT5A (WNT signaling)
- TNF (Tumor necrosis factor)
- APP (Amyloid precursor protein)

**Biological Theme:** Developmental signaling, WNT pathway, transcriptional regulation

---

## Module 3 - Chromatin Remodeling Complex
**Size:** 3 genes  
**Connectivity Score:** 399,333  
**Quality Score:** 399.3  

**Genes:**
- **CHD1L** (Chromatin remodeler)
- RUVBL2 (AAA+ ATPase, chromatin remodeling)
- YY1 (Transcription factor)

**Biological Theme:** Chromatin remodeling, transcriptional regulation

---

## Module 4 - SWI/SNF Complex Components
**Size:** 4 genes  
**Connectivity Score:** 365,333  
**Quality Score:** 365.3  

**Genes:**
- **CHD1L** (Chromatin remodeler)
- SMARCC1 (BAF155, SWI/SNF complex)
- SS18 (SWI/SNF complex)
- ACTL6A (BAF53A, SWI/SNF complex)

**Biological Theme:** SWI/SNF chromatin remodeling complex, nucleosome reorganization

---

## Module 5 - SWI/SNF Core Module
**Size:** 3 genes  
**Connectivity Score:** 441  
**Quality Score:** 441.3  

**Genes:**
- **CHD1L** (Chromatin remodeler)
- ACTL6A (BAF53A, SWI/SNF complex)
- SMARCC1 (BAF155, SWI/SNF complex)

**Biological Theme:** Core SWI/SNF chromatin remodeling complex

---

## Module 6 - Epigenetic Regulation & Cell Cycle
**Size:** 4 genes  
**Connectivity Score:** 224,000  
**Quality Score:** 224.0  

**Genes:**
- **CHD1L** (Chromatin remodeler)
- EZH1 (Histone methyltransferase, PRC2 complex)
- HDAC5 (Histone deacetylase)
- RB1 (Retinoblastoma tumor suppressor)

**Biological Theme:** Epigenetic silencing, histone modification, cell cycle control

---

## Summary of Biological Functions

### Primary Functional Categories:

1. **Chromatin Remodeling (Modules 3, 4, 5)**
   - SWI/SNF complex components
   - Nucleosome reorganization
   - Gene accessibility regulation

2. **Transcriptional Control (Modules 2, 3, 6)**
   - Transcription factors (SOX9, YY1)
   - Histone modifications (EZH1, HDAC5)
   - Gene expression programs

3. **Developmental Signaling (Modules 1, 2)**
   - WNT pathway (WNT5A, SFRP1)
   - Cell fate determination
   - Tissue development

4. **Immune Response (Module 1)**
   - TLR signaling (TLR3, TLR4, TLR9)
   - Inflammatory response (TNF, IL1B)
   - Innate immunity

5. **Cell Cycle & Tumor Suppression (Module 6)**
   - RB1 pathway
   - Growth control
   - Cancer checkpoint regulation

### Key Partner Genes:

- **SOX9:** Appears in 2 modules - developmental transcription factor
- **SMARCC1, ACTL6A:** Appear in 2 modules - SWI/SNF complex core
- **WNT5A, TNF, APP:** Co-occur in signaling modules

### Clinical Relevance:

All modules contain genes with established roles in cancer:
- **Chromatin remodeling defects** → Aberrant gene expression
- **WNT pathway dysregulation** → Uncontrolled proliferation  
- **Immune evasion** → Tumor progression
- **Cell cycle disruption** → Genomic instability

---

## Pathway Enrichment Status

All 6 modules have been evaluated for enrichment in 50 Hallmark pathways using hypergeometric tests (p ≤ 0.05). See `evaluation_results.json` for detailed pathway associations.

---

## Data Files

- **Raw modules:** `validation_cohort/CHD1L_modules.json`
- **Pathway enrichment:** `validation_cohort/evaluation_results.json`
- **TCGA comparison:** `validation_cohort/tcga_comparison.json`
- **Network data:** `validation_cohort/ValidationCohort_network_features.gml`

---

**Note:** These modules represent functional gene communities where CHD1L plays a central role, identified through network analysis of genetic alterations in the validation cohort.
