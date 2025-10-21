#!/usr/bin/env python3
"""
Quick verification script to check if validation setup is ready
"""

import os
import sys

def check_file(path, description):
    """Check if a file exists"""
    if os.path.exists(path):
        size = os.path.getsize(path)
        print(f"✓ {description}: {path} ({size:,} bytes)")
        return True
    else:
        print(f"✗ {description}: {path} NOT FOUND")
        return False

def check_directory(path, description):
    """Check if a directory exists"""
    if os.path.exists(path) and os.path.isdir(path):
        print(f"✓ {description}: {path}")
        return True
    else:
        print(f"✗ {description}: {path} NOT FOUND")
        return False

def check_chd1l_in_data():
    """Check if CHD1L is present in the data files"""
    print("\nChecking CHD1L presence...")
    
    # Check CNA file
    cna_file = "./data_cna.txt"
    if os.path.exists(cna_file):
        with open(cna_file, 'r') as f:
            content = f.read()
            if 'CHD1L' in content:
                print("✓ CHD1L found in data_cna.txt")
                return True
            else:
                print("✗ CHD1L NOT found in data_cna.txt")
    
    # Check mutations file
    mut_file = "./data_mutations.txt"
    if os.path.exists(mut_file):
        with open(mut_file, 'r') as f:
            content = f.read()
            if 'CHD1L' in content:
                print("✓ CHD1L found in data_mutations.txt")
                return True
            else:
                print("✗ CHD1L NOT found in data_mutations.txt")
    
    return False

def main():
    print("="*70)
    print("VALIDATION SETUP VERIFICATION")
    print("="*70)
    
    all_ok = True
    
    # Check required data files
    print("\n[1] Checking Input Data Files...")
    all_ok &= check_file("./data_mutations.txt", "Mutation data")
    all_ok &= check_file("./data_cna.txt", "CNA data")
    all_ok &= check_file("./patients.included.txt", "Patient metadata")
    
    # Check validation script
    print("\n[2] Checking Validation Script...")
    all_ok &= check_file("./validate_new_cohort.py", "Validation script")
    
    # Check genepioneer package
    print("\n[3] Checking GenePioneer Package...")
    all_ok &= check_file("./genepioneer/network_analysis.py", "Network analysis module")
    all_ok &= check_file("./genepioneer/network_builder.py", "Network builder module")
    all_ok &= check_file("./genepioneer/evaluation.py", "Evaluation module")
    
    # Check data directory
    print("\n[4] Checking Data Directory...")
    all_ok &= check_directory("./GenesData", "Gene data directory")
    
    # Check TCGA results for comparison
    print("\n[5] Checking TCGA Results (for comparison)...")
    check_file("./evaluated_modules_result.json", "TCGA evaluation results")
    
    # Check output directory
    print("\n[6] Checking Output Directory...")
    if not os.path.exists("./validation_cohort"):
        os.makedirs("./validation_cohort")
        print("✓ Created validation_cohort directory")
    else:
        print("✓ validation_cohort directory exists")
    
    # Check CHD1L presence
    print("\n[7] Checking CHD1L Presence...")
    chd1l_present = check_chd1l_in_data()
    
    # Final summary
    print("\n" + "="*70)
    print("VERIFICATION SUMMARY")
    print("="*70)
    
    if all_ok and chd1l_present:
        print("\n✓ ALL CHECKS PASSED")
        print("\nYou are ready to run the validation analysis!")
        print("\nRun: python validate_new_cohort.py")
    elif chd1l_present:
        print("\n⚠️  SOME CHECKS FAILED")
        print("\nCHD1L is present but some files are missing.")
        print("Review the checks above and ensure all files are available.")
    else:
        print("\n✗ CRITICAL: CHD1L NOT FOUND")
        print("\nCHD1L was not found in the data files.")
        print("The validation cannot proceed without CHD1L in the dataset.")
        print("\nOptions:")
        print("1. Verify CHD1L spelling in data files")
        print("2. Check if data files are complete")
        print("3. Consider alternative approaches")
    
    print("\n" + "="*70)

if __name__ == "__main__":
    main()
