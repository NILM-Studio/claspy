import os
import glob
import pandas as pd
import re

import random

# ================= Configuration =================
INPUT_DIR = r"f:\B__ProfessionProject\NILM\Clasp\mean_reversion(out-of-date)\select\data"
OUTPUT_DIR = r"f:\B__ProfessionProject\NILM\Clasp\mean_reversion(out-of-date)\select\output"

# Pre-set start and end dates (YYYYMMDD format)
START_DATE = "20140101"
END_DATE = "20141231"

# Sampling configuration
SAMPLE_SIZE = 7000  # Number of files to sample. Set to None to select all matching files.
# =================================================

def extract_dates_from_filename(filename):
    """
    Extracts start and end dates from filenames like:
    Device_YYYYMMDD_HHMMSS_YYYYMMDD_HHMMSS_Duration.csv
    Returns (start_date_str, end_date_str) or None if not matching.
    """
    # Pattern to match: Any_YYYYMMDD_HHMMSS_YYYYMMDD_HHMMSS_*.csv
    # We focus on the date parts.
    pattern = r"_(\d{8})_\d{6}_(\d{8})_\d{6}_"
    match = re.search(pattern, filename)
    if match:
        return match.group(1), match.group(2)
    return None

def main():
    # Ensure output directory exists
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created output directory: {OUTPUT_DIR}")

    files = glob.glob(os.path.join(INPUT_DIR, "*.csv"))
    print(f"Found {len(files)} files in {INPUT_DIR}")
    
    # Step 1: Filter files based on date range
    valid_files = []
    for file_path in files:
        filename = os.path.basename(file_path)
        dates = extract_dates_from_filename(filename)
        
        if dates:
            file_start_date, file_end_date = dates
            if START_DATE <= file_start_date <= END_DATE:
                valid_files.append(file_path)
        else:
            print(f"Skipped (format mismatch): {filename}")

    print(f"Found {len(valid_files)} files matching the date range [{START_DATE}, {END_DATE}]")

    # Step 2: Sampling
    if SAMPLE_SIZE is not None and len(valid_files) > SAMPLE_SIZE:
        print(f"Sampling {SAMPLE_SIZE} files from {len(valid_files)} candidates...")
        selected_files = random.sample(valid_files, SAMPLE_SIZE)
    else:
        print("Selecting all matching files (count <= sample size or sample size not set).")
        selected_files = valid_files

    # Step 3: Save selected files
    count_selected = 0
    for file_path in selected_files:
        filename = os.path.basename(file_path)
        try:
            # Read the CSV file
            df = pd.read_csv(file_path)
            
            # Save to output directory
            output_path = os.path.join(OUTPUT_DIR, filename)
            df.to_csv(output_path, index=False)
            
            # print(f"Selected and saved: {filename}")
            count_selected += 1
        except Exception as e:
            print(f"Error reading/saving {filename}: {e}")

    print(f"Process completed. Selected {count_selected} files saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
