import os
import sys

# ============== Configuration ==============
project_dir = "dishwasher"
# ===========================================


# Ensure we can import from the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

import silhouette_distribution
import group_adjust
import tsd
import shutil

def main():
    base_dir = current_dir + "/project"
    
    # 1. Silhouette Analysis
    print("=== Step 1: Silhouette Score Distribution Analysis ===")
    data_input_dir = os.path.join(base_dir, project_dir, 'related', 'data')
    plot_output_dir = os.path.join(base_dir, project_dir, 'related', 'plot')
    score_output_dir = os.path.join(base_dir, project_dir, 'related', 'score')
    
    # Returns the split point (threshold)
    split_point = silhouette_distribution.analyze_distribution(
        data_input_dir, 
        plot_output_dir, 
        score_output_dir
    )
    print(f"Obtained split point: {split_point}")



    # 2. Group Adjustment
    print("\n=== Step 2: Group Adjustment ===")
    adjusted_data_dir = os.path.join(base_dir, project_dir, 'data')

    group_adjust.adjust_groups(
            data_input_dir, 
            adjusted_data_dir, 
            split_point
        )

    # 3. Time Series Segmentation
    print("\n=== Step 3: Time Series Segmentation ===")
    label_output_dir = os.path.join(base_dir, project_dir, 'label')
    score_file_path = os.path.join(score_output_dir, 'file_scores.csv')
    
    tsd.run_segmentation(
        adjusted_data_dir, 
        label_output_dir, 
        score_file_path
    )
    
    print("\n=== Workflow Completed ===")

if __name__ == "__main__":
    main()
