import os
import pandas as pd
import numpy as np

def remove_outliers_zscore(df, split_point=None, value_col='power'):
    """Remove outliers from dataframe using Z-score method.
    
    Args:
        df: pandas DataFrame containing the data
        split_point: Threshold value to group data (None if no grouping)
        value_col: Column name containing the values to process
        
    Returns:
        Filtered pandas DataFrame with outliers removed
    """
    if split_point is None:
        # No split point, apply Z-score to entire column
        z_scores = np.abs((df[value_col] - df[value_col].mean()) / df[value_col].std())
        return df[z_scores < 3]  # Keep rows with Z-score < 3
    else:
        # Group by split_point threshold and apply Z-score within each group
        def filter_group(group):
            z_scores = np.abs((group[value_col] - group[value_col].mean()) / group[value_col].std())
            return group[z_scores < 3]
        
        # Create a copy of the dataframe to avoid modifying the original
        df_copy = df.copy()
        
        # Split the dataframe into two parts based on split_point
        # Part 1: values greater than split_point
        greater_than_split = df_copy[df_copy[value_col] > split_point]
        # Part 2: values less than or equal to split_point
        less_than_or_equal_split = df_copy[df_copy[value_col] <= split_point]
        
        # Process each part separately with Z-score method
        filtered_parts = []
        
        # Process part 1: greater than split_point
        if not greater_than_split.empty:
            z_scores_greater = np.abs((greater_than_split[value_col] - greater_than_split[value_col].mean()) / greater_than_split[value_col].std())
            filtered_greater = greater_than_split[z_scores_greater < 3]
            filtered_parts.append(filtered_greater)
        
        # Process part 2: less than or equal to split_point
        if not less_than_or_equal_split.empty:
            z_scores_less = np.abs((less_than_or_equal_split[value_col] - less_than_or_equal_split[value_col].mean()) / less_than_or_equal_split[value_col].std())
            filtered_less = less_than_or_equal_split[z_scores_less < 3]
            filtered_parts.append(filtered_less)
        
        # Combine the filtered parts
        filtered_df = pd.concat(filtered_parts, ignore_index=True)
        
        return filtered_df

def run_outlier_removal(input_data_dir, output_data_dir, split_point):
    """Run outlier removal on all CSV files in the specified input directory.
    
    Args:
        input_data_dir: Directory containing the input data files
        output_data_dir: Directory to save the filtered data files
        split_point: Split point threshold (None if no split)
    """
    # Ensure output directory exists
    if not os.path.exists(output_data_dir):
        os.makedirs(output_data_dir)
    
    # Process each file in input_data_dir
    for filename in os.listdir(input_data_dir):
        if filename.endswith(".csv"):
            input_file_path = os.path.join(input_data_dir, filename)
            output_file_path = os.path.join(output_data_dir, filename)
            try:
                df = pd.read_csv(input_file_path)
                
                if 'power' not in df.columns:
                    print(f"  Warning: 'power' column not found in {filename}, skipping")
                    continue
                
                # Remove outliers using Z-score method with split_point grouping
                filtered_df = remove_outliers_zscore(df, split_point=split_point)
                
                # Save the filtered data
                filtered_df.to_csv(output_file_path, index=False)
                print(f"  Processed {filename}: removed outliers, {len(df) - len(filtered_df)} points removed")
            except Exception as e:
                print(f"  Error processing {filename}: {e}")