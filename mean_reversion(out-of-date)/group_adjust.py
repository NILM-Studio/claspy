import os
import glob
import gc
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def sliding_window_outlier_removal(series, window_size=20, z_threshold=3.0, interpolation_method='linear'):
    """
    Apply sliding window outlier removal using Z-score method with median
    
    Args:
        series: pandas Series containing the data to process
        window_size: Size of the sliding window
        z_threshold: Z-score threshold for identifying outliers
        interpolation_method: Method for interpolating outliers
            Options: 'linear', 'polynomial', 'spline', 'nearest', 'zero'
        
    Returns:
        tuple: (cleaned_series, outlier_count, outlier_mask)
            cleaned_series: pandas Series with outliers removed (replaced with interpolated values)
            outlier_count: Number of outliers detected and removed
            outlier_mask: Boolean Series indicating which points were interpolated
    """
    # Create a copy of the series
    cleaned_series = series.copy()
    
    # Calculate rolling median and standard deviation (using median instead of mean)
    rolling_median = series.rolling(window=window_size, min_periods=1, center=True).median()
    rolling_std = series.rolling(window=window_size, min_periods=1, center=True).std()
    
    # Handle cases where std is 0 (all values in window are the same)
    rolling_std = rolling_std.replace(0, 1e-6)
    
    # Calculate Z-scores using median instead of mean
    z_scores = (series - rolling_median) / rolling_std
    
    # Identify outliers
    outlier_mask = abs(z_scores) > z_threshold
    outlier_count = outlier_mask.sum()
    
    if outlier_count > 0:
        # Create a temporary series where outliers are set to NaN
        temp_series = series.copy()
        temp_series[outlier_mask] = np.nan
        
        # Apply interpolation to fill NaN values (outliers)
        if interpolation_method == 'polynomial':
            cleaned_series = temp_series.interpolate(method='polynomial', order=2)
        elif interpolation_method == 'spline':
            cleaned_series = temp_series.interpolate(method='spline', order=3)
        else:
            # Use other methods like linear, nearest, zero
            cleaned_series = temp_series.interpolate(method=interpolation_method)
    
    return cleaned_series, outlier_count, outlier_mask

def process_file(file_path, target_dir, threshold):
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return

    if 'power' not in df.columns:
        print(f"Column 'power' not found in {file_path}")
        return

    data = df['power'].values.reshape(-1, 1)
    
    # Check if we have enough data points
    if len(data) < 2:
        return

    best_score = -1 
    best_k = -1
    best_labels = None
    
    # Iterate through possible k values
    max_k = min(10, len(data))
    
    for k in range(2, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(data)
        
        # Calculate Silhouette Score
        if len(data) > 10000:
             sil_score = silhouette_score(data, labels, sample_size=10000)
        else:
             sil_score = silhouette_score(data, labels)

        if sil_score > best_score:
            best_score = sil_score
            best_k = k
            best_labels = labels
            
    # print(f"File: {os.path.basename(file_path)}, Best Score: {best_score:.4f}, k={best_k}")

    # Initialize interpolation flag column
    df['is_interpolated'] = False
    total_outliers = 0
    
    # Rename original power to power_origin
    df['power_origin'] = df['power'].copy()
    
    if threshold is not None and best_score > threshold:
        # Add labels to dataframe
        df['cluster_label'] = best_labels
        
        # Calculate mean power for each cluster using original power
        cluster_means = {}
        for label in range(best_k):
            cluster_data = df[df['cluster_label'] == label]['power_origin']
            cluster_means[label] = cluster_data.mean()
            
        # Find the cluster with the minimum mean power
        min_mean_label = min(cluster_means, key=cluster_means.get)
        min_mean_val = cluster_means[min_mean_label]
        
        # Apply sliding window outlier removal to original power for each cluster group
        # This will clean the data before calculating power_new
        df['power'] = df['power_origin'].copy()
        
        # Process each cluster separately
        for label in range(best_k):
            # Get the mask for this cluster
            cluster_mask = df['cluster_label'] == label
            
            # Apply outlier removal to this cluster's original power
            cleaned_series, outlier_count, cluster_outlier_mask = sliding_window_outlier_removal(
                df.loc[cluster_mask, 'power_origin'],
                interpolation_method='linear'
            )
            
            # Update power with cleaned values
            df.loc[cluster_mask, 'power'] = cleaned_series
            
            # Update interpolation flag
            df.loc[cluster_mask, 'is_interpolated'] = cluster_outlier_mask.values
            
            total_outliers += outlier_count
        
        # Calculate adjusted power using cleaned power
        def adjust_power(row):
            cluster_mean = cluster_means[row['cluster_label']]
            diff = cluster_mean - min_mean_val
            return row['power'] - diff
            
        df['power_new'] = df.apply(adjust_power, axis=1)
    
    else:
        df['cluster_label'] = -1
        
        # Apply sliding window outlier removal directly to original power
        # This will clean the data before calculating power_new
        df['power'] = df['power_origin'].copy()
        
        # Apply outlier removal to the entire dataset
        cleaned_series, total_outliers, outlier_mask = sliding_window_outlier_removal(
            df['power_origin'],
            interpolation_method='linear'
        )
        
        # Update power with cleaned values
        df['power'] = cleaned_series
        
        # Update interpolation flag
        df['is_interpolated'] = outlier_mask
        
        # Calculate power_new using cleaned power
        # In this case, since there's no clustering, power_new is just the cleaned power
        df['power_new'] = df['power']
    
    # Print the number of outliers removed
    filename = os.path.basename(file_path)
    print(f"File: {filename}, Outliers removed: {total_outliers}")
    
    # Save to target directory
    filename = os.path.basename(file_path)
    output_path = os.path.join(target_dir, filename)
    df.to_csv(output_path, index=False)
    # print(f"  Saved adjusted file to {output_path}")

def adjust_groups(source_dir, target_dir, threshold):
    if not os.path.exists(target_dir):
        try:
            os.makedirs(target_dir)
        except OSError:
            pass

    files = glob.glob(os.path.join(source_dir, '*.csv'))
    if not files:
        print(f"No CSV files found in {source_dir}")
        return

    print(f"Found {len(files)} files in {source_dir}")
    
    for i, file_path in enumerate(files):
        process_file(file_path, target_dir, threshold)
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(files)} files...")
            gc.collect()

if __name__ == "__main__":
    # Test with project test data
    SOURCE_DIR = r'f:\B__ProfessionProject\NILM\Clasp\mean_reversion\project\test\data'
    TARGET_DIR = r'f:\B__ProfessionProject\NILM\Clasp\mean_reversion\project\test\output'
    adjust_groups(SOURCE_DIR, TARGET_DIR, 0.82)
