import pandas as pd
import numpy as np
import os
from sklearn.metrics import precision_score, recall_score, f1_score

# 滑动窗口异常值去除函数（从group_adjust.py复制并更新）
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

def split_point_grouping(df, power_col='power_new', window_size=20, z_threshold=3.0):
    """
    Apply split point grouping for outlier removal
    
    Args:
        df: pandas DataFrame containing the data
        power_col: Column name for power values to process
        window_size: Size of the sliding window
        z_threshold: Z-score threshold for identifying outliers
        
    Returns:
        tuple: (detected_split_points, cleaned_series, outlier_count, outlier_mask)
    """
    # Calculate rolling median to smooth the data (using median instead of mean for better robustness)
    rolling_median = df[power_col].rolling(window=window_size, min_periods=1, center=True).median()
    
    # Calculate rolling standard deviation
    rolling_std = df[power_col].rolling(window=window_size, min_periods=1, center=True).std()
    rolling_std = rolling_std.replace(0, 1e-6)
    
    # Calculate Z-scores using median instead of mean
    z_scores = (df[power_col] - rolling_median) / rolling_std
    
    # Identify significant changes (potential split points)
    # Split points are where Z-score changes significantly
    z_score_diff = z_scores.diff().abs()
    
    # Find indices where Z-score difference is above threshold
    # Use 90th percentile as dynamic threshold
    dynamic_threshold = z_score_diff.quantile(0.9)
    detected_split_points = df.index[z_score_diff > dynamic_threshold].tolist()
    
    # Apply sliding window outlier removal
    cleaned_series, outlier_count, outlier_mask = sliding_window_outlier_removal(df[power_col], window_size, z_threshold)
    
    return detected_split_points, cleaned_series, outlier_count, outlier_mask

def evaluate_split_points(detected, ground_truth, tolerance=5):
    """
    Evaluate detected split points against ground truth
    
    Args:
        detected: List of detected split point indices
        ground_truth: List of ground truth split point indices
        tolerance: Number of indices to allow as tolerance
        
    Returns:
        dict: Evaluation metrics (precision, recall, f1)
    """
    # Create a binary array for detected and ground truth
    max_index = max(max(detected, default=0), max(ground_truth, default=0))
    
    detected_binary = [0] * (max_index + 1)
    for idx in detected:
        if idx <= max_index:
            detected_binary[idx] = 1
    
    ground_truth_binary = [0] * (max_index + 1)
    for idx in ground_truth:
        if idx <= max_index:
            ground_truth_binary[idx] = 1
    
    # Apply tolerance: expand ground truth to include nearby indices
    expanded_ground_truth = [0] * (max_index + 1)
    for idx in ground_truth:
        if idx <= max_index:
            for i in range(max(0, idx - tolerance), min(max_index + 1, idx + tolerance + 1)):
                expanded_ground_truth[i] = 1
    
    # Calculate metrics
    precision = precision_score(expanded_ground_truth, detected_binary)
    recall = recall_score(expanded_ground_truth, detected_binary)
    f1 = f1_score(expanded_ground_truth, detected_binary)
    
    # Calculate accuracy for exact matches (within tolerance)
    correct = 0
    detected_matched = set()
    
    for gt_idx in ground_truth:
        for det_idx in detected:
            if abs(det_idx - gt_idx) <= tolerance and det_idx not in detected_matched:
                correct += 1
                detected_matched.add(det_idx)
                break
    
    accuracy = correct / len(ground_truth) if ground_truth else 1.0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'accuracy_within_tolerance': accuracy,
        'detected_count': len(detected),
        'ground_truth_count': len(ground_truth)
    }

def process_test_files(data_dir, label_dir):
    """
    Process all test files and evaluate split point grouping
    
    Args:
        data_dir: Directory containing test data files
        label_dir: Directory containing test label files
    """
    # Get all data files
    data_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    
    all_metrics = []
    
    print("=== Evaluating Split Point Grouping ===")
    print(f"Found {len(data_files)} test files")
    print("=" * 60)
    
    for data_file in data_files:
        print(f"\nProcessing: {data_file}")
        
        # Load data file
        data_path = os.path.join(data_dir, data_file)
        df = pd.read_csv(data_path)
        
        # Get corresponding label file
        label_file = f"Changepoints_{data_file}"
        label_path = os.path.join(label_dir, label_file)
        
        if not os.path.exists(label_path):
            print(f"  Warning: Label file {label_file} not found")
            continue
        
        # Load label file
        label_df = pd.read_csv(label_path)
        
        # Extract ground truth split point indices
        ground_truth_indices = label_df['changepoint_index'].tolist()
        
        # Apply split point grouping
        detected_indices, cleaned_power, outlier_count, outlier_mask = split_point_grouping(df)
        
        # Evaluate
        metrics = evaluate_split_points(detected_indices, ground_truth_indices)
        
        # Print results for this file
        print(f"  Ground truth split points: {ground_truth_indices}")
        print(f"  Detected split points: {detected_indices}")
        print(f"  Metrics: Precision={metrics['precision']:.4f}, Recall={metrics['recall']:.4f}, F1={metrics['f1']:.4f}, Accuracy={metrics['accuracy_within_tolerance']:.4f}")
        print(f"  Outliers detected during processing: {outlier_count}")
        
        all_metrics.append(metrics)
    
    # Calculate average metrics
    if all_metrics:
        avg_metrics = {
            'precision': np.mean([m['precision'] for m in all_metrics]),
            'recall': np.mean([m['recall'] for m in all_metrics]),
            'f1': np.mean([m['f1'] for m in all_metrics]),
            'accuracy_within_tolerance': np.mean([m['accuracy_within_tolerance'] for m in all_metrics])
        }
        
        print("\n" + "=" * 60)
        print("=== Average Evaluation Results ===")
        print(f"Average Precision: {avg_metrics['precision']:.4f}")
        print(f"Average Recall: {avg_metrics['recall']:.4f}")
        print(f"Average F1 Score: {avg_metrics['f1']:.4f}")
        print(f"Average Accuracy (within tolerance): {avg_metrics['accuracy_within_tolerance']:.4f}")
        print("=" * 60)
    
    return all_metrics

def visualize_results(data_file, data_dir, label_dir):
    """
    Visualize the results for a specific file
    
    Args:
        data_file: Name of the data file to visualize
        data_dir: Directory containing test data files
        label_dir: Directory containing test label files
    """
    import matplotlib.pyplot as plt
    
    # Load data
    data_path = os.path.join(data_dir, data_file)
    df = pd.read_csv(data_path)
    
    # Load labels
    label_file = f"Changepoints_{data_file}"
    label_path = os.path.join(label_dir, label_file)
    label_df = pd.read_csv(label_path)
    ground_truth_indices = label_df['changepoint_index'].tolist()
    
    # Apply split point grouping
    detected_indices, cleaned_power, outlier_count, outlier_mask = split_point_grouping(df)
    
    # Create plot
    plt.figure(figsize=(15, 12))
    
    # Get the power column used for analysis
    if 'power_new' in df.columns:
        original_col = 'power_new'
    elif 'power' in df.columns:
        original_col = 'power'
    elif 'power_origin' in df.columns:
        original_col = 'power_origin'
    
    # Plot original power data
    plt.subplot(3, 1, 1)
    plt.plot(df[original_col], label=f'Original {original_col}', alpha=0.7)
    plt.title(f"Original vs Cleaned Power for {data_file}")
    plt.ylabel('Power')
    plt.legend()
    
    # Mark ground truth split points
    for idx in ground_truth_indices:
        plt.axvline(x=idx, color='green', linestyle='--', label='Ground Truth Split Point' if idx == ground_truth_indices[0] else "")
    
    # Mark detected split points
    for idx in detected_indices:
        plt.axvline(x=idx, color='red', linestyle=':', label='Detected Split Point' if idx == detected_indices[0] else "")
    
    plt.legend()
    
    # Plot cleaned power
    plt.subplot(3, 1, 2)
    plt.plot(cleaned_power, label='Cleaned Power', color='blue', alpha=0.7)
    plt.ylabel('Power')
    plt.legend()
    
    # Mark ground truth split points
    for idx in ground_truth_indices:
        plt.axvline(x=idx, color='green', linestyle='--', label='Ground Truth Split Point' if idx == ground_truth_indices[0] else "")
    
    # Mark detected split points
    for idx in detected_indices:
        plt.axvline(x=idx, color='red', linestyle=':', label='Detected Split Point' if idx == detected_indices[0] else "")
    
    plt.legend()
    
    # Plot outlier mask
    plt.subplot(3, 1, 3)
    plt.plot(outlier_mask.astype(int), label='Outlier Mask', color='orange', alpha=0.7)
    plt.xlabel('Index')
    plt.ylabel('Is Outlier (1=Yes, 0=No)')
    plt.ylim(-0.1, 1.1)
    plt.legend()
    
    # Mark ground truth split points
    for idx in ground_truth_indices:
        plt.axvline(x=idx, color='green', linestyle='--', label='Ground Truth Split Point' if idx == ground_truth_indices[0] else "")
    
    # Mark detected split points
    for idx in detected_indices:
        plt.axvline(x=idx, color='red', linestyle=':', label='Detected Split Point' if idx == detected_indices[0] else "")
    
    plt.legend()
    plt.tight_layout()
    
    # Save plot
    plot_filename = f"split_point_analysis_{os.path.splitext(data_file)[0]}.png"
    plt.savefig(plot_filename)
    print(f"\nPlot saved as: {plot_filename}")
    
    return plt

def main():
    # Define test directories
    DATA_DIR = r'F:\B__ProfessionProject\NILM\Clasp\mean_reversion\project\test\data'
    LABEL_DIR = r'F:\B__ProfessionProject\NILM\Clasp\mean_reversion\project\test\label'
    
    # Process all test files and evaluate
    all_metrics = process_test_files(DATA_DIR, LABEL_DIR)
    
    # Visualize results for the first file
    data_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.csv')]
    if data_files:
        visualize_results(data_files[0], DATA_DIR, LABEL_DIR)
    
    print("\n=== Test Completed ===")

if __name__ == "__main__":
    main()
