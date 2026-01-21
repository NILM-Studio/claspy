import pandas as pd
import numpy as np
import os
from group_adjust import sliding_window_outlier_removal

def test_outlier_removal():
    """
    Test the sliding window outlier removal with interpolation
    """
    # Create test data with outliers
    np.random.seed(42)
    
    # Create normal data
    normal_data = np.random.normal(100, 10, 100)
    
    # Insert outliers
    outlier_indices = [10, 25, 50, 75, 90]
    outliers = [300, -50, 200, -20, 250]
    
    test_data = normal_data.copy()
    for idx, outlier in zip(outlier_indices, outliers):
        test_data[idx] = outlier
    
    # Create pandas Series
    series = pd.Series(test_data)
    
    print("=== Testing Sliding Window Outlier Removal with Interpolation ===")
    print(f"Original data has {len(outlier_indices)} known outliers")
    print(f"Outlier indices: {outlier_indices}")
    print(f"Outlier values: {outliers}")
    print("=" * 60)
    
    # Test with different interpolation methods
    interpolation_methods = ['linear', 'polynomial', 'spline', 'nearest', 'zero']
    
    for method in interpolation_methods:
        print(f"\nTesting with {method} interpolation:")
        cleaned_series, outlier_count, outlier_mask = sliding_window_outlier_removal(series, interpolation_method=method)
        print(f"  Detected outliers: {outlier_count}")
        print(f"  Outlier mask shape: {outlier_mask.shape}")
        
        # Check if known outliers were detected
        detected = 0
        for idx in outlier_indices:
            if not np.isclose(cleaned_series[idx], series[idx], rtol=1e-5):
                detected += 1
        print(f"  Known outliers detected: {detected}/{len(outlier_indices)}")
        
        # Check outlier mask
        mask_detected = sum(outlier_mask[outlier_indices])
        print(f"  Outlier mask correctly identifies: {mask_detected}/{len(outlier_indices)} known outliers")
        
        # Show some values around outliers
        print(f"  Values around outlier at index 10:")
        print(f"    Original: {series[8:13].values}")
        print(f"    Cleaned:  {cleaned_series[8:13].values}")
        print(f"    Mask:     {outlier_mask[8:13].values}")
    
    print("\n" + "=" * 60)
    print("=== Test Completed ===")
    
    return series, cleaned_series

def test_real_data():
    """
    Test with real data from the project
    """
    # Use the first test file
    data_file = r'F:\B__ProfessionProject\NILM\Clasp\mean_reversion\project\test\data\Washing_Machine_20121110_182407_20121110_191850_463s.csv'
    
    if os.path.exists(data_file):
        print(f"\n=== Testing with Real Data: {os.path.basename(data_file)} ===")
        df = pd.read_csv(data_file)
        
        # Determine which column to use for testing
        # If power_origin exists (output from group_adjust.py), use it
        # Otherwise, use the original power column
        if 'power_origin' in df.columns:
            test_column = 'power_origin'
            print(f"Using {test_column} column for outlier detection")
        else:
            test_column = 'power'
            print(f"Using {test_column} column for outlier detection")
        
        # Test on the selected column
        series = df[test_column]
        cleaned_series, outlier_count, outlier_mask = sliding_window_outlier_removal(series)
        
        print(f"Original data points: {len(series)}")
        print(f"Outliers removed: {outlier_count}")
        print(f"Percentage of outliers: {outlier_count/len(series)*100:.2f}%")
        
        # Add interpolation flag to dataframe for testing
        df['is_interpolated'] = outlier_mask
        print(f"DataFrame now has 'is_interpolated' column")
        print(f"Number of interpolated points in DataFrame: {df['is_interpolated'].sum()}")
        
        # Show sample of dataframe with interpolation flag
        print(f"\nSample DataFrame with interpolation flag:")
        
        # Select columns to display based on available columns
        display_columns = [test_column, 'is_interpolated']
        if 'power' in df.columns and 'power' != test_column:
            display_columns.insert(1, 'power')
        if 'power_new' in df.columns:
            display_columns.append('power_new')
        
        sample = df[display_columns].iloc[45:55]
        print(sample.to_string(index=True))
        
        return series, cleaned_series, df
    else:
        print(f"Real data file not found: {data_file}")
        return None, None, None

if __name__ == "__main__":
    test_outlier_removal()
    test_real_data()
