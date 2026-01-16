import pandas as pd
import numpy as np
import pywt
import matplotlib.pyplot as plt
import os
import sys

# Add project root to sys.path if needed
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(script_dir, ".."))

from claspy.segmentation import BinaryClaSPSegmentation

def sliding_window_outlier_removal(series, window_size=20, z_threshold=3.0, interpolation_method='linear'):
    """
    Perform sliding window outlier removal using median-based Z-score detection.
    
    Args:
        series (array-like): Input time series data
        window_size (int): Size of the sliding window
        z_threshold (float): Z-score threshold for outlier detection
        interpolation_method (str): Interpolation method for replacing outliers
            Options: 'linear', 'polynomial', 'spline', 'nearest', 'zero'
    
    Returns:
        tuple: (cleaned_series, outlier_count, outlier_mask)
    """
    import pandas as pd
    import numpy as np
    
    # Convert series to pandas Series for rolling window operations
    s = pd.Series(series)
    
    # Calculate rolling median and standard deviation
    rolling_median = s.rolling(window=window_size, center=True, min_periods=1).median()
    rolling_std = s.rolling(window=window_size, center=True, min_periods=1).std()
    
    # Calculate Z-scores (using median instead of mean for robustness)
    z_scores = (s - rolling_median) / rolling_std
    
    # Handle NaN values in Z-scores (at the beginning and end)
    z_scores = z_scores.fillna(0)
    
    # Identify outliers
    outlier_mask = np.abs(z_scores) > z_threshold
    outlier_count = outlier_mask.sum()
    
    # Create a copy of the original series for cleaning
    cleaned_series = s.copy()
    
    # Interpolate outliers
    if outlier_count > 0:
        # Create a mask for valid values (non-outliers)
        valid_mask = ~outlier_mask
        
        # Interpolate the outliers based on valid values
        cleaned_series[outlier_mask] = np.nan
        
        # Use the specified interpolation method
        cleaned_series = cleaned_series.interpolate(method=interpolation_method, limit_direction='both')
        
        # If there are still NaN values (at the very beginning or end), use bfill and ffill
        cleaned_series = cleaned_series.bfill().ffill()
    
    # Convert back to numpy array
    cleaned_series = cleaned_series.values
    
    return cleaned_series, outlier_count, outlier_mask

def get_segmentation_points(time_series):
    """Segmentation logic adapted from tsd.py"""
    try:
        clasp = BinaryClaSPSegmentation(
            n_segments="learn",
            window_size="suss",
            validation="score_threshold",
            threshold=0.001,
        )
        clasp.fit_predict(time_series)
        return clasp.change_points
    except Exception as e:
        print(f"Segmentation error: {e}")
        return []

def synthesize_changepoints(orig_cp, low_cp, high_cp):
    """
    Synthesizes changepoints by selecting a reference from low/high freq 
    and mapping others to minimize distance sum.
    Only considers Low-Freq and High-Freq.
    """
    if len(low_cp) == 0 and len(high_cp) == 0:
        return [], "None"

    # 1. Determine reference (one with most points between low and high)
    if len(low_cp) >= len(high_cp):
        ref_cp = np.sort(low_cp)
        others = [np.sort(high_cp)]
        ref_name = "Low-Freq"
    else:
        ref_cp = np.sort(high_cp)
        others = [np.sort(low_cp)]
        ref_name = "High-Freq"

    if len(ref_cp) == 0:
        return [], "None"

    # 2. Map other changepoints to reference points to minimize distance sum
    groups = {i: [ref_val] for i, ref_val in enumerate(ref_cp)}
    
    for other_list in others:
        for p in other_list:
            # Find index of closest ref point
            closest_idx = np.argmin(np.abs(ref_cp - p))
            groups[closest_idx].append(p)
            
    # 3. Calculate synthesized changepoints (mean of each group)
    synthesized_cp = []
    for i in sorted(groups.keys()):
        group_mean = np.mean(groups[i])
        synthesized_cp.append(group_mean)
        
    return sorted(synthesized_cp), ref_name

def run_wavelet_analysis(signal, wavelet, orig_cp):
    """Performs wavelet separation and segmentation for a given wavelet."""
    # 1. Wavelet Transform (level=2)
    level = 2
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    cA2, cD2, cD1 = coeffs
    
    # 2. Reconstruct components
    zeros_cD2 = np.zeros_like(cD2)
    zeros_cD1 = np.zeros_like(cD1)
    zeros_cA2 = np.zeros_like(cA2)
    
    low_freq_signal = pywt.waverec([cA2, zeros_cD2, zeros_cD1], wavelet)
    high_freq_combined = pywt.waverec([zeros_cA2, cD2, cD1], wavelet)
    
    # Truncate to match original signal length
    low_freq_signal = low_freq_signal[:len(signal)]
    high_freq_combined = high_freq_combined[:len(signal)]
    
    # 3. Segmentation
    low_cp = get_segmentation_points(low_freq_signal)
    high_cp = get_segmentation_points(high_freq_combined)
    
    # 4. Synthesis
    synthesized_cp, ref_name = synthesize_changepoints(orig_cp, low_cp, high_cp)
    
    return {
        'wavelet': wavelet,
        'low_freq_signal': low_freq_signal,
        'high_freq_combined': high_freq_combined,
        'low_cp': low_cp,
        'high_cp': high_cp,
        'synthesized_cp': synthesized_cp,
        'ref_name': ref_name,
        'num_low_cp': len(low_cp),
        'num_high_cp': len(high_cp)
    }

def plot_results(signal, signal_cleaned, orig_cp, results, output_dir, csv_path):
    """Generates the 4-panel plot for a specific analysis result."""
    wavelet = results['wavelet']
    low_freq_signal = results['low_freq_signal']
    high_freq_combined = results['high_freq_combined']
    low_cp = results['low_cp']
    high_cp = results['high_cp']
    synth_cp = results['synthesized_cp']
    ref_name = results['ref_name']
    
    plt.figure(figsize=(15, 12))
    
    # Color definitions with specified RGB values
    blue = (74/255, 75/255, 157/255)
    red = (200/255, 22/255, 29/255)
    green = (90/255, 164/255, 174/255)
    yellow = (250/255, 192/255, 61/255)
    synth_color = (166/255, 85/255, 157/255)
    cleaned_color = (204/255, 93/255, 32/255)  # RGB: 204, 93, 32
    
    # 1. Original Signal with Cleaned Signal
    plt.subplot(4, 1, 1)
    plt.plot(signal, label='Original Signal (Power)', color='gray', alpha=0.6)
    plt.plot(signal_cleaned, label='Cleaned Signal (Outliers Removed)', color=cleaned_color, alpha=0.8)
    for cp in orig_cp:
        plt.axvline(x=cp, color=red, linestyle='--', alpha=0.8)
    plt.title('Original Power Signal with Cleaned Signal and Segmentation')
    plt.legend(loc='upper right')
    plt.grid(True)
    
    # 2. Low Frequency Component
    plt.subplot(4, 1, 2)
    plt.plot(low_freq_signal, label=f'Low Frequency (Approx A2 - {wavelet})', color=blue)
    for cp in low_cp:
        plt.axvline(x=cp, color=red, linestyle='--', alpha=0.8)
    plt.title(f'Low Frequency Component ({wavelet} Approx) with Segmentation')
    plt.legend(loc='upper right')
    plt.grid(True)
    
    # 3. High Frequency Component
    plt.subplot(4, 1, 3)
    plt.plot(high_freq_combined, label=f'High Frequency (Details D1+D2 - {wavelet})', color=green, alpha=0.8)
    for cp in high_cp:
        plt.axvline(x=cp, color=red, linestyle='--', alpha=0.8)
    plt.title(f'High Frequency Component ({wavelet} Details) with Segmentation')
    plt.legend(loc='upper right')
    plt.grid(True)
    
    # 4. Comparison with Synthesized Changepoints
    plt.subplot(4, 1, 4)
    plt.plot(signal, label='Original Signal', color=blue, alpha=0.8)
    plt.plot(low_freq_signal, label=f'Separated Square Wave ({wavelet})', color='gray', alpha=0.5, linewidth=1.5)
    
    # Custom color for Synthesized CP: RGB (166, 85, 157)
    

    # Add Original Changepoints (Green) - alpha set to 1.0
    for i, cp in enumerate(orig_cp):
        label = 'Original CP' if i == 0 else ""
        plt.axvline(x=cp, color=green, linestyle='--', linewidth=1, alpha=1.0, label=label)
        
    # Add Low-Freq Changepoints (Orange)
    for i, cp in enumerate(low_cp):
        label = 'Low-Freq CP' if i == 0 else ""
        plt.axvline(x=cp, color='orange', linestyle=':', linewidth=1, alpha=0.5, label=label)

    # Add High-Freq Changepoints (Yellow)
    for i, cp in enumerate(high_cp):
        label = 'High-Freq CP' if i == 0 else ""
        plt.axvline(x=cp, color=yellow, linestyle='-.', linewidth=1, alpha=0.5, label=label)
        
    # Add Synthesized Changepoints (Magenta Solid -> Specified RGB)
    for i, cp in enumerate(synth_cp):
        label = f'Synthesized CP (Ref: {ref_name})' if i == 0 else ""
        plt.axvline(x=cp, color=synth_color, linestyle='-', linewidth=2, alpha=1.0, label=label)
        
    plt.title(f'Comparison: Components vs Synthesized (Wavelet: {wavelet})')
    plt.legend(loc='upper right', fontsize='small', ncol=2)
    plt.grid(True)
    
    plt.tight_layout()
    
    filename_base = os.path.basename(csv_path).split('.')[0]
    plot_path = os.path.join(output_dir, f'wavelet_separation_{filename_base}_{wavelet}.png')
    plt.savefig(plot_path)
    plt.close() # Close to free memory
    print(f"Result plot saved to {plot_path}")

def export_synthesized_cp(df, synth_cp, output_dir, csv_path, wavelet):
    """
    Exports synthesized changepoints to a CSV file.
    Format: timestamp, power, datetime, changepoint_index
    """
    export_data = []
    for cp in synth_cp:
        idx = int(round(cp))
        if 0 <= idx < len(df):
            export_data.append({
                "timestamp": df.iloc[idx]["timestamp"] if "timestamp" in df.columns else idx,
                "power": df.iloc[idx]["power"],
                "datetime": df.iloc[idx]["datetime"] if "datetime" in df.columns else None,
                "changepoint_index": idx
            })
    
    if export_data:
        export_df = pd.DataFrame(export_data)
        # Reorder columns
        cols = ["timestamp", "power", "datetime", "changepoint_index"]
        export_df = export_df[[c for c in cols if c in export_df.columns]]
        
        filename_base = os.path.basename(csv_path).split('.')[0]
        output_path = os.path.join(output_dir, f"Changepoints_{filename_base}.csv")
        export_df.to_csv(output_path, index=False)
        print(f"Synthesized changepoints exported to {output_path}")

def main(input_path, output_dir, n=2, m=None, is_plot=True):
    # 1. Resolve input files
    if os.path.isfile(input_path):
        target_files = [input_path]
    elif os.path.isdir(input_path):
        target_files = [os.path.join(input_path, f) for f in os.listdir(input_path) if f.lower().endswith('.csv')]
        print(f"Found {len(target_files)} CSV files in {input_path}")
        if m is not None:
            target_files = target_files[:m]
            print(f"Limiting to first {m} files.")
    else:
        print(f"Error: Input path {input_path} not found.")
        return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for csv_path in target_files:
        print(f"\n" + "="*50)
        print(f"Processing: {os.path.basename(csv_path)}")
        
        # 2. Load data
        df = pd.read_csv(csv_path)
        signal = df['power'].values

        # 3. Apply sliding window outlier removal
        print("Applying sliding window outlier removal...")
        signal_cleaned, outlier_count, outlier_mask = sliding_window_outlier_removal(signal)
        print(f"  Detected and removed {outlier_count} outliers")

        # 4. Segment original signal once
        print("Performing segmentation on original signal...")
        orig_cp = get_segmentation_points(signal_cleaned)
        
        # 4. Test wavelets in order 4 -> 3 -> 2 -> 1
        wavelets_to_test = ['db4', 'db3', 'db2', 'db1']
        all_results = []
        
        for idx, wv in enumerate(wavelets_to_test):
            print(f"Testing wavelet: {wv} ({idx+1}/{len(wavelets_to_test)})...")
            res = run_wavelet_analysis(signal_cleaned, wv, orig_cp)
            res['order_priority'] = idx # Lower is better (db4=0, db1=3)
            all_results.append(res)
        
        # 5. Sorting logic:
        # - Low-Freq points (descending)
        # - High-Freq points (descending)
        # - Order priority (ascending)
        all_results.sort(key=lambda x: (-x['num_low_cp'], -x['num_high_cp'], x['order_priority']))
        
        # 6. Generate outputs for top n
        print(f"Top {n} Results for {os.path.basename(csv_path)}:")
        for i in range(min(n, len(all_results))):
            res = all_results[i]
            print(f"  Rank {i+1}: Wavelet={res['wavelet']}, Low-CP={res['num_low_cp']}, High-CP={res['num_high_cp']}")
            
            # Export CSV only for Rank 1 (best result) to match naming requirement
            if i == 0:
                export_synthesized_cp(df, res['synthesized_cp'], output_dir, csv_path, res['wavelet'])
            
            # Plot if requested
            if is_plot:
                plot_results(signal, signal_cleaned, orig_cp, res, output_dir, csv_path)

if __name__ == "__main__":
    # Can be a file path or a directory path
    input_source = r"F:\B__ProfessionProject\NILM\Clasp\mean_reversion\project\washing_machine\related\data"
    output_directory = r"F:\B__ProfessionProject\NILM\Clasp\wavelet_clasp_segmentation\result"
    
    # Parameter n: generate top n plots for each file
    n_plots = 1
    
    # Parameter m: limit the number of files to process from a directory (None for all)
    m_files = 20

    # is_plot: whether to generate and save plots
    is_plot = True
    
    main(input_source, output_directory, n=n_plots, m=m_files, is_plot=is_plot)
