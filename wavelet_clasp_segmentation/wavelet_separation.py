import pandas as pd
import numpy as np
import pywt
import matplotlib.pyplot as plt
import os
import sys

# Add project root to sys.path if needed
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(script_dir, ".."))
from scipy.signal import medfilt
from claspy.segmentation import BinaryClaSPSegmentation

def medfilt_outlier_removal(series):
    """
    Perform outlier removal using median filter.
    
    Args:
        series (array-like): Input time series data
        window_size (int): Size of the sliding window (unused, kept for compatibility)
        z_threshold (float): Z-score threshold for outlier detection (unused, kept for compatibility)
        interpolation_method (str): Interpolation method for replacing outliers (unused, kept for compatibility)
    
    Returns:
        tuple: (cleaned_series, outlier_count, outlier_mask)
    """
    import numpy as np
    
    # Convert to numpy array if not already
    ts = np.asarray(series)
    
    # Apply median filter with kernel size 5
    cleaned_series = medfilt(ts, kernel_size=5)
    
    # For compatibility, return dummy values for outlier_count and outlier_mask
    # Since medfilt doesn't explicitly identify outliers, we'll return 0 and a mask of False
    outlier_mask = np.zeros_like(ts, dtype=bool)
    
    return cleaned_series, outlier_mask

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
        'num_high_cp': len(high_cp),
        'cleaned_signal': signal
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

    # ---------------------------------------------------------
    # Generate Wavelet Transform Heatmap (Scalogram)
    # ---------------------------------------------------------
    # Continuous Wavelet Transform (CWT) to visualize Time-Frequency-Energy
    # Scales: 1 to 128. Small scale = High Freq, Large scale = Low Freq
    scales = np.arange(1, 128)
    
    # Use a continuous wavelet for visualization (e.g. Complex Morlet)
    # Discrete wavelets like 'db4' are not always suitable or supported for CWT in all pywt versions
    cwt_wavelet = 'cmor1.5-1.0'
    try:
        cwtmatr, freqs = pywt.cwt(low_freq_signal, scales, cwt_wavelet)
    except Exception:
        # Fallback to Mexican Hat if Complex Morlet is not available
        cwt_wavelet = 'mexh'
        cwtmatr, freqs = pywt.cwt(low_freq_signal, scales, cwt_wavelet)
    
    plt.figure(figsize=(15, 8))
    # X-axis: Time, Y-axis: Scale
    # origin='lower' ensures Scale 1 is at the bottom
    plt.imshow(np.abs(cwtmatr), extent=[0, len(low_freq_signal), scales[0], scales[-1]], 
               cmap='jet', aspect='auto', interpolation='nearest', origin='lower')
    
    plt.colorbar(label='Coefficient Magnitude (Energy)')
    plt.xlabel('Time (Index)')
    plt.ylabel('Wavelet Scale (Small=High Freq, Large=Low Freq)')
    plt.title(f'Wavelet Transform Heatmap (Scalogram) of Low Freq Signal\nWavelet: {cwt_wavelet}')
    
    heatmap_path = os.path.join(output_dir, f'wavelet_heatmap_{filename_base}_{cwt_wavelet}.png')
    plt.savefig(heatmap_path)
    plt.close()
    print(f"Heatmap plot saved to {heatmap_path}")

def export_synthesized_cp(df, results, output_dir, csv_path):
    """
    Exports signals to data/ subfolder and changepoints to label/ subfolder.
    """
    filename_base = os.path.basename(csv_path).split('.')[0]

    # 1. Export Signals to data/
    data_dir = os.path.join(output_dir, 'data')
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # Combine signals into a single DataFrame
    # Target columns: timestamp, power, cleaned_power, high_freq, low_freq, datetime
    signal_export_data = {
        "timestamp": df["timestamp"] if "timestamp" in df.columns else df.index,
        "power": df["power"],
        "cleaned_power": results['cleaned_signal'] if 'cleaned_signal' in results else np.zeros(len(df)),
        "high_freq": results['high_freq_combined'] if 'high_freq_combined' in results else np.zeros(len(df)),
        "low_freq": results['low_freq_signal'] if 'low_freq_signal' in results else np.zeros(len(df)),
        "datetime": df["datetime"] if "datetime" in df.columns else [None]*len(df)
    }
    
    sig_df = pd.DataFrame(signal_export_data)
    # Ensure column order
    sig_cols = ["timestamp", "power", "cleaned_power", "high_freq", "low_freq", "datetime"]
    sig_df = sig_df[[c for c in sig_cols if c in sig_df.columns]]
    
    out_name = f"{filename_base}.csv"
    out_path = os.path.join(data_dir, out_name)
    sig_df.to_csv(out_path, index=False)
    print(f"Signals exported to {out_path}")

    # 2. Export Changepoints to label/
    label_dir = os.path.join(output_dir, 'label')
    if not os.path.exists(label_dir):
        os.makedirs(label_dir)

    export_data = []
    # 0: synthesized_cp, 1: low_cp, 2: high_cp
    cp_sets = [
        (results.get('synthesized_cp', []), 0),
        (results.get('low_cp', []), 1),
        (results.get('high_cp', []), 2)
    ]

    for cp_list, label_type in cp_sets:
        for cp in cp_list:
            idx = int(round(cp))
            if 0 <= idx < len(df):
                export_data.append({
                    "timestamp": df.iloc[idx]["timestamp"] if "timestamp" in df.columns else idx,
                    "power": df.iloc[idx]["power"],
                    "datetime": df.iloc[idx]["datetime"] if "datetime" in df.columns else None,
                    "changepoint_index": idx,
                    "label_type": label_type
                })

    if export_data:
        export_df = pd.DataFrame(export_data)
        # Reorder columns
        cols = ["timestamp", "power", "datetime", "changepoint_index", "label_type"]
        export_df = export_df[[c for c in cols if c in export_df.columns]]
        
        output_path = os.path.join(label_dir, f"Changepoints_{filename_base}.csv")
        export_df.to_csv(output_path, index=False)
        print(f"Changepoints exported to {output_path}")

def main(input_path, output_dir, n=2, m=None, is_plot=True, apply_diff=False):
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

        if apply_diff:
            print("Applying rate of change conversion (diff)...")
            # Calculate difference: posterior - anterior
            # Pad with 0 at the beginning to maintain signal length
            signal_cleaned = np.concatenate(([0], np.diff(signal)))
        else:
            signal_cleaned = signal
        
        # 3. Apply median filter outlier removal
        print("Applying median filter outlier removal...")
        signal_cleaned, outlier_mask = medfilt_outlier_removal(signal_cleaned)
        print(f"  Detected and removed {outlier_mask.sum()} outliers")

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
                export_synthesized_cp(df, res, output_dir, csv_path)
            
            # Plot if requested
            if is_plot:
                plot_results(signal, signal_cleaned, orig_cp, res, output_dir, csv_path)

if __name__ == "__main__":
    # Can be a file path or a directory path
    input_source = r"F:\B__ProfessionProject\NILM\Clasp\mean_reversion(out-of-date)\project\washing_machine\related\data"
    output_directory = r"F:\B__ProfessionProject\NILM\Clasp\wavelet_clasp_segmentation\result7"
    
    # Parameter n: generate top n plots for each file
    n_plots = 1
    
    # Parameter m: limit the number of files to process from a directory (None for all)
    m_files = 10000
    
    # Parameter apply_diff: whether to apply rate of change conversion (diff)
    # If True, calculates the difference between consecutive points (posterior - anterior)
    apply_diff = False

    # is_plot: whether to generate and save plots
    is_plot = False
    
    main(input_source, output_directory, n=n_plots, m=m_files, is_plot=is_plot, apply_diff=apply_diff)
