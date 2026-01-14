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
        
        if len(clasp.change_points) == 0:
            clasp = BinaryClaSPSegmentation(
                n_segments="learn",
                window_size="acf",
                validation="score_threshold",
                threshold=0.001,
            )
            clasp.fit_predict(time_series)
        return clasp.change_points
    except Exception as e:
        print(f"Segmentation error: {e}")
        return []

def run_wavelet_analysis(signal, wavelet):
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
    
    return {
        'wavelet': wavelet,
        'low_freq_signal': low_freq_signal,
        'high_freq_combined': high_freq_combined,
        'low_cp': low_cp,
        'high_cp': high_cp,
        'num_low_cp': len(low_cp),
        'num_high_cp': len(high_cp)
    }

def plot_results(signal, orig_cp, results, output_dir, csv_path):
    """Generates the 4-panel plot for a specific analysis result."""
    wavelet = results['wavelet']
    low_freq_signal = results['low_freq_signal']
    high_freq_combined = results['high_freq_combined']
    low_cp = results['low_cp']
    high_cp = results['high_cp']
    
    plt.figure(figsize=(15, 12))
    
    # 1. Original Signal
    plt.subplot(4, 1, 1)
    plt.plot(signal, label='Original Signal (Power)', color='gray', alpha=0.6)
    for cp in orig_cp:
        plt.axvline(x=cp, color='red', linestyle='--', alpha=0.8)
    plt.title('Original Power Signal with Segmentation')
    plt.legend()
    plt.grid(True)
    
    # 2. Low Frequency Component
    plt.subplot(4, 1, 2)
    plt.plot(low_freq_signal, label=f'Low Frequency (Approx A2 - {wavelet})', color='blue')
    for cp in low_cp:
        plt.axvline(x=cp, color='red', linestyle='--', alpha=0.8)
    plt.title(f'Low Frequency Component ({wavelet} Approx) with Segmentation')
    plt.legend()
    plt.grid(True)
    
    # 3. High Frequency Component
    plt.subplot(4, 1, 3)
    plt.plot(high_freq_combined, label=f'High Frequency (Details D1+D2 - {wavelet})', color='green', alpha=0.8)
    for cp in high_cp:
        plt.axvline(x=cp, color='red', linestyle='--', alpha=0.8)
    plt.title(f'High Frequency Component ({wavelet} Details) with Segmentation')
    plt.legend()
    plt.grid(True)
    
    # 4. Comparison
    plt.subplot(4, 1, 4)
    plt.plot(signal, label='Original Signal', color='gray', alpha=0.3)
    plt.plot(low_freq_signal, label=f'Separated Square Wave ({wavelet})', color='blue', linewidth=1.5)
    
    for i, cp in enumerate(orig_cp):
        label = 'Original Changepoint' if i == 0 else ""
        plt.axvline(x=cp, color='green', linestyle='--', linewidth=1.5, alpha=0.7, label=label)
        
    for i, cp in enumerate(low_cp):
        label = 'Low-Freq Changepoint' if i == 0 else ""
        plt.axvline(x=cp, color='red', linestyle='-', linewidth=1.5, alpha=0.7, label=label)
        
    plt.title(f'Comparison: Original vs Low-Freq Changepoints ({wavelet})')
    plt.legend(loc='best')
    plt.grid(True)
    
    plt.tight_layout()
    
    filename_base = os.path.basename(csv_path).split('.')[0]
    plot_path = os.path.join(output_dir, f'wavelet_separation_{wavelet}_{filename_base}.png')
    plt.savefig(plot_path)
    plt.close() # Close to free memory
    print(f"Result plot saved to {plot_path}")

def main(input_path, output_dir, n=2, m=None):
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

    for csv_path in target_files:
        print(f"\n" + "="*50)
        print(f"Processing: {os.path.basename(csv_path)}")
        
        # 2. Load data
        df = pd.read_csv(csv_path)
        signal = df['power'].values
        
        # 3. Segment original signal once
        print("Performing segmentation on original signal...")
        orig_cp = get_segmentation_points(signal)
        
        # 4. Test wavelets in order 4 -> 3 -> 2 -> 1
        wavelets_to_test = ['db4', 'db3', 'db2', 'db1']
        all_results = []
        
        for idx, wv in enumerate(wavelets_to_test):
            print(f"Testing wavelet: {wv} ({idx+1}/{len(wavelets_to_test)})...")
            res = run_wavelet_analysis(signal, wv)
            res['order_priority'] = idx # Lower is better (db4=0, db1=3)
            all_results.append(res)
        
        # 5. Sorting logic:
        # - Low-Freq points (descending)
        # - High-Freq points (descending)
        # - Order priority (ascending)
        all_results.sort(key=lambda x: (-x['num_low_cp'], -x['num_high_cp'], x['order_priority']))
        
        # 6. Generate plots for top n
        print(f"Top {n} Results for {os.path.basename(csv_path)}:")
        for i in range(min(n, len(all_results))):
            res = all_results[i]
            print(f"  Rank {i+1}: Wavelet={res['wavelet']}, Low-CP={res['num_low_cp']}, High-CP={res['num_high_cp']}")
            plot_results(signal, orig_cp, res, output_dir, csv_path)

if __name__ == "__main__":
    # Can be a file path or a directory path
    input_source = r"f:\B__ProfessionProject\NILM\Clasp\mean_reversion\test\input"
    output_directory = r"F:\B__ProfessionProject\NILM\Clasp\mean_reversion\test\test\best_plot"
    
    # Parameter n: generate top n plots for each file
    n_plots = 1
    
    # Parameter m: limit the number of files to process from a directory (None for all)
    m_files = 1000
    
    main(input_source, output_directory, n=n_plots, m=m_files)
