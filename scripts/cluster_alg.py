import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def calculate_sse(data):
    if len(data) == 0:
        return 0
    mean = np.mean(data)
    return np.sum((data - mean) ** 2)

def f_test_crit(alpha, df1, df2):
    return stats.f.ppf(1 - alpha, df1, df2)

def find_change_point(data, alpha=0.05, min_size=5):
    n = len(data)
    if n < 2 * min_size:
        return None, 0, []

    sse_total = calculate_sse(data)
    best_f = -1
    best_k = -1
    
    f_stats = []
    
    # Iterate through possible split points
    # We leave min_size points on each side
    for k in range(min_size, n - min_size + 1):
        group1 = data[:k]
        group2 = data[k:]
        
        sse_split = calculate_sse(group1) + calculate_sse(group2)
        
        # Degrees of freedom
        # p_total = 1 (mean of whole)
        # p_split = 2 (mean of left + mean of right)
        # df1 = p_split - p_total = 1
        # df2 = n - p_split = n - 2
        
        if sse_split == 0:
             if sse_total > 0:
                 f_stat = np.inf
             else:
                 f_stat = 0
        else:
            f_stat = ((sse_total - sse_split) / 1) / (sse_split / (n - 2))
            
        f_stats.append(f_stat)
        
        if f_stat > best_f:
            best_f = f_stat
            best_k = k
            
    # Critical value
    f_crit = f_test_crit(alpha, 1, n - 2)
    
    if best_f > f_crit:
        return best_k, best_f, f_stats
    else:
        return None, best_f, f_stats

def recursive_segmentation(data, start_idx=0, alpha=0.05, min_size=5):
    """
    Recursively find change points using F-test.
    Returns a list of absolute indices of change points.
    """
    k, f_val, _ = find_change_point(data, alpha, min_size)
    
    if k is None:
        return []
    
    # Split found at relative index k
    abs_k = start_idx + k
    
    # Recurse on both segments
    # Note: We need to be careful not to over-segment. 
    # The F-test checks if *one* split is better than *none*.
    # When recursing, we test if *another* split is better than just the current segment mean.
    
    left_cps = recursive_segmentation(data[:k], start_idx, alpha, min_size)
    right_cps = recursive_segmentation(data[k:], abs_k, alpha, min_size)
    
    return left_cps + [abs_k] + right_cps

def main():
    # Setup paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "washing_machine")
    plot_dir = os.path.join(script_dir, "plot", "cluster_alg")
    
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
        
    # Get list of CSV files
    if not os.path.exists(data_dir):
        print(f"Data directory not found: {data_dir}")
        return

    files = sorted([f for f in os.listdir(data_dir) if f.lower().endswith(".csv")])
    
    # Process the first 100 files
    target_files = files[:100]
    print(f"Processing {len(target_files)} files...")
    print("Note: Processing only 'power' column, ignoring datetime/timestamp.")
    
    for fname in target_files:
        path = os.path.join(data_dir, fname)
        try:
            # Explicitly read only the power column to demonstrate we ignore datetime
            # We assume the file has a header. If 'power' is not in header, we'll catch it.
            # Reading all then selecting is safer for parsing, but we will select immediately.
            df = pd.read_csv(path)
            
            if "power" not in df.columns:
                print(f"Skipping {fname}: 'power' column not found.")
                continue
            
            # Select ONLY power column, drop NaNs, and get numpy array
            data = df["power"].dropna().values
            
            if len(data) < 10:
                print(f"Skipping {fname}: Not enough data points ({len(data)}).")
                continue

            # 1. Calculate Global F-stats for the bottom plot (Score)
            # This shows the F-statistic for a single split at each point
            n = len(data)
            min_size = 5
            alpha = 0.05
            
            _, _, f_stats_global = find_change_point(data, alpha, min_size)
            
            # Pad f_stats to match data length for plotting
            f_curve = np.zeros(n)
            if f_stats_global:
                # f_stats_global corresponds to indices min_size ... n-min_size
                f_curve[min_size : min_size + len(f_stats_global)] = f_stats_global
            
            # 2. Find all change points recursively
            cps = sorted(recursive_segmentation(data, 0, alpha, min_size))
            
            # 3. Plotting
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8), sharex=True)
            
            # Plot 1: Time Series with colored segments
            boundaries = [0] + cps + [len(data)]
            # Use a colormap
            cmap = plt.get_cmap("tab10")
            
            for i in range(len(boundaries)-1):
                start = boundaries[i]
                end = boundaries[i+1]
                # To make the line continuous, we can plot from start to end (inclusive of start of next)
                # But simple segments are fine. 
                # Ideally, point 'end' belongs to the second segment, but for line continuity...
                # Let's plot x from start to end-1
                x_vals = np.arange(start, end)
                segment_data = data[start:end]
                
                # If we want to connect lines, we can include the next point
                # But strictly speaking, they are different clusters.
                ax1.plot(x_vals, segment_data, color=cmap(i % 10))
            
            # Draw Change Points
            for cp in cps:
                ax1.axvline(x=cp, color='green', linestyle='-', linewidth=2, label='Predicted Change Point' if cp==cps[0] else "")
            
            ax1.set_title(f"Segmentation of {os.path.splitext(fname)[0]}")
            ax1.set_ylabel("power")
            
            # Legend
            handles, labels = ax1.get_legend_handles_labels()
            if handles:
                ax1.legend([handles[0]], [labels[0]])
            
            # Plot 2: F-Score
            ax2.plot(f_curve, color='black')
            ax2.set_ylabel("F-Test Score (Global)")
            ax2.set_xlabel("split point")
            
            # Mark max score point
            if len(cps) > 0:
                 # The global best split might be one of the cps
                 pass
            
            plt.tight_layout()
            out_path = os.path.join(plot_dir, f"segmentation_{os.path.splitext(fname)[0]}.png")
            plt.savefig(out_path)
            plt.close()
            
            print(f"Processed {fname}: Found {len(cps)} segments.")
            
        except Exception as e:
            print(f"Error processing {fname}: {e}")

if __name__ == "__main__":
    main()
