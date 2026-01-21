import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

def visualize_data_with_labels(data_dir, label_dir):
    """
    Visualize power, power_new and split points from ground truth labels
    
    Args:
        data_dir: Directory containing test data files
        label_dir: Directory containing test label files
    """
    # Get all data files
    data_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    
    print("=== Creating Simple Visualizations ===")
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
        ground_truth_indices = label_df['changepoint_index'].tolist()
        
        # Create plot
        plt.figure(figsize=(15, 10))
        
        # Plot power and power_new
        plt.subplot(3, 1, 1)
        plt.plot(df['power'], label='Original Power', color='blue', alpha=0.7)
        plt.title(f"Power Analysis for {data_file}")
        plt.ylabel('Power')
        plt.legend()
        
        # Plot power_new
        plt.subplot(3, 1, 2)
        plt.plot(df['power_new'], label='Adjusted Power (power_new)', color='green', alpha=0.7)
        plt.ylabel('Power')
        plt.legend()
        
        # Plot both power and power_new together
        plt.subplot(3, 1, 3)
        plt.plot(df['power'], label='Original Power', color='blue', alpha=0.5)
        plt.plot(df['power_new'], label='Adjusted Power (power_new)', color='green', alpha=0.8)
        plt.xlabel('Index')
        plt.ylabel('Power')
        plt.legend()
        
        # Add ground truth split points to all subplots
        for i in range(1, 4):
            plt.subplot(3, 1, i)
            for idx in ground_truth_indices:
                plt.axvline(x=idx, color='red', linestyle='--', linewidth=1.5, 
                           label='Split Point' if idx == ground_truth_indices[0] else "")
            plt.legend()
        
        plt.tight_layout()
        
        # Save plot
        plot_filename = f"simple_visualization_{os.path.splitext(data_file)[0]}.png"
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"  Plot saved as: {plot_filename}")
        
        # Close the plot to free memory
        plt.close()
    
    print("\n" + "=" * 60)
    print("=== All Visualizations Completed ===")
    print("=" * 60)

def main():
    # Define test directories
    DATA_DIR = r'F:\B__ProfessionProject\NILM\Clasp\mean_reversion\project\test\data'
    LABEL_DIR = r'F:\B__ProfessionProject\NILM\Clasp\mean_reversion\project\test\label'
    
    # Create visualizations
    visualize_data_with_labels(DATA_DIR, LABEL_DIR)

if __name__ == "__main__":
    main()
