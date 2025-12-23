import os
import pandas as pd
import matplotlib.pyplot as plt

def plot_distribution(data, title, xlabel, filename, output_dir):
    plt.figure(figsize=(10, 6))
    plt.hist(data, bins=30, edgecolor='black', alpha=0.7)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Frequency")
    plt.grid(True, alpha=0.3)
    
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path)
    plt.close() # Close the figure to free memory
    print(f"Plot saved to {output_path}")

def print_stats(name, data):
    if not data:
        print(f"No data for {name}")
        return
    print(f"--- {name} Statistics ---")
    print(f"Count: {len(data)}")
    print(f"Min: {min(data):.2f}")
    print(f"Max: {max(data):.2f}")
    print(f"Average: {sum(data) / len(data):.2f}")
    print("")

def main():
    # Define directories
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "washing_machine")
    plot_dir = os.path.join(script_dir, "plot" , "analysis")
    
    # Ensure plot directory exists
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
        
    means = []
    maxs = []
    vars = []
    
    # Iterate through all CSV files in the directory
    if not os.path.exists(data_dir):
        print(f"Directory not found: {data_dir}")
        return

    files = [f for f in os.listdir(data_dir) if f.lower().endswith(".csv")]
    print(f"Found {len(files)} CSV files in {data_dir}")

    for fname in files:
        csv_path = os.path.join(data_dir, fname)
        try:
            df = pd.read_csv(csv_path)
            if "power" in df.columns:
                # Calculate metrics
                mean_val = df["power"].mean()
                max_val = df["power"].max()
                var_val = df["power"].var()
                
                # Append to lists
                means.append(mean_val)
                maxs.append(max_val)
                vars.append(var_val)
            else:
                print(f"Warning: 'power' column not found in {fname}")
        except Exception as e:
            print(f"Error reading {fname}: {e}")
            
    if not means:
        print("No power data found to plot.")
        return
        
    # Plot distributions
    plot_distribution(means, "Distribution of Mean Power Values", "Mean Power", "power_mean_distribution.png", plot_dir)
    plot_distribution(maxs, "Distribution of Max Power Values", "Max Power", "power_max_distribution.png", plot_dir)
    plot_distribution(vars, "Distribution of Power Variance", "Power Variance", "power_var_distribution.png", plot_dir)
    
    # Print statistics
    print_stats("Mean Power", means)
    print_stats("Max Power", maxs)
    print_stats("Power Variance", vars)

if __name__ == "__main__":
    main()
