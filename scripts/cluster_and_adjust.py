import os
import glob
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Directories
SOURCE_DIR = r'f:\B__ProfessionProject\NILM\Clasp\scripts\washing_machine'
TARGET_DIR = r'f:\B__ProfessionProject\NILM\Clasp\scripts\washing_machine_cluster'

if not os.path.exists(TARGET_DIR):
    try:
        os.makedirs(TARGET_DIR)
    except OSError:
        pass

def process_file(file_path):
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
            
    print(f"File: {os.path.basename(file_path)}, Best Score: {best_score:.4f}, k={best_k}")

    if best_score > 0.82:
        # Add labels to dataframe
        df['cluster_label'] = best_labels
        
        # Calculate mean power for each cluster
        cluster_means = {}
        for label in range(best_k):
            cluster_data = df[df['cluster_label'] == label]['power']
            cluster_means[label] = cluster_data.mean()
            
        # Find the cluster with the minimum mean power
        min_mean_label = min(cluster_means, key=cluster_means.get)
        min_mean_val = cluster_means[min_mean_label]
        
        print(f"  Processing: Score > 0.82. Min cluster: {min_mean_label} (mean={min_mean_val:.2f})")
        
        # Calculate adjusted power
        # power_new = power - (cluster_mean - min_mean_val)
        
        def adjust_power(row):
            cluster_mean = cluster_means[row['cluster_label']]
            diff = cluster_mean - min_mean_val
            return row['power'] - diff
            
        df['power_new'] = df.apply(adjust_power, axis=1)
        
        # Save to target directory
        filename = os.path.basename(file_path)
        output_path = os.path.join(TARGET_DIR, filename)
        df.to_csv(output_path, index=False)
        print(f"  Saved adjusted file to {output_path}")

def main():
    files = glob.glob(os.path.join(SOURCE_DIR, '*.csv'))
    if not files:
        print(f"No CSV files found in {SOURCE_DIR}")
        return

    print(f"Found {len(files)} files in {SOURCE_DIR}")
    
    for file_path in files:
        process_file(file_path)

if __name__ == "__main__":
    main()
