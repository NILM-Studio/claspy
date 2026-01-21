import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score, silhouette_score
from sklearn.neighbors import KernelDensity
from scipy.signal import argrelextrema

def get_best_silhouette_score(file_path):
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

    if 'power' not in df.columns:
        print(f"Column 'power' not found in {file_path}")
        return None

    data = df['power'].values.reshape(-1, 1)
    
    # Check if we have enough data points
    if len(data) < 2:
        return None

    best_score = -1 
    
    # Iterate through possible k values
    max_k = min(10, len(data))
    
    # We are looking for the configuration that satisfies F-score > 100 AND has the best Silhouette score
    found_valid_config = False

    for k in range(2, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(data)
        
        # Calculate Silhouette Score
        if len(data) > 10000:
             sil_score = silhouette_score(data, labels, sample_size=10000)
        else:
             sil_score = silhouette_score(data, labels)

        # Criteria: Just select best Silhouette Score
        if sil_score > best_score:
            best_score = sil_score
            found_valid_config = True
    
    if found_valid_config:
        return best_score
    else:
        return None

def analyze_distribution(data_dir, plot_dir, score_dir):
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir, exist_ok=True)
    if not os.path.exists(score_dir):
        os.makedirs(score_dir, exist_ok=True)

    files = glob.glob(os.path.join(data_dir, '*.csv'))
    if not files:
        print(f"No CSV files found in {data_dir}")
        return None

    print(f"Found {len(files)} files. Calculating Silhouette scores...")
    
    silhouette_scores = []
    file_score_map = []

    for i, file_path in enumerate(files):
        score = get_best_silhouette_score(file_path)
        if score is not None:
            silhouette_scores.append(score)
            file_score_map.append({'filename': os.path.basename(file_path), 'score': score})
        
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(files)} files...")

    if not silhouette_scores:
        print("No valid silhouette scores found.")
        return None

    print(f"Collected {len(silhouette_scores)} scores.")

    # Save scores to numpy file
    npy_path = os.path.join(score_dir, 'silhouette_scores.npy')
    np.save(npy_path, np.array(silhouette_scores))
    print(f"Silhouette scores saved to {npy_path}")

    # Save file scores to CSV
    csv_path = os.path.join(score_dir, 'file_scores.csv')
    pd.DataFrame(file_score_map).to_csv(csv_path, index=False)
    print(f"File scores saved to {csv_path}")

    # Plotting the distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(silhouette_scores, kde=True, bins=20, color='blue')
    plt.title('Probability Distribution of Silhouette Scores')
    plt.xlabel('Silhouette Score')
    plt.ylabel('Count / Density')
    plt.grid(True, alpha=0.3)
    
    output_path = os.path.join(plot_dir, 'silhouette_score_distribution.png')
    plt.savefig(output_path)
    plt.close()
    print(f"Distribution plot saved to {output_path}")
    
    # Also save as a KDE plot only
    plt.figure(figsize=(10, 6))
    sns.kdeplot(silhouette_scores, fill=True, color='green')
    plt.title('Kernel Density Estimation of Silhouette Scores')
    plt.xlabel('Silhouette Score')
    plt.ylabel('Density')
    plt.grid(True, alpha=0.3)
    
    output_path_kde = os.path.join(plot_dir, 'silhouette_score_kde.png')
    plt.savefig(output_path_kde)
    plt.close()
    print(f"KDE plot saved to {output_path_kde}")

    # Perform Kernel Density Estimation
    X = np.array(silhouette_scores).reshape(-1, 1)
    kde = KernelDensity(kernel='gaussian', bandwidth=0.05).fit(X)
    
    # Generate points for evaluation
    X_plot = np.linspace(min(silhouette_scores) - 0.1, max(silhouette_scores) + 0.1, 1000).reshape(-1, 1)
    log_dens = kde.score_samples(X_plot)
    dens = np.exp(log_dens)
    
    # Find local minima in the density function to use as split points
    # We look for indices where the density is lower than both neighbors
    minima_indices = argrelextrema(dens, np.less)[0]
    
    # Filter minima that are too close to edges or insignificant (optional, here we take all valid local minima)
    split_points = X_plot[minima_indices].flatten()
    
    print(f"Found {len(split_points)} potential split points based on density minima: {split_points}")
    
    # Plot segmentation
    plt.figure(figsize=(10, 6))
    plt.plot(X_plot, dens, label='Density', color='black')
    plt.fill_between(X_plot[:, 0], 0, dens, alpha=0.3, color='gray')
    
    colors = plt.cm.rainbow(np.linspace(0, 1, len(split_points) + 1))
    
    # Add vertical lines for split points
    for i, split_point in enumerate(split_points):
        plt.axvline(x=split_point, color='red', linestyle='--', label=f'Split {i+1}: {split_point:.3f}')
        
    plt.title('Silhouette Score Density Segmentation')
    plt.xlabel('Silhouette Score')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    output_path_seg = os.path.join(plot_dir, 'silhouette_score_segmentation.png')
    plt.savefig(output_path_seg)
    plt.close()
    print(f"Segmentation plot saved to {output_path_seg}")
    
    # Save split points
    split_npy_path = os.path.join(score_dir, 'density_split_points.npy')
    np.save(split_npy_path, split_points)
    print(f"Split points saved to {split_npy_path}")

    # Return the first split point if available, else a default or None
    if len(split_points) > 0:
        return split_points[0]
    else:
        return None

if __name__ == "__main__":
    # Default behavior for standalone run (using previous defaults)
    DATA_DIR = r'./project/related/data'
    RESULTS_DIR = r'./project/related/plot'
    SCORE_DIR = r'./project/related/score'
    analyze_distribution(DATA_DIR, RESULTS_DIR, SCORE_DIR)
