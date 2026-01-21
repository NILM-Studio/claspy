import os
import sys
import pandas as pd
import numpy as np
from claspy.segmentation import BinaryClaSPSegmentation

# Add project root to sys.path to use local claspy package
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(script_dir, ".."))

def run_segmentation(data_dir, label_dir, score_file_path=None):
    if not os.path.exists(label_dir):
        os.makedirs(label_dir)
        
    # Load scores if provided
    file_scores = {}
    if score_file_path and os.path.exists(score_file_path):
        try:
            scores_df = pd.read_csv(score_file_path)
            # Assume columns are 'filename' and 'score' based on previous step
            if 'filename' in scores_df.columns and 'score' in scores_df.columns:
                file_scores = dict(zip(scores_df['filename'], scores_df['score']))
            print(f"Loaded {len(file_scores)} scores from {score_file_path}")
        except Exception as e:
            print(f"Error reading score file: {e}")

    # For each CSV file in the directory
    if not os.path.exists(data_dir):
        print(f"Data directory not found: {data_dir}")
        return

    files = [f for f in os.listdir(data_dir) if f.lower().endswith(".csv")]
    print(f"Found {len(files)} files in {data_dir}")

    for i, fname in enumerate(files):
        # Optional: Check if file has a score (if we wanted to filter)
        # current_score = file_scores.get(fname, None)
        
        csv_path = os.path.join(data_dir, fname)
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"Error reading {fname}: {e}")
            continue
        
        target_col = "power_new" if "power_new" in df.columns else "power"
        
        # Check if data is sufficient
        if len(df) < 10: # arbitrary small limit to avoid errors
            # print(f"Skipping {fname}: too short ({len(df)})")
            continue

        time_series = df[target_col].values
        
        score_info = f" | Score: {file_scores.get(fname, 'N/A')}" if file_scores else ""
        # print(f"File: {fname} | Using column: {target_col} | Time Series Length: {len(time_series)}{score_info}")

        try:
            clasp = BinaryClaSPSegmentation(
                n_segments="learn",
                window_size="suss", #!==  fixed number | strategy (suss fft acf)
                validation="score_threshold",
                threshold=0.001, #!== <=0.3 remain the same for the output
            )
            clasp.fit_predict(time_series)
            tag = "suss"
            
            if len(clasp.change_points) == 0:
                clasp = BinaryClaSPSegmentation(
                n_segments="learn",
                window_size="acf", #!==  fixed number | strategy (suss fft acf)
                validation="score_threshold",
                threshold=0.001, #!== <=0.3 remain the same for the output
                )
                clasp.fit_predict(time_series)
                tag = "acf"
                
            # print(f"[{tag}] File: {fname} | Found change points: {clasp.change_points}")

            if len(clasp.change_points) > 0:
                export_data = []
                for idx in clasp.change_points:
                    idx = int(idx)
                    if 0 <= idx < len(df):
                        export_data.append({
                            "timestamp": df.iloc[idx]["timestamp"] if "timestamp" in df.columns else idx,
                            "power": df.iloc[idx]["power"],
                            "datetime": df.iloc[idx]["datetime"] if "datetime" in df.columns else None,
                            "changepoint_index": idx
                        })
                
                if export_data:
                    export_df = pd.DataFrame(export_data)
                    # Reorder columns to match requirement: timestamp,power,datetime,changepoint_index
                    cols_to_keep = [c for c in ["timestamp", "power", "datetime", "changepoint_index"] if c in export_df.columns]
                    export_df = export_df[cols_to_keep]
                    
                    output_path = os.path.join(label_dir, f"Changepoints_{fname}")
                    export_df.to_csv(output_path, index=False)
                    # print(f"Exported to {output_path}")
        except Exception as e:
            print(f"Error processing {fname}: {e}")
        
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(files)} files...")

if __name__ == "__main__":
    dir_path = os.path.join(script_dir, "washing_machine_cluster")
    label_output_dir = os.path.join(script_dir, "label_output")
    run_segmentation(dir_path, label_output_dir)
