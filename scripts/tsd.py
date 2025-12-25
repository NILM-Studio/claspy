import os
import sys
import pandas as pd
import numpy as np
from claspy.segmentation import BinaryClaSPSegmentation
from claspy.data_loader import load_tssb_dataset


# Add project root to sys.path to use local claspy package
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(script_dir, ".."))
dataset, window_size, true_cps, time_series = load_tssb_dataset(names=("CricketX",)).iloc[0,:]



# ======== Configuration ========
dir_path = os.path.join(script_dir, "washing_machine_cluster")
label_output_dir = os.path.join(script_dir, "label_output")
# plot_output_dir = os.path.join(script_dir, "plot/new")
# ===============================



if not os.path.exists(label_output_dir):
    os.makedirs(label_output_dir)
# For each CSV file in the directory
for fname in os.listdir(dir_path):
    if not fname.lower().endswith(".csv"):
        continue
    csv_path = os.path.join(dir_path, fname)
    df = pd.read_csv(csv_path)
    
    target_col = "power_new" if "power_new" in df.columns else "power"
    time_series = df[target_col].values
    print(f"File: {fname} | Using column: {target_col} | Time Series Length: {len(time_series)}")

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
        
    print(f"[{tag}] File: {fname} | Found change points: {clasp.change_points}")

    if len(clasp.change_points) > 0:
        export_data = []
        for idx in clasp.change_points:
            idx = int(idx)
            if 0 <= idx < len(df):
                export_data.append({
                    "timestamp": df.iloc[idx]["timestamp"],
                    "power": df.iloc[idx]["power"],
                    "datetime": df.iloc[idx]["datetime"],
                    "changepoint_index": idx
                })
        
        if export_data:
            export_df = pd.DataFrame(export_data)
            # Reorder columns to match requirement: timestamp,power,datetime,changepoint_index
            export_df = export_df[["timestamp", "power", "datetime", "changepoint_index"]]
            output_path = os.path.join(label_output_dir, f"Changepoints_{fname}")
            export_df.to_csv(output_path, index=False)
            print(f"Exported to {output_path}")

    # base = os.path.splitext(fname)[0]
    # out_path = os.path.join(plot_output_dir, f"segmentation_{base}.png")
    # clasp.plot(gt_cps=None, heading=f"Segmentation of {base}", ts_name=target_col, file_path=out_path)
