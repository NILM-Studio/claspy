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

dir_path = os.path.join(script_dir, "washing_machine_cluster")  #!== 文件夹名
# For each CSV file in the directory
for fname in os.listdir(dir_path):
    if not fname.lower().endswith(".csv"):
        continue
    csv_path = os.path.join(dir_path, fname)
    df = pd.read_csv(csv_path)
    time_series = df["power_new"].values
    print(f"File: {fname} | Time Series Length: {len(time_series)}")

    clasp = BinaryClaSPSegmentation(
        n_segments="learn",
        window_size="suss", #!==  fixed number | strategy (suss fft acf)
        validation="score_threshold",
        # threshold=0.001, #!== <=0.3 remain the same for the output
    )
    clasp.fit_predict(time_series)
    print(f"File: {fname} | Found change points: {clasp.change_points}")

    base = os.path.splitext(fname)[0]
    out_path = os.path.join(script_dir+ "/plot/new", f"segmentation_{base}.png")
    clasp.plot(gt_cps=None, heading=f"Segmentation of {base}", ts_name="power_new", file_path=out_path)
