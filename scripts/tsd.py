import os
import sys
import pandas as pd

# Add project root to sys.path to use local claspy package
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(script_dir, ".."))

from claspy.segmentation import BinaryClaSPSegmentation
from claspy.data_loader import load_tssb_dataset

dataset, window_size, true_cps, time_series = load_tssb_dataset(names=("CricketX",)).iloc[0,:]

csv_path = os.path.join(script_dir, "washing_machine", "Washing_Machine_20131216_171003_20131216_174732_305s.csv")  # 文件夹名，文件名
df = pd.read_csv(csv_path)
time_series = df["power"].values


print(f"Time Series Length: {len(time_series)}")

# Adjust parameters for short time series
# n_segments="learn": automatically learn number of segments
# window_size=10: smaller window size for shorter series (default might be too large)
# validation="score_threshold": use score threshold instead of significance test
# threshold=0.6: lower threshold to be more sensitive
clasp = BinaryClaSPSegmentation(
    n_segments="learn",
    window_size="acf",  # suss, fft, acf
    validation="score_threshold",
    threshold=0.6
)
clasp.fit_predict(time_series)

print(f"Found change points: {clasp.change_points}")

clasp.plot(gt_cps=None, heading="Segmentation of Washing_Machine_20131216_171003_20131216_174732_305s", ts_name="power", file_path=os.path.join(script_dir, "segmentation_example.png"))
