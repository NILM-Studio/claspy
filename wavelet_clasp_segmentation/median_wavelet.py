# 导入需要的库（都是数据分析常用库，pip install numpy pandas scipy matplotlib 即可安装）
import numpy as np
import pandas as pd
from scipy.signal import medfilt
import matplotlib.pyplot as plt

# -------------------------- 1. 生成【带阶跃+带孤立毛刺】的测试时间序列（贴合你的场景） --------------------------
# 模拟：前300个点稳定在100附近，后300个点阶跃到1000附近，手动插入3个孤立毛刺
np.random.seed(42)  # 固定随机种子，结果可复现
ts_100 = pd.Series(np.random.normal(loc=100, scale=2, size=300))  # 平稳段1：均值100，小波动
ts_1000 = pd.Series(np.random.normal(loc=1000, scale=5, size=300)) # 平稳段2：阶跃到均值1000，小波动
ts = pd.concat([ts_100, ts_1000]).reset_index(drop=True)           # 拼接成完整序列

file = pd.read_csv(r"F:\B__ProfessionProject\NILM\Clasp\mean_reversion\project\washing_machine\related\data\Washing_Machine_20121129_210111_20121129_215705_456s.csv")
ts = file["power"]
# 手动插入【孤立毛刺】（模拟采集的异常点）
ts.iloc[80] = 250   # 100平稳段的上跳毛刺
ts.iloc[150] = 20   # 100平稳段的下跳毛刺
ts.iloc[400] = 600  # 1000平稳段的下跳毛刺

# -------------------------- 2. 中值滤波核心代码（一行去毛刺，重中之重） --------------------------
# 方案1：窗口=3，最常用！仅剔除【单点孤立毛刺】，对阶跃无任何影响，推荐优先用
ts_filter_3 = medfilt(ts.values, kernel_size=3)  # 入参传numpy数组即可，返回也是数组

# 方案2：窗口=5，可选！如果偶尔有【连续2个点的毛刺】，用5，依然完美保留阶跃
ts_filter_5 = medfilt(ts.values, kernel_size=5)

# 转回pandas Series，保持索引一致（方便后续处理/绘图）
ts_filter_3 = pd.Series(ts_filter_3, index=ts.index)
ts_filter_5 = pd.Series(ts_filter_5, index=ts.index)

# -------------------------- 3. 可视化对比（原始序列 vs 滤波后序列，直观看到去毛刺效果） --------------------------
plt.figure(figsize=(12, 6))
plt.plot(ts, label='原始序列（含阶跃+毛刺）', color='#ff4d4f', alpha=0.7, linewidth=1.2)
plt.plot(ts_filter_3, label='中值滤波后（窗口=3，推荐）', color='#1677ff', linewidth=2)
# plt.plot(ts_filter_5, label='中值滤波后（窗口=5）', color='#36cbcb', linewidth=2) # 可选显示
plt.title('中值滤波去除时间序列毛刺（保留阶跃不变）', fontsize=14)
plt.xlabel('时间点')
plt.ylabel('数值')
plt.legend()
plt.grid(alpha=0.3)
plt.show()