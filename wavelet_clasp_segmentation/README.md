# Wavelet ClaSP Segmentation

本项目旨在结合 **小波变换 (Wavelet Transform)** 和 **ClaSPSegmentation** 算法，对时间序列数据（特别是非侵入式负荷监测 NILM 场景下的功率信号）进行高精度的变点检测 (Change Point Detection, CPD)。

核心脚本为 `wavelet_separation.py`，它通过多尺度分解和分频段处理，显著提升了信号在复杂背景下的分割效果。

---

## 核心算法流程与数学原理

### 1. 异常值去除 (Outlier Removal)
在进行信号分解前，使用中值滤波器 (Median Filter) 处理原始信号 $x[n]$，以剔除孤立的尖峰噪声（毛刺），同时保留重要的阶跃边缘。

**数学公式：**
对于长度为 $2k+1$ 的窗口，中值滤波后的输出 $y[n]$ 定义为：
$$y[n] = \text{median}\{x[n-k], \dots, x[n], \dots, x[n+k]\}$$
脚本中默认使用 `kernel_size=5` ($k=2$)。

### 2. 离散小波分解 (Discrete Wavelet Transform, DWT)
利用离散小波变换将清洗后的信号 $s[n]$ 分解为不同频段的系数。

**分解公式：**
$$a_{j+1}[n] = \sum_{k} h[k-2n] a_j[k]$$
$$d_{j+1}[n] = \sum_{k} g[k-2n] a_j[k]$$
其中：
- $a_j$ 为第 $j$ 层的近似系数 (Approximation Coefficients)。
- $d_j$ 为第 $j$ 层的细节系数 (Detail Coefficients)。
- $h[k]$ 和 $g[k]$ 分别为低通和高通滤波器。

脚本默认进行 2 层分解 ($level=2$)，得到 $cA_2$ (低频) 和 $cD_2, cD_1$ (高频)。

### 3. 信号重构与分频段 (Signal Reconstruction)
通过逆离散小波变换 (IDWT) 重构出低频分量 $S_{low}$ 和高频分量 $S_{high}$。

**重构公式：**
$$S = A_J + \sum_{j=1}^J D_j$$
- **低频分量 ($S_{low}$)**：反映信号的宏观趋势，如电器的开关状态切换。
- **高频分量 ($S_{high}$)**：捕捉信号的瞬态突变和细节。

### 4. ClaSPSegmentation 变点检测
对重构后的分量分别应用 `BinaryClaSPSegmentation` 算法。ClaSPSegmentation 通过计算 **分类得分剖面 (Classification Score Profile, ClasP)** 来识别信号中的语义变化点。

### 5. 变点合成 (Changepoint Synthesis)
将不同频段检测到的变点集进行映射和均值合成。

**合成逻辑：**
1. 选取点数较多的集合作为参考集 $CP_{ref}$（通常是低频或高频变点）。
2. 将其他集合中的点 $p$ 映射到 $CP_{ref}$ 中最近的索引 $i$。
3. 对每个索引 $i$ 对应的所有关联点进行平均，得到最终的合成变点 $CP_{synth}$：
$$CP_{synth, i} = \frac{1}{|G_i|} \sum_{p \in G_i} p$$
其中 $G_i$ 是映射到参考点 $i$ 的所有变点集合。

---

## 主要特性

1.  **多分辨率分析**：通过小波变换解耦趋势项和细节项，提高变点定位精度。
2.  **自动小波优选**：自动测试 `db1` 到 `db4` 小波，并根据检测到的变点数量进行排序，选择最优分析结果。
3.  **多维度可视化**：
    *   **4 面板分析图**：展示原始/清洗信号、低频/高频分量及各阶段变点的对比。
    *   **时频能量热图 (Scalogram)**：使用连续小波变换 (CWT) 展示低频信号的能量分布。
4.  **自动化导出**：一键生成处理后的信号数据 (`data/`) 和变点标注 (`label/`)，支持 CSV 格式。

---

## 依赖项

```bash
pip install numpy pandas PyWavelets matplotlib scipy claspy
```

---

## 使用说明

1.  **路径配置**：
    修改 `wavelet_separation.py` 中 `if __name__ == "__main__":` 下的路径：
    *   `input_source`: 输入 CSV 文件或包含多个 CSV 的目录。
    *   `output_directory`: 结果保存的主目录。

2.  **运行脚本**：
    ```bash
    python wavelet_separation.py
    ```

3.  **关键参数说明**：
    *   `apply_diff`: (bool) 是否在处理前对信号应用一阶差分。
    *   `is_plot`: (bool) 是否生成 PNG 可视化图表。
    *   `n_plots`: (int) 为每个文件生成排名最高的前 n 个小波结果图。

---

## 目录结构

运行后，`output_directory` 将生成：
- `data/`: 处理后的 CSV 信号（包含 `power`, `cleaned_power`, `high_freq`, `low_freq`）。
- `label/`: 包含变点索引及其类型的 CSV 文件。
- `*.png`: 详细的分析图和小波热图。
