# FLUSS 批量切分模型设计说明 (Model Description)

## 1. 概述
`fluss.py` 是一个基于 **FLUSS (Fast Low-cost Unsupervised Semantic Segmentation)** 算法的时间序列切分工具。该模型专门设计用于 NILM（非侵入式负荷监测）场景下的电器功率序列切分，旨在自动识别电器运行状态的切换点。

为了提高切分的准确性，本模型引入了**动态参数调整机制**，通过参考 Wavelet & Clasp 算法的预切分结果来自动配置 FLUSS 的核心参数。

---

## 2. 核心算法：FLUSS
FLUSS 算法的核心流程如下：
1.  **Matrix Profile (矩阵轮廓) 计算**：使用 `stumpy.stump` 计算时间序列的 Matrix Profile 和 Matrix Profile Index (MPI)。
2.  **Arc Curve (弧曲线) 计算**：基于 MPI 计算弧曲线，记录每个子序列与其最近邻之间的跨越关系。
3.  **Corrected Arc Curve (CAC, 校正弧曲线) 计算**：对弧曲线进行校正，消除序列边缘效应。CAC 曲线的极小值点即为潜在的状态切换边界（Changepoints）。
4.  **分割点有效性过滤**：为了过滤掉无效的分割结果，模型会排除索引为 0 的点。若最终未检测到任何大于 0 的有效分割点，则不生成输出文件。

---

## 3. 动态参数设计逻辑
本模型不使用固定的静态参数，而是根据参考标签（来自 Wavelet & Clasp 结果）动态计算：

### 3.1 动态设置 `n_regimes` (分段数)
-   **参考源**：读取 `all_machine/{电器}/label/*.csv` 中 `label_type == 0` 的标签。
-   **逻辑**：若参考标签个数为 $k$，则设置 `n_regimes = k + 1`。
-   **目的**：确保 FLUSS 检测到的分割点数量与基准算法一致，便于横向对比效果。

### 3.2 动态设置 `excl_factor` (排除因子)
-   **计算逻辑**：
    1.  根据参考标签位置计算所有分段的宽度（Widths）。
    2.  找到最小分段宽度 `min_width`。
    3.  设置 `excl_factor = max(1, min_width // (2 * window_size))`。
-   **目的**：动态调整最小分割点间隔。确保 FLUSS 的排除区域（Exclusion Zone）不会覆盖掉较短的有效状态段，从而提高对短时运行状态的敏感度。

---

## 4. 批量处理架构
模型支持全电器自动遍历处理：
1.  **多电器支持**：自动扫描 `data/` 目录下的所有电器子目录（如 microwave, dishwasher 等）。
2.  **断点续传**：检查输出目录，自动跳过已生成的 `Changepoints_*.csv` 文件。
3.  **零段优化**：若参考标签显示该序列无显著分段（$n\_regimes \le 1$），或 FLUSS 运行后未检测到大于 0 的有效分割点，则直接跳过，不生成结果文件。

---

## 5. 输入输出规格
-   **输入 (Data)**：包含 `timestamp`, `power`, `datetime` 列的原始功率序列 CSV。
-   **输入 (Reference Labels)**：Wavelet & Clasp 生成的包含 `label_type` 和 `changepoint_index` 的 CSV。
-   **输出 (FLUSS Labels)**：包含 `timestamp`, `power`, `datetime`, `changepoint_index` 的 CSV 结果文件。

---

## 6. 关键代码参考
-   核心切分函数：`fluss(ts, window_size, n_regimes, excl_factor)`
-   可视化函数：`fluss_visualize(...)`
-   参数计算逻辑：见 `if __name__ == "__main__"` 中的动态调整部分。
