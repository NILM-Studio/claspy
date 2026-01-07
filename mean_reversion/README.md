# Mean Reversion & Segmentation Workflow

本项目实现了一套针对时间序列数据的自适应预处理与分割工作流。主要用于非侵入式负荷监测（NILM）场景下，针对不同特征的设备数据（特别是具有多状态的设备），先通过聚类分析判断是否需要进行均值回归（Mean Reversion）调整，然后再进行变化点检测（Change Point Detection）。

## 核心逻辑

整个工作流分为三个主要步骤，由 `main.py` 统一调度：

1.  **分布分析与阈值确定 (Step 1)**:
    *   计算所有样本数据的 K-Means 聚类 Silhouette Score（轮廓系数）。
    *   分析分数的概率密度分布（KDE），寻找密度极小值点作为 **Split Point（分割阈值）**，用于区分“强聚类特征数据”和“弱聚类特征数据”。

2.  **自适应均值回归与异常值处理 (Step 2)**:    
    *   根据 Step 1 计算的阈值，对原始数据进行处理。
    *   **预处理**: 对所有数据先进行滑动窗口异常值去除：
        - 使用基于滚动中位数的 Z-score 方法检测异常值
        - 支持多种插值方法（线性、多项式、样条、最近邻、零阶）处理异常值
        - 添加 `is_interpolated` 列标记插值点
    *   **Score > 阈值**: 判定为多状态数据：
        - 计算各聚类中心的均值
        - 将数据平移（`power - (cluster_mean - min_cluster_mean)`），消除不同状态间的基线差异
        - 处理后的数据命名为 `power_new`，使其更适合变化点检测
    *   **Score <= 阈值**: 判定为单状态或噪声数据：
        - 直接将经过异常值去除后的 `power` 赋值给 `power_new`

3.  **时间序列分割 (Step 3)**:
    *   使用 `claspy` 库对处理后的数据进行变化点检测（Segmentation）。
    *   采用两阶段策略：优先使用 `suss` 窗口策略，若未检测到变化点，回退到 `acf` 策略。
    *   输出包含变化点时间戳和索引的标签文件。

## 目录结构

```text
mean_reversion/
├── main.py                     # 工作流入口脚本 (Orchestrator)
├── silhouette_distribution.py  # Step 1: 分布分析脚本
├── group_adjust.py             # Step 2: 分组调整脚本
├── tsd.py                      # Step 3: 时间序列分割脚本
└── project/
    └── project/
        ├── related/
        │   ├── data/           # [输入] 原始 CSV 数据文件
        │   ├── plot/           # [输出] 分布分析图表 (分布图, KDE, 分割点)
        │   └── score/          # [输出] 中间结果 (Silhouette Scores, Split Point)
        ├── data/               # [输出] 经过均值回归处理后的中间数据
        └── label/              # [输出] 最终的变化点标签文件
```

## 环境依赖

*   Python 3.x
*   pandas
*   numpy
*   scikit-learn
*   matplotlib
*   seaborn
*   scipy
*   claspy (需确保已安装或在 Python 路径中)

## 使用方法
- 默认`<项目名称>`为`project`，如果需要自定义项目名称，需要在`main.py`中修改`project_dir`变量。
1.  **准备数据**: 将原始 CSV 数据放入 `/project/<项目名称>/related/data` 目录中。数据需包含 `power` 列（以及 `timestamp`, `datetime` 等用于输出的列）。
2.  **运行程序**: 在 `mean_reversion` 目录下运行主程序：

```bash
python main.py
```

3.  **查看结果**:
    *   **分布分析**: 在 `<项目名称>/related/plot` 查看分数分布图和分割示意图。
    *   **中间数据**: 在 `<项目名称>/data` 查看预处理后的数据（包含 `power_new` 列）。
    *   **最终标签**: 在 `<项目名称>/label` 查看生成的分割点标签文件 (`Changepoints_*.csv`)。

## 脚本详细说明

### `main.py`
工作流的总指挥。它依次调用以下三个模块，并在模块间传递参数（如动态计算出的 Split Point）。

### `silhouette_distribution.py`
*   **功能**: 遍历 `<项目名称>/related/data` 下的所有 CSV 文件，对 `power` 数据进行 K-Means 聚类（K=2~10），选取最佳 Silhouette Score。
*   **核心算法**: 使用 Kernel Density Estimation (KDE) 拟合分数的分布，并利用 `scipy.signal.argrelextrema` 寻找概率密度的局部极小值，将其作为区分不同数据类型的天然阈值。

### `group_adjust.py`
*   **功能**: 读取原始数据，进行滑动窗口异常值去除，然后根据传入的阈值进行均值回归调整。
*   **核心步骤**:
    1.  **异常值去除**: 
        *   使用基于滚动中位数的 Z-score 方法检测异常值
        *   支持多种插值方法（线性、多项式、样条、最近邻、零阶）处理异常值
        *   添加 `is_interpolated` 列标记插值点
        *   列名调整：原始 power 列重命名为 `power_origin`，处理后的干净数据命名为 `power`
    
    2.  **均值回归调整**:
        *   如果文件的最佳 Silhouette Score 高于阈值，说明数据有明显的稳态台阶：
            *   计算每个聚类的均值
            *   将所有聚类“拉平”到最低能量聚类的水平 (`power_new = power - (cluster_mean - min_cluster_mean)`)
            *   消除不同状态间的基线差异，突出变化瞬间
        *   如果分数低于阈值，直接将干净的 `power` 赋值给 `power_new`
        
*   **输出**: 处理后的数据包含以下列：
    *   `power_origin`: 原始功率数据
    *   `power`: 经过异常值去除后的干净数据
    *   `power_new`: 经过均值回归调整后的数据
    *   `is_interpolated`: 标记是否为插值点 (True/False)

### `tsd.py`
*   **功能**: 读取 `<项目名称>/data` 中经过均值回归处理的数据，执行变化点检测。
*   **策略**:
    *   默认使用 `claspy.segmentation.BinaryClaSPSegmentation`，参数 `window_size="suss"`。
    *   如果 `suss` 策略未检测到任何变化点，自动回退到 `window_size="acf"` 策略再次尝试。
    *   结果保存为 CSV，包含变化点的时间索引。
