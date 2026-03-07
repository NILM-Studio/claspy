# FLUSS 与 Wavelet & Clasp 切分效果对比说明

该文件夹包含用于时间序列切分的 FLUSS 算法实现及其与 Wavelet & Clasp 算法结果的对比工具。

## 目录结构
- `fluss.py`: 基于 FLUSS 算法的批量切分脚本。
- `compare_labels.py`: 对比 FLUSS 与 Wavelet & Clasp 切分结果的可视化脚本。
- `outputs/`: 存放切分结果和对比图表。
  - `label/`: 存放 FLUSS 生成的 changepoint 标签文件（CSV）。

---

## 步骤 1: 使用 FLUSS 进行时间序列切分

运行 `fluss.py` 对原始洗衣机功率数据进行批量切分。

### 1.1 输入与输出
- **输入数据**: `f:\B__ProfessionProject\NILM\Clasp\data\washing_machine` 中的原始 CSV 文件。
- **输出路径**: `f:\B__ProfessionProject\NILM\Clasp\comparison\outputs\label`。
- **输出格式**: `Changepoints_{原始文件名}.csv`。

### 1.2 参数设置 (在 `if __name__ == "__main__"` 中)
- `data_dir`: 原始数据存放目录。
- `output_dir`: 结果保存目录。
- `n`: 处理文件的数量（默认 10）。
- `window_size`: 窗口大小（默认 20）。
- `n_regimes`: 期望分段数（默认 3，即检测 2 个切分点）。
- `excl_factor`: 排除区域因子（默认 1）。

### 1.3 执行命令
```bash
python f:\B__ProfessionProject\NILM\Clasp\comparison\fluss.py
```

---

## 步骤 2: 生成效果对比图表

运行 `compare_labels.py` 将 FLUSS 的切分结果与 Wavelet & Clasp 的结果进行可视化对比。

### 2.1 输入与输出
- **FLUSS 标签**: `f:\B__ProfessionProject\NILM\Clasp\comparison\outputs\label`。
- **Wavelet 标签**: `f:\B__ProfessionProject\NILM\Clasp\wavelet_clasp_segmentation\outputs\result7\label`。
- **原始数据**: `f:\B__ProfessionProject\NILM\Clasp\mean_reversion(out-of-date)\project\washing_machine\related\data`。
- **输出图表**: `f:\B__ProfessionProject\NILM\Clasp\comparison\outputs\comparison_result_n{n}.png`。

### 2.2 参数设置 (在 `if __name__ == "__main__"` 中)
- `n`: 选择进行对比展示的文件数量（默认 3）。
- `label_type`: 在 Wavelet 标签中过滤 `label_type == 0` 的记录（代码已硬编码处理）。

### 2.3 执行命令
```bash
python f:\B__ProfessionProject\NILM\Clasp\comparison\compare_labels.py
```

---

## 注意事项
1. **执行顺序**: 必须先执行 `fluss.py` 生成标签文件，然后再执行 `compare_labels.py` 进行对比，否则对比脚本将找不到匹配的标签。
2. **环境依赖**: 确保已安装 `stumpy`, `pandas`, `matplotlib`, `numpy` 等库。
3. **路径配置**: 若原始数据或 Wavelet 结果路径发生变动，请在脚本的 `if __name__ == "__main__"` 部分手动更新。
