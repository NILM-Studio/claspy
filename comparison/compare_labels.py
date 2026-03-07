import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_comparison(fluss_label_dir, wavelet_label_dir, data_dir, n):
    """
    比较 FLUSS 和 Wavelet 分割结果的可视化函数
    """
    # 获取 FLUSS 和 Wavelet 文件夹中的所有 CSV 文件
    fluss_files = set(os.path.basename(f) for f in glob.glob(os.path.join(fluss_label_dir, "*.csv")))
    wavelet_files = set(os.path.basename(f) for f in glob.glob(os.path.join(wavelet_label_dir, "*.csv")))
    
    # 找到两个文件夹中共同存在的文件
    common_files = sorted(list(fluss_files.intersection(wavelet_files)))
    
    if not common_files:
        print("未发现共同的 label 文件，请检查路径。")
        return
    
    # 选择前 n 个文件
    selected_files = common_files[:n]
    actual_n = len(selected_files)
    
    # 设置中文字体（根据系统情况调整）
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 创建画布：2*actual_n 个子图，两列
    fig, axes = plt.subplots(actual_n, 2, figsize=(15, 4 * actual_n), squeeze=False)
    plt.suptitle(f'Comparison of label location between FLUSS and Wavelet & Clasp (n={actual_n})', fontsize=16, fontweight='bold')

    for i, file_name in enumerate(selected_files):
        # 1. 确定原始数据路径
        # 假设 label 文件名是 "Changepoints_" + original_filename
        original_file_name = file_name.replace("Changepoints_", "")
        data_path = os.path.join(data_dir, original_file_name)
        
        if not os.path.exists(data_path):
            print(f"找不到原始数据文件: {data_path}，跳过该文件的可视化。")
            continue
        
        # 读取原始数据
        try:
            df_data = pd.read_csv(data_path)
            ts = df_data['power'].values
        except Exception as e:
            print(f"读取原始数据出错: {e}")
            continue

        # 2. 读取 FLUSS Label
        fluss_label_path = os.path.join(fluss_label_dir, file_name)
        df_fluss = pd.read_csv(fluss_label_path)
        fluss_cps = df_fluss['changepoint_index'].values

        # 3. 读取 Wavelet Label
        wavelet_label_path = os.path.join(wavelet_label_dir, file_name)
        df_wavelet = pd.read_csv(wavelet_label_path)
        df_wavelet = df_wavelet[df_wavelet["label_type"] == 0]
        wavelet_cps = df_wavelet['changepoint_index'].values

        # 4. 绘图 - 左侧子图 (FLUSS)
        ax_left = axes[i, 0]
        ax_left.plot(ts, label='Power', color='gray', alpha=0.5)
        for cp in fluss_cps:
            ax_left.axvline(x=cp, color='red', linestyle='--', linewidth=1.5)
        ax_left.set_title(f"FLUSS: {original_file_name}")
        ax_left.set_ylabel("Power")
        ax_left.legend()

        # 5. 绘图 - 右侧子图 (Wavelet)
        ax_right = axes[i, 1]
        ax_right.plot(ts, label='Power', color='gray', alpha=0.5)
        for cp in wavelet_cps:
            ax_right.axvline(x=cp, color='blue', linestyle='--', linewidth=1.5)
        ax_right.set_title(f"Wavelet: {original_file_name}")
        ax_right.set_ylabel("Power")
        ax_right.legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    
    # 询问是否保存
    save_dir = os.path.dirname(fluss_label_dir) # 存在 comparison/outputs 文件夹
    save_path = os.path.join(save_dir, f'comparison_result_n{actual_n}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"对比图已保存至: {save_path}")
    plt.show()

if __name__ == "__main__":
    # 配置路径
    fluss_label_dir = r"f:\B__ProfessionProject\NILM\Clasp\comparison\outputs\label"
    wavelet_label_dir = r"f:\B__ProfessionProject\NILM\Clasp\wavelet_clasp_segmentation\outputs\result7\label"
    data_dir = r"f:\B__ProfessionProject\NILM\Clasp\mean_reversion(out-of-date)\project\washing_machine\related\data"
    
    # 设置变量n代表所选label个数
    n = 5
    
    plot_comparison(fluss_label_dir, wavelet_label_dir, data_dir, n)
