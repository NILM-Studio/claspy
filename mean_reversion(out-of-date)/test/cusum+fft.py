import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import medfilt
from scipy.fftpack import fft

# -------------------------- 1. 数据预处理：降噪+标准化 --------------------------
def preprocess_signal(signal, win_size=3):
    """中值滤波降噪 + Z-score标准化，不破坏阶跃突变特征"""
    # 中值滤波去脉冲噪声
    signal_denoise = medfilt(signal, kernel_size=win_size)
    # Z-score标准化
    mu = np.mean(signal_denoise)
    sigma = np.std(signal_denoise)
    signal_norm = (signal_denoise - mu) / sigma
    return signal_norm, mu, sigma

# -------------------------- 2. 双向CUSUM算法：检测所有阶跃突变点 --------------------------
def cusum_step_detect(signal, threshold=3.0, drift=0.1):
    """
    双向CUSUM，检测正向阶跃（上升）+ 负向阶跃（下降）
    :param signal: 预处理后的标准化信号
    :param threshold: 突变阈值，越大越严格，推荐2.5~4.0
    :param drift: 漂移量，抑制噪声，推荐0.05~0.2
    :return: 所有阶跃突变点的时间戳列表
    """
    n = len(signal)
    s_pos = np.zeros(n)  # 正向累积和：检测上升阶跃
    s_neg = np.zeros(n)  # 负向累积和：检测下降阶跃
    change_points = []
    baseline = signal[0]  # 初始基准值

    for i in range(1, n):
        delta = signal[i] - baseline
        # 更新累积和
        s_pos[i] = max(0, s_pos[i-1] + delta - drift)
        s_neg[i] = max(0, s_neg[i-1] - delta - drift)
        # 超过阈值，判定为突变点
        if s_pos[i] > threshold or s_neg[i] > threshold:
            change_points.append(i)
            # 重置基准值为当前值，适配新稳态
            baseline = signal[i]
            s_pos[i] = 0
            s_neg[i] = 0
    return np.array(change_points)

# -------------------------- 3. FFT算法：提取突变点的周期特征 --------------------------
def fft_period_detect(change_points, fs=1):
    """
    对突变点序列做FFT，提取周期特征
    :param change_points: CUSUM检测出的突变点时间戳
    :param fs: 采样频率，单位：Hz（根据你的信号设置，如1s一个点则fs=1）
    :return: 突变周期T、是否有显著周期性、周期性标签
    """
    if len(change_points) < 3:
        return 0, False, np.array([])  # 突变点太少，无周期可言
    
    # 计算相邻突变点的时间间隔
    delta_t = np.diff(change_points)
    # FFT变换：时域转频域
    n = len(delta_t)
    fft_vals = fft(delta_t)
    fft_freq = np.fft.fftfreq(n, d=1/fs)
    # 取正频率部分的幅值
    pos_mask = fft_freq > 0
    freq = fft_freq[pos_mask]
    amp = np.abs(fft_vals[pos_mask])
    # 找幅值最大的峰值频率
    peak_idx = np.argmax(amp)
    peak_freq = freq[peak_idx]
    # 计算周期
    T = int(1 / peak_freq) if peak_freq != 0 else 0
    # 验证周期性：间隔值接近T的占比≥70%则判定为有周期
    period_ratio = np.sum(np.abs(delta_t - T) < T*0.2) / len(delta_t)
    is_periodic = period_ratio >= 0.7
    # 标注突变点类型：周期性/随机
    period_label = np.array(['periodic' if abs(dt-T)<T*0.2 else 'random' for dt in delta_t])
    return T, is_periodic, period_label

# -------------------------- 4. 突变点优化 + 时间序列分割 --------------------------
def signal_segment(signal, change_points, T, is_periodic):
    """
    优化突变点，生成最终的时间分割结果
    :return: 分割区间列表，每个元素为[start, end, type]
    """
    n = len(signal)
    # 优化：剔除间隔过小的冗余突变点
    if len(change_points) > 1:
        delta_t = np.diff(change_points)
        keep_idx = delta_t > T*0.2 if T !=0 else delta_t > 5
        change_points = change_points[np.concatenate([[True], keep_idx])]
    # 生成分割边界：起始0 + 突变点 + 结束n-1
    boundaries = np.concatenate([[0], change_points, [n-1]])
    # 生成分割结果
    segments = []
    for i in range(len(boundaries)-1):
        start = boundaries[i]
        end = boundaries[i+1]
        if i == 0:
            seg_type = '平稳段'
        elif i < len(change_points)+1:
            if is_periodic:
                seg_type = '周期性阶跃突变段'
            else:
                seg_type = '随机阶跃突变段'
        else:
            seg_type = '新稳态段'
        segments.append([start, end, seg_type])
    return segments

# -------------------------- 主函数：全流程执行 --------------------------
if __name__ == "__main__":
    # 从CSV文件导入power列作为信号
    csv_file = r"f:\B__ProfessionProject\NILM\Clasp\mean_reversion\project\washing_machine\related\data\Washing_Machine_20121110_182407_20121110_191850_463s.csv"
    df = pd.read_csv(csv_file)
    signal = df['power'].values
    n = len(signal)  # 信号长度

    # 全流程执行
    signal_norm, _, _ = preprocess_signal(signal)
    change_points = cusum_step_detect(signal_norm, threshold=3.0)
    T, is_periodic, _ = fft_period_detect(change_points, fs=1)
    segments = signal_segment(signal, change_points, T, is_periodic)

    # 打印结果
    print(f"检测到的阶跃突变点：{change_points}")
    print(f"突变周期：{T} 个采样点")
    print(f"是否有显著周期性：{is_periodic}")
    print("时间序列分割结果：")
    for seg in segments:
        print(f"区间[{seg[0]}, {seg[1]}] → {seg[2]}")

    # 绘图可视化
    plt.figure(figsize=(12,6))
    plt.plot(signal, label='原始信号', color='gray', alpha=0.7)
    plt.scatter(change_points, signal[change_points], color='red', s=50, label='阶跃突变点')
    for seg in segments:
        plt.axvspan(seg[0], seg[1], alpha=0.1, color='blue' if seg[2]=='平稳段' else 'orange')
    plt.legend()
    plt.title('CUSUM+FFT 周期性阶跃突变检测与分割')
    plt.show()