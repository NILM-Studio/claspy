import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']  # 解决中文显示问题
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 读取洗衣机数据
data_path = r'f:\B__ProfessionProject\NILM\Clasp\mean_reversion\project\washing_machine\related\data'
data_files = os.listdir(data_path)

# 使用第一个数据文件
data_file = data_files[0]
file_path = os.path.join(data_path, data_file)
print(f"使用数据文件: {data_file}")

# 读取CSV数据
df = pd.read_csv(file_path)

# 提取功率数据
power = df['power'].values

# 执行傅里叶变换
fft_result = np.fft.fft(power)

# 计算频率轴
n = len(power)
sample_rate = 1  # 假设采样频率为1Hz（根据实际数据调整）
frequencies = np.fft.fftfreq(n, d=1/sample_rate)

# 计算振幅谱
amplitude = np.abs(fft_result) / n

# 只取正频率部分
positive_freqs = frequencies[:n//2]
positive_amps = amplitude[:n//2]

# 显示主要频率分量
top_indices = np.argsort(positive_amps)[-5:][::-1]  # 取前5个最大振幅的频率

# 绘制原始功率数据和频谱
plt.figure(figsize=(15, 10))

# 原始功率数据
plt.subplot(221)
plt.plot(power)
plt.title('原始功率数据')
plt.xlabel('时间 (采样点)')
plt.ylabel('功率 (W)')

# 绘制频谱
plt.subplot(222)
plt.plot(positive_freqs, positive_amps)
plt.title('傅里叶变换频谱')
plt.xlabel('频率 (Hz)')
plt.ylabel('振幅')
plt.grid(True)

# 选择一个主要频率分量进行正弦余弦波分离（跳过直流分量，选择第一个交流分量）
if len(top_indices) > 1:  # 确保有交流分量
    main_freq_idx = top_indices[1]  # 选择振幅最大的交流分量
    main_freq = positive_freqs[main_freq_idx]
    
    # 计算正弦和余弦分量
    t = np.arange(n)
    phase = np.angle(fft_result[main_freq_idx])  # 相位角
    amplitude = positive_amps[main_freq_idx] * 2  # 乘以2是因为我们只取了正频率部分
    
    # 重建正弦和余弦分量
    cos_wave = amplitude * np.cos(2 * np.pi * main_freq * t + phase)
    sin_wave = amplitude * np.sin(2 * np.pi * main_freq * t + phase)
    
    # 绘制正弦分量
    plt.subplot(223)
    plt.plot(t, sin_wave)
    plt.title(f'正弦分量 (频率: {main_freq:.4f} Hz)')
    plt.xlabel('时间 (采样点)')
    plt.ylabel('振幅 (W)')
    plt.grid(True)
    
    # 绘制余弦分量
    plt.subplot(224)
    plt.plot(t, cos_wave)
    plt.title(f'余弦分量 (频率: {main_freq:.4f} Hz)')
    plt.xlabel('时间 (采样点)')
    plt.ylabel('振幅 (W)')
    plt.grid(True)

plt.tight_layout()
plt.savefig('fourier_transform_result.png')
print("傅里叶变换完成，结果图已保存为 fourier_transform_result.png")

# 显示主要频率分量
print("\n主要频率分量：")
for idx in top_indices:
    freq = positive_freqs[idx]
    amp = positive_amps[idx]
    print(f"频率: {freq:.4f} Hz, 振幅: {amp:.4f}")
