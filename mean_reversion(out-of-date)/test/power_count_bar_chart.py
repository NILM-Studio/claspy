import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import glob

# 设置数据路径
data_path = r'f:\B__ProfessionProject\NILM\Clasp\mean_reversion\project\washing_machine\data'

# 获取所有CSV文件
csv_files = glob.glob(os.path.join(data_path, '*.csv'))
print(f"找到 {len(csv_files)} 个数据文件")

# 初始化一个空的DataFrame来存储所有power数据
all_power_data = pd.DataFrame()

# 读取所有文件并合并power列
for file in csv_files:
    try:
        df = pd.read_csv(file)
        # 检查是否包含'power'列
        if 'power' in df.columns:
            power_data = df[['power']].copy()
            all_power_data = pd.concat([all_power_data, power_data], ignore_index=True)
        else:
            print(f"警告: 文件 {os.path.basename(file)} 不包含'power'列")
    except Exception as e:
        print(f"读取文件 {os.path.basename(file)} 时出错: {e}")

# 检查是否有数据
if all_power_data.empty:
    print("没有找到有效的power数据")
else:
    print(f"总共读取了 {len(all_power_data)} 条power数据")
    
    # 对power列进行计数统计
    power_counts = all_power_data['power'].value_counts().sort_index()
    
    # 绘制条形统计图
    plt.figure(figsize=(15, 8))
    
    # 设置条形图的宽度
    bar_width = 0.8
    
    # 绘制条形图
    bars = plt.bar(power_counts.index, power_counts.values, width=bar_width, color='skyblue')

    # 在每个柱状图上方标注power数值
    for x, y in zip(power_counts.index, power_counts.values):
        plt.text(x, y, str(x), ha='center', va='bottom', fontsize=8, rotation=90)
    
    # 设置图表标题和标签
    plt.title('洗衣机Power值计数统计', fontsize=16)
    plt.xlabel('Power值 (W)', fontsize=14)
    plt.ylabel('计数', fontsize=14)
    
    # 添加网格线
    plt.grid(True, axis='y', alpha=0.3)
    
    # 设置x轴刻度标签的旋转角度，避免重叠
    plt.xticks(rotation=45, ha='right')
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图表
    output_path = os.path.join(os.path.dirname(__file__), 'power_count_bar_chart.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"条形统计图已保存到: {output_path}")
    
    # 显示图表
    plt.show()