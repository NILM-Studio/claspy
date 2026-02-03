import pandas as pd
import numpy as np
from typing import List, Tuple, Optional, Union
import os
import gc
from datetime import datetime


class ApplianceDataSegmenter:
    def __init__(self, appliance_name: str, power_threshold: float = 1.0,
                 min_duration_seconds: int = 30,
                 context_seconds: int = 120):
        """
        电器数据切割器

        Args:
            appliance_name: 电器名称，用于文件命名
            power_threshold: 功率阈值，用于检测工作状态开始和结束
            min_duration_seconds: 最小持续时间(秒)，避免噪声误判
            context_seconds: 上下文时间(秒)，在工作区间前后额外包含的数据
        """
        self.appliance_name = appliance_name
        self.power_threshold = power_threshold
        self.min_duration_seconds = min_duration_seconds
        self.context_seconds = context_seconds

    def process_dataset(self, input_file: str, output_dir: str) -> List[str]:
        """
        处理数据集并分割工作区间

        Args:
            input_file: 输入文件路径，支持 .dat、.csv 和 .npy 格式
            output_dir: 输出目录

        Returns:
            生成的CSV文件路径列表
        """
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)

        print(f"开始处理电器数据: {self.appliance_name}")
        print(f"输入文件: {input_file}")
        print(f"输出目录: {output_dir}")
        print(f"功率阈值: {self.power_threshold}")
        print(f"最小持续时间: {self.min_duration_seconds}秒")
        print(f"上下文: ±{self.context_seconds}秒")
        print("-" * 50)

        # 第一步：读取数据
        print("第一步：读取数据...")
        data = self._read_data(input_file)
        print(f"成功读取 {len(data)} 个数据点")

        # 第二步：检测所有工作区间
        print("第二步：检测工作区间...")
        segments = self._detect_working_segments_from_data(data)

        print(f"检测到 {len(segments)} 个潜在工作区间")

        # 第三步：提取并保存工作区间数据
        print("第三步：提取并保存工作区间数据...")
        output_files = self._extract_all_segments_from_data(data, segments, output_dir)

        print(f"处理完成！共提取 {len(output_files)} 个工作区间")
        return output_files

    def _read_data(self, input_file: str) -> List[Tuple[int, float]]:
        """
        读取不同格式的数据文件

        Args:
            input_file: 输入文件路径

        Returns:
            数据列表，每个元素是 (timestamp, power) 元组
        """
        file_ext = os.path.splitext(input_file)[1].lower()
        
        print(f"检测到文件格式: {file_ext}")
        
        if file_ext == '.dat':
            return self._read_dat_file(input_file)
        elif file_ext == '.csv':
            return self._read_csv_file(input_file)
        elif file_ext == '.npy':
            return self._read_npy_file(input_file)
        else:
            raise ValueError(f"不支持的文件格式: {file_ext}，仅支持 .dat、.csv 和 .npy 格式")

    def _read_dat_file(self, input_file: str) -> List[Tuple[int, float]]:
        """
        读取 .dat 文件

        Args:
            input_file: 输入 .dat 文件路径

        Returns:
            数据列表，每个元素是 (timestamp, power) 元组
        """
        data = []
        
        with open(input_file, 'r') as f:
            line_count = 0
            for line in f:
                line_count += 1
                if line_count % 1000000 == 0:
                    print(f"已读取 {line_count} 行数据...")

                parts = line.strip().split()
                if len(parts) < 2:
                    continue

                try:
                    timestamp = int(parts[0])
                    power = float(parts[1])
                    data.append((timestamp, power))
                except (ValueError, IndexError):
                    continue
        
        return data

    def _read_csv_file(self, input_file: str) -> List[Tuple[int, float]]:
        """
        读取 .csv 文件

        Args:
            input_file: 输入 .csv 文件路径

        Returns:
            数据列表，每个元素是 (timestamp, power) 元组
        """
        try:
            df = pd.read_csv(input_file)
            # 确保列名正确
            if 'timestamp' in df.columns and 'power' in df.columns:
                data = [(int(row['timestamp']), float(row['power'])) for _, row in df.iterrows()]
            else:
                # 假设第一列是 timestamp，第二列是 power
                data = [(int(row[0]), float(row[1])) for _, row in df.iterrows()]
            return data
        except Exception as e:
            raise ValueError(f"读取 CSV 文件失败: {e}")

    def _read_npy_file(self, input_file: str) -> List[Tuple[int, float]]:
        """
        读取 .npy 文件

        Args:
            input_file: 输入 .npy 文件路径

        Returns:
            数据列表，每个元素是 (timestamp, power) 元组
        """
        try:
            data_array = np.load(input_file)
            # 确保数据格式正确
            if data_array.ndim == 2 and data_array.shape[1] >= 2:
                data = [(int(row[0]), float(row[1])) for row in data_array]
                return data
            else:
                raise ValueError(f"NPY 文件格式不正确，需要至少两列数据")
        except Exception as e:
            raise ValueError(f"读取 NPY 文件失败: {e}")

    def _detect_working_segments_from_data(self, data: List[Tuple[int, float]]) -> List[Tuple[int, int, int, int, int]]:
        """
        从数据列表中检测所有工作区间

        Args:
            data: 数据列表，每个元素是 (timestamp, power) 元组

        Returns:
            列表格式: [(start_idx, end_idx, start_time, end_time, duration), ...]
        """
        segments = []
        current_segment_start = None
        current_segment_start_time = None
        consecutive_above_count = 0

        print("正在检测工作区间...")

        for i, (timestamp, power) in enumerate(data):
            if i % 1000000 == 0:
                print(f"已处理 {i} 个数据点...")

            if power >= self.power_threshold:
                if current_segment_start is None:
                    current_segment_start = i  # 0-based index
                    current_segment_start_time = timestamp
                consecutive_above_count += 1
            else:
                # 检查是否结束了一个有效的工作区间
                if (current_segment_start is not None and
                        consecutive_above_count >= self.min_duration_seconds):
                    segment_end = i - 1  # 上一个数据点是结束点
                    segment_end_time = timestamp
                    duration = consecutive_above_count
                    segments.append((
                        current_segment_start,
                        segment_end,
                        current_segment_start_time,
                        segment_end_time,
                        duration
                    ))

                current_segment_start = None
                current_segment_start_time = None
                consecutive_above_count = 0

        # 处理数据末尾可能的工作区间
        if (current_segment_start is not None and
                consecutive_above_count >= self.min_duration_seconds):
            segment_end = len(data) - 1
            segment_end_time = data[-1][0]
            duration = consecutive_above_count
            segments.append((
                current_segment_start,
                segment_end,
                current_segment_start_time,
                segment_end_time,
                duration
            ))

        print(f"检测完成，共处理 {len(data)} 个数据点")
        return segments

    def _extract_all_segments_from_data(self, data: List[Tuple[int, float]], segments: List[Tuple[int, int, int, int, int]],
                                      output_dir: str) -> List[str]:
        """从数据列表中提取所有检测到的工作区间数据"""
        output_files = []

        for seg_idx, (start_idx, end_idx, start_time, end_time, duration) in enumerate(segments):
            try:
                output_file = self._extract_single_segment_from_data(
                    data, start_idx, end_idx, start_time, end_time, duration,
                    output_dir, seg_idx
                )
                if output_file:
                    output_files.append(output_file)

                if (seg_idx + 1) % 10 == 0:
                    print(f"已提取 {seg_idx + 1}/{len(segments)} 个工作区间")

            except Exception as e:
                print(f"提取区间 {seg_idx} 时出错: {e}")
                continue

        return output_files

    def _extract_single_segment_from_data(self, data: List[Tuple[int, float]], start_idx: int, end_idx: int,
                                        start_time: int, end_time: int, duration: int,
                                        output_dir: str, segment_id: int) -> Optional[str]:
        """从数据列表中提取单个工作区间数据（包含上下文）"""
        # 计算包含上下文的边界
        context_start_idx = max(0, start_idx - self.context_seconds)
        context_end_idx = min(len(data) - 1, end_idx + self.context_seconds)

        # 提取数据
        segment_data = data[context_start_idx:context_end_idx + 1]

        # 生成文件名
        start_dt = datetime.fromtimestamp(start_time)
        end_dt = datetime.fromtimestamp(end_time)

        # 格式化时间字符串，用于文件名
        start_str = start_dt.strftime("%Y%m%d_%H%M%S")
        end_str = end_dt.strftime("%Y%m%d_%H%M%S")

        # 创建文件名：{电器名称_起始时间_结束时间_持续时长}
        filename = f"{self.appliance_name}_{start_str}_{end_str}_{duration}s.csv"
        output_file = os.path.join(output_dir, filename)

        # 如果文件已存在，跳过
        if os.path.exists(output_file):
            print(f"文件已存在，跳过: {filename}")
            return output_file

        print(f"提取区间 {segment_id}: {start_str} - {end_str}, 持续 {duration}秒")

        # 转换为DataFrame并保存
        if segment_data:
            df = pd.DataFrame(segment_data, columns=['timestamp', 'power'])

            # 添加可读时间列
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')

            # 保存CSV文件
            df.to_csv(output_file, index=False)

            print(f"  保存了 {len(df)} 个数据点到 {filename}")
            return output_file

        return None


def main():
    """
    主函数 - 在这里设置您的参数
    """
    # ========== 在这里设置您的参数 ==========

    # 电器名称 (用于文件命名)
    APPLIANCE_NAME = "fridge"  # 例如: fridge, washing_machine, air_conditioner

    # 输入文件路径
    INPUT_FILE = "D:\Code-Program\TimeVAE_modified\pre_data\Kettle.dat"  # 替换为您的.dat文件路径

    # 输出目录
    OUTPUT_DIR = "D:\Code-Program\TimeVAE_modified\data\{APPLIANCE_NAME}"  # 替换为您想要的输出目录

    # 三个超参数
    POWER_THRESHOLD = 30.0  # 功率阈值
    MIN_DURATION_SECONDS = 60  # 最小持续时间(秒)
    CONTEXT_SECONDS = 30  # 上下文时间(秒)

    # ========== 参数设置结束 ==========

    # 创建分割器实例
    segmenter = ApplianceDataSegmenter(
        appliance_name=APPLIANCE_NAME,
        power_threshold=POWER_THRESHOLD,
        min_duration_seconds=MIN_DURATION_SECONDS,
        context_seconds=CONTEXT_SECONDS
    )

    # 处理数据
    try:
        output_files = segmenter.process_dataset(INPUT_FILE, OUTPUT_DIR)
        print(f"\n成功提取 {len(output_files)} 个工作区间")
        print(f"所有文件保存在: {OUTPUT_DIR}")

    except Exception as e:
        print(f"处理过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


# 批量处理多个电器的函数
def batch_process_appliances(configs: List[dict]):
    """
    批量处理多个电器的数据

    Args:
        configs: 配置字典列表，每个字典包含:
            - appliance_name: 电器名称
            - input_file: 输入文件路径
            - output_dir: 输出目录
            - power_threshold: 功率阈值
            - min_duration_seconds: 最小持续时间
            - context_seconds: 上下文时间
    """
    results = {}

    for config in configs:
        print(f"\n处理电器: {config['appliance_name']}")
        print("=" * 50)

        segmenter = ApplianceDataSegmenter(
            appliance_name=config['appliance_name'],
            power_threshold=config['power_threshold'],
            min_duration_seconds=config['min_duration_seconds'],
            context_seconds=config['context_seconds']
        )

        try:
            output_files = segmenter.process_dataset(
                input_file=config['input_file'],
                output_dir=config['output_dir']
            )
            results[config['appliance_name']] = {
                'output_dir': config['output_dir'],
                'segments_count': len(output_files),
                'status': 'success'
            }
        except Exception as e:
            results[config['appliance_name']] = {
                'output_dir': config['output_dir'],
                'segments_count': 0,
                'status': f'error: {e}'
            }

    # 打印汇总结果
    print("\n批量处理完成!")
    print("=" * 50)
    for appliance, result in results.items():
        print(f"{appliance}: {result['segments_count']} 个区间, 状态: {result['status']}")

    return results


if __name__ == "__main__":
    # 运行单个电器处理
    main()

    # 如果需要批量处理多个电器，可以使用下面的示例
    # appliance_configs = [
    #     {
    #         'appliance_name': 'fridge',
    #         'input_file': 'path/to/fridge_data.dat',
    #         'output_dir': 'output/fridge_segments',
    #         'power_threshold': 1.0,
    #         'min_duration_seconds': 30,
    #         'context_seconds': 120
    #     },
    #     {
    #         'appliance_name': 'washing_machine',
    #         'input_file': 'path/to/washing_machine_data.dat',
    #         'output_dir': 'output/washing_machine_segments',
    #         'power_threshold': 50.0,
    #         'min_duration_seconds': 600,  # 10分钟
    #         'context_seconds': 300
    #     }
    # ]
    #
    # batch_process_appliances(appliance_configs)