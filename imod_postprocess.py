#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
对cryoET的tomogram进行后处理的脚本
支持根据CSV文件中的信息对tomogram进行Z轴范围截取、维度交换、下采样等操作
"""

import argparse
import csv
import os
import numpy as np
from scipy.ndimage import zoom
import mrcfile


def parse_range(range_str):
    """
    解析范围字符串，例如 "120-353" 返回 (120, 353)
    """
    parts = range_str.split('-')
    return int(parts[0]), int(parts[1])


def swap_dimensions(data, swap_option):
    """
    交换数据的维度
    MRC文件中数据排布为 (z, y, x)
    """
    if swap_option == 'yz':
        # 交换Y和Z轴 (z,y,x) -> (y,z,x)
        return np.transpose(data, axes=(1, 0, 2))
    elif swap_option == 'xy':
        # 交换X和Y轴 (z,y,x) -> (z,x,y)
        return np.transpose(data, axes=(0, 2, 1))
    elif swap_option == 'xz':
        # 交换X和Z轴 (z,y,x) -> (x,y,z)
        return np.transpose(data, axes=(2, 1, 0))
    else:
        # 默认只交换Y和Z轴 (z,y,x) -> (y,z,x)
        return np.transpose(data, axes=(1, 0, 2))


def bin_data_3d(data):
    """
    对三维数据进行金字塔下采样（每个维度缩小2倍）
    """
    zoom_factors = (0.5, 0.5, 0.5)
    binned_data = zoom(data, zoom_factors, order=1)  # 使用线性插值
    return binned_data


def process_tomogram(input_file, output_file, z_range, z_shift_value=None, bin2=False, swap_option=None):
    """
    处理单个tomogram文件
    """
    # 读取MRC文件
    with mrcfile.open(input_file, permissive=True) as mrc:
        data = mrc.data.copy()
        voxel_size = mrc.voxel_size

    original_shape = data.shape
    print(f"原始数据形状: {original_shape}")

    # 如果指定了swap操作，先交换维度
    if swap_option is not None:
        print(f"交换维度: {swap_option}")
        data = swap_dimensions(data, swap_option)
        current_shape = data.shape
        print(f"交换后数据形状: {current_shape}")
    else:
        current_shape = original_shape

    # 计算Z轴范围
    original_z_size = current_shape[0]  # 当前Z轴大小（可能因为交换维度而改变）
    
    z_min, z_max = z_range
    
    # 如果有z-shift参数，调整Z轴范围
    if z_shift_value is not None:
        adjustment = (z_shift_value - original_z_size) // 2
        z_min = z_min - adjustment
        z_max = z_max - adjustment
        print(f"应用z-shift调整，调整值: {adjustment}, 调整后范围: {z_min}-{z_max}")
    
    # 确保范围在有效范围内
    z_min = max(0, z_min)
    z_max = min(current_shape[0], z_max)
    
    print(f"截取Z轴范围: {z_min} 到 {z_max}")
    
    # 截取Z轴范围
    data = data[z_min:z_max, :, :]
    
    print(f"截取后数据形状: {data.shape}")

    # 如果需要bin2下采样
    if bin2:
        print("执行3维金字塔下采样")
        data = bin_data_3d(data)
        print(f"下采样后数据形状: {data.shape}")

    # 保存处理后的数据
    with mrcfile.new(output_file, overwrite=True) as output_mrc:
        output_mrc.set_data(data)
        if voxel_size is not None:
            output_mrc.voxel_size = voxel_size



def main():
    parser = argparse.ArgumentParser(description='对cryoET的tomogram进行后处理')
    parser.add_argument('csv_file', help='包含文件名和Z轴范围的CSV文件路径')
    parser.add_argument('--z-shift', type=int, help='Z轴范围调整参数')
    parser.add_argument('--bin2', action='store_true', help='在Z截取后进行3维金字塔下采样')
    parser.add_argument('--swap', nargs='?', const='yz', 
                       choices=['yz', 'xy', 'xz'], 
                       help='交换维度 (默认交换Y和Z), 可选: xy, xz, yz')
    parser.add_argument('--prefix', type=str, help='输出文件名前缀')
    parser.add_argument('--suffix', type=str, help='输出文件名后缀')
    
    args = parser.parse_args()

    # 读取CSV文件
    with open(args.csv_file, 'r', newline='') as csvfile:
        # 检查是否为tab分隔
        sample = csvfile.read(1024)
        csvfile.seek(0)
        if '\t' in sample:
            delimiter = '\t'
        else:
            delimiter = ','
        
        reader = csv.reader(csvfile, delimiter=delimiter)
        rows = list(reader)

    for row in rows:
        if len(row) < 2:
            continue
            
        input_filename = row[0].strip()
        z_range_str = row[1].strip()
        
        # 如果文件名没有.mrc后缀，添加后缀
        if not input_filename.lower().endswith('.mrc'):
            input_filename = input_filename + '.mrc'
        
        # 解析Z轴范围
        z_range = parse_range(z_range_str)
        
        # 生成输出文件名
        name_part = os.path.splitext(input_filename)[0]
        ext_part = os.path.splitext(input_filename)[1]
        
        # 根据prefix和suffix参数生成输出文件名
        if args.prefix is not None and args.suffix is not None:
            output_filename = f"{args.prefix}{name_part}{args.suffix}{ext_part}"
        elif args.prefix is not None:
            output_filename = f"{args.prefix}{name_part}{ext_part}"
        elif args.suffix is not None:
            output_filename = f"{name_part}{args.suffix}{ext_part}"
        else:
            # 默认使用_processed后缀
            output_filename = name_part + '_processed' + ext_part
        
        print(f"处理文件: {input_filename}")
        print(f"输出文件: {output_filename}")
        print(f"Z轴范围: {z_range_str}")
        
        # 检查输入文件是否存在
        if not os.path.exists(input_filename):
            print(f"错误: 输入文件不存在 - {input_filename}")
            continue
        
        # 处理tomogram
        process_tomogram(
            input_file=input_filename,
            output_file=output_filename,
            z_range=z_range,
            z_shift_value=args.z_shift,
            bin2=args.bin2,
            swap_option=args.swap
        )
        
        print(f"完成处理: {input_filename}\n")


if __name__ == '__main__':
    main()
