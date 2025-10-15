#!/usr/bin/env python3
"""
ND2/TIFF文件通道帧提取脚本
将指定目录内所有nd2/tif文件的指定通道指定帧提取出来保存为tif格式
默认为第一个通道的第1帧
支持shell风格的通配符匹配
"""

import os
import sys
import argparse
from pathlib import Path
import numpy as np
from nd2 import ND2File
import tifffile
from PIL import Image
import glob


def extract_frame_from_nd2(nd2_path, output_dir, channel_index=0, frame_index=0):
    """
    从单个ND2文件中提取指定通道和帧
    
    Args:
        nd2_path: ND2文件路径
        output_dir: 输出目录
        channel_index: 通道索引（默认0）
        frame_index: 帧索引（默认0）
    """
    try:
        with ND2File(nd2_path) as nd2_file:
            # 获取图像信息
            print(f"处理ND2文件: {nd2_path.name}")
            print(f"  - 尺寸: {nd2_file.sizes}")
            print(f"  - 形状: {nd2_file.shape}")
            
            # 获取维度信息
            sizes = nd2_file.sizes
            shape = nd2_file.shape
            
            # 检查维度并获取各维度的大小
            time_dim = sizes.get('T', 1)
            channel_dim = sizes.get('C', 1)
            
            print(f"  - 时间点数: {time_dim}")
            print(f"  - 通道数: {channel_dim}")
            
            # 检查通道索引是否有效
            if channel_index >= channel_dim:
                print(f"  - 警告: 指定通道索引 {channel_index} 超出范围 (0-{channel_dim-1})")
                channel_index = 0
                print(f"  - 使用默认通道: {channel_index}")
            
            # 检查帧索引是否有效
            if frame_index >= time_dim:
                print(f"  - 警告: 指定帧索引 {frame_index} 超出范围 (0-{time_dim-1})")
                frame_index = 0
                print(f"  - 使用默认帧: {frame_index}")
            
            # 读取整个数据并切片
            all_data = nd2_file.asarray()
            
            # 根据维度顺序进行切片
            slicing = [slice(None)] * len(shape)
            
            # 找到时间维度和通道维度的索引
            dim_order = list(sizes.keys())
            
            for i, dim_name in enumerate(dim_order):
                if dim_name == 'T':
                    slicing[i] = frame_index
                elif dim_name == 'C':
                    slicing[i] = channel_index
            
            image_data = all_data[tuple(slicing)]
            
            # 确保数据是numpy数组
            if not isinstance(image_data, np.ndarray):
                image_data = np.array(image_data)
            
            # 获取原始文件名（不带扩展名）
            base_name = nd2_path.stem
            channel_name = f"channel_{channel_index}"
            frame_name = f"frame_{frame_index}"
            
            # 构建输出文件名
            output_filename = f"{base_name}_{channel_name}_{frame_name}.tif"
            output_path = output_dir / output_filename
            
            print(f"  - 尝试保存到: {output_path}")
            
            # 检查输出目录的写权限
            if not os.access(output_dir, os.W_OK):
                print(f"  - 错误: 没有写入目录 {output_dir} 的权限")
                return False
            
            # 保存为TIFF格式
            try:
                if image_data.dtype == np.uint16:
                    tifffile.imwrite(output_path, image_data, dtype=np.uint16)
                else:
                    # 如果不是uint16，转换为合适的格式
                    if image_data.max() <= 255:
                        image_data = image_data.astype(np.uint8)
                        tifffile.imwrite(output_path, image_data, dtype=np.uint8)
                    else:
                        image_data = image_data.astype(np.uint16)
                        tifffile.imwrite(output_path, image_data, dtype=np.uint16)
                
                print(f"  - 已保存: {output_path.name}")
                print(f"  - 图像尺寸: {image_data.shape}")
                print(f"  - 数据类型: {image_data.dtype}")
                print(f"  - 像素范围: {image_data.min()} - {image_data.max()}")
                return True
                
            except PermissionError as pe:
                print(f"  - 权限错误: {str(pe)}")
                print(f"  - 请检查目录权限，可能需要使用 'chmod' 命令修改权限")
                return False
            except Exception as e:
                print(f"  - 保存文件时出错: {str(e)}")
                return False
            
    except Exception as e:
        print(f"处理文件 {nd2_path} 时出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def extract_frame_from_tif(tif_path, output_dir, channel_index=0, frame_index=0):
    """
    从单个TIFF文件中提取指定通道和帧
    
    Args:
        tif_path: TIFF文件路径
        output_dir: 输出目录
        channel_index: 通道索引（默认0）
        frame_index: 帧索引（默认0）
    """
    try:
        # 读取TIFF文件
        image_data = tifffile.imread(str(tif_path))
        
        print(f"处理TIFF文件: {tif_path.name}")
        print(f"  - 原始形状: {image_data.shape}")
        
        # 获取维度信息
        shape = image_data.shape
        
        # 处理不同维度的数据
        if len(shape) == 2:  # 二维图像 (Y, X)
            if channel_index > 0 or frame_index > 0:
                print(f"  - 警告: 二维图像没有额外的通道或帧维度，使用原始图像")
            extracted_data = image_data
        elif len(shape) == 3:  # 三维图像 (Z, Y, X) 或 (C, Y, X) 或 (T, Y, X)
            # 假设第三维是通道或时间
            if shape[0] > 4:  # 可能是时间序列
                max_frame = shape[0] - 1
                if frame_index > max_frame:
                    print(f"  - 警告: 指定帧索引 {frame_index} 超出范围 (0-{max_frame})")
                    frame_index = 0
                    print(f"  - 使用默认帧: {frame_index}")
                extracted_data = image_data[frame_index]
            else:  # 可能是通道
                max_channel = shape[0] - 1
                if channel_index > max_channel:
                    print(f"  - 警告: 指定通道索引 {channel_index} 超出范围 (0-{max_channel})")
                    channel_index = 0
                    print(f"  - 使用默认通道: {channel_index}")
                extracted_data = image_data[channel_index]
        elif len(shape) == 4:  # 四维图像 (T, C, Y, X)
            max_frame = shape[0] - 1
            max_channel = shape[1] - 1
            
            if frame_index > max_frame:
                print(f"  - 警告: 指定帧索引 {frame_index} 超出范围 (0-{max_frame})")
                frame_index = 0
                print(f"  - 使用默认帧: {frame_index}")
            
            if channel_index > max_channel:
                print(f"  - 警告: 指定通道索引 {channel_index} 超出范围 (0-{max_channel})")
                channel_index = 0
                print(f"  - 使用默认通道: {channel_index}")
            
            extracted_data = image_data[frame_index, channel_index]
        else:
            print(f"  - 不支持的维度: {len(shape)}D")
            return False
        
        # 获取原始文件名（不带扩展名）
        base_name = tif_path.stem
        channel_name = f"channel_{channel_index}"
        frame_name = f"frame_{frame_index}"
        
        # 构建输出文件名
        output_filename = f"{base_name}_{channel_name}_{frame_name}.tif"
        output_path = output_dir / output_filename
        
        print(f"  - 尝试保存到: {output_path}")
        
        # 检查输出目录的写权限
        if not os.access(output_dir, os.W_OK):
            print(f"  - 错误: 没有写入目录 {output_dir} 的权限")
            return False
        
        # 保存为TIFF格式
        try:
            if extracted_data.dtype == np.uint16:
                tifffile.imwrite(output_path, extracted_data, dtype=np.uint16)
            else:
                # 如果不是uint16，转换为合适的格式
                if extracted_data.max() <= 255:
                    extracted_data = extracted_data.astype(np.uint8)
                    tifffile.imwrite(output_path, extracted_data, dtype=np.uint8)
                else:
                    extracted_data = extracted_data.astype(np.uint16)
                    tifffile.imwrite(output_path, extracted_data, dtype=np.uint16)
            
            print(f"  - 已保存: {output_path.name}")
            print(f"  - 图像尺寸: {extracted_data.shape}")
            print(f"  - 数据类型: {extracted_data.dtype}")
            print(f"  - 像素范围: {extracted_data.min()} - {extracted_data.max()}")
            return True
            
        except PermissionError as pe:
            print(f"  - 权限错误: {str(pe)}")
            print(f"  - 请检查目录权限，可能需要使用 'chmod' 命令修改权限")
            return False
        except Exception as e:
            print(f"  - 保存文件时出错: {str(e)}")
            return False
            
    except Exception as e:
        print(f"处理文件 {tif_path} 时出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def process_file(file_path, output_dir, channel_index=0, frame_index=0):
    """
    处理单个文件（ND2或TIFF）
    """
    file_ext = file_path.suffix.lower()
    
    if file_ext == '.nd2':
        return extract_frame_from_nd2(file_path, output_dir, channel_index, frame_index)
    elif file_ext in ['.tif', '.tiff']:
        return extract_frame_from_tif(file_path, output_dir, channel_index, frame_index)
    else:
        print(f"跳过不支持的文件类型: {file_path.name}")
        return False


def get_files_by_pattern(input_pattern):

    # 使用glob进行通配符匹配
    if '*' in input_pattern or '?' in input_pattern or '[' in input_pattern:
        # 如果包含通配符，使用glob.glob
        matched_paths = glob.glob(input_pattern, recursive=True)
        # 过滤出文件（排除目录）
        files = [Path(p) for p in matched_paths if Path(p).is_file()]
    else:
        # 如果不包含通配符，检查是否是目录
        input_path = Path(input_pattern)
        if input_path.is_dir():
            # 如果是目录，获取该目录下所有ND2和TIFF文件
            files = list(input_path.glob("*.nd2")) + list(input_path.glob("*.tif")) + list(input_path.glob("*.tiff"))
        elif input_path.is_file():
            # 如果是单个文件
            files = [input_path]
        else:
            # 如果路径不存在
            print(f"警告: 路径不存在 - {input_pattern}")
            files = []
    
    # 过滤出支持的文件类型
    supported_files = []
    for file_path in files:
        if file_path.suffix.lower() in ['.nd2', '.tif', '.tiff']:
            supported_files.append(file_path)
        else:
            print(f"跳过不支持的文件类型: {file_path.name}")
    
    return sorted(supported_files, key=lambda x: x.name)


def process_files(input_pattern, output_dir, channel_index=0, frame_index=0):

    output_path = Path(output_dir)
    
    # 创建输出目录
    try:
        output_path.mkdir(parents=True, exist_ok=True)
    except PermissionError:
        print(f"错误: 没有创建目录 {output_dir} 的权限")
        return
    
    # 检查输出目录写权限
    if not os.access(output_path, os.W_OK):
        print(f"错误: 没有写入目录 {output_dir} 的权限")
        return
    
    # 获取匹配的文件
    all_files = get_files_by_pattern(input_pattern)
    
    if not all_files:
        print(f"没有找到匹配的ND2或TIFF文件: {input_pattern}")
        return
    
    print(f"找到 {len(all_files)} 个匹配的文件")
    print(f"提取通道: {channel_index}, 帧: {frame_index}")
    print(f"输出目录: {output_dir}")
    print("-" * 50)
    
    success_count = 0
    for file_path in all_files:
        if process_file(file_path, output_path, channel_index, frame_index):
            success_count += 1
        print("-" * 30)
    
    print(f"\n处理完成! 成功处理 {success_count}/{len(all_files)} 个文件")


def main():
    parser = argparse.ArgumentParser(
        description="从ND2/TIFF文件中提取指定通道和帧并保存为TIFF格式",
        formatter_class=argparse.RawDescriptionHelpFormatter    )
    
    parser.add_argument("input_pattern", help="输入模式（支持shell风格通配符或目录路径）")
    parser.add_argument("output_dir", nargs='?', default='extract_frames', 
                       help="输出目录 (默认: extract_frames)")
    parser.add_argument("-c", "--channel", type=int, default=0,
                       help="通道索引 (默认: 0)")
    parser.add_argument("-f", "--frame", type=int, default=0,
                       help="帧索引 (默认: 0)")
    
    args = parser.parse_args()
    
    # 处理文件
    process_files(
        args.input_pattern,
        args.output_dir,
        args.channel,
        args.frame
    )


if __name__ == "__main__":
    main()
