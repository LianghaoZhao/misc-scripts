import argparse
import os
import numpy as np
from scipy import ndimage
from skimage.registration import phase_cross_correlation
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import nd2
from pathlib import Path
import json
from datetime import datetime
import tifffile  # 需要安装：pip install tifffile

def calculate_drift_correlation(image1, image2, upsample_factor=10):
    """使用相位相关计算两帧之间的漂移"""
    shift, error, diffphase = phase_cross_correlation(
        image1, image2, upsample_factor=upsample_factor
    )
    return (-shift[0],-shift[1])

def crop_to_valid_region(frames, cumulative_shifts, border=0):
    """
    根据漂移信息裁剪图像，确保所有帧都保持在原始视野内
    
    Parameters:
    - frames: 原始图像序列 (T, H, W)
    - cumulative_shifts: 累积漂移 (T, 2) - [x, y]
    - border: 额外保留的边框像素数
    """
    H, W = frames.shape[1], frames.shape[2]
    
    # 计算所有帧的漂移范围
    x_shifts = cumulative_shifts[:, 0]
    y_shifts = cumulative_shifts[:, 1]
    
    # 计算需要裁剪的边界 - 修正方向
    min_x_shift, max_x_shift = np.min(x_shifts), np.max(x_shifts)
    min_y_shift, max_y_shift = np.min(y_shifts), np.max(y_shifts)
    
    # 计算裁剪边界（考虑漂移后的有效区域）
    # 修正：left_crop应该考虑最大负偏移，right_crop考虑最大正偏移
    left_crop = int(np.ceil(max(0, -min_x_shift + border)))
    right_crop = int(np.floor(W - max(0, max_x_shift + border)))
    top_crop = int(np.ceil(max(0, -min_y_shift + border)))
    bottom_crop = int(np.floor(H - max(0, max_y_shift + border)))
    
    # 确保裁剪边界有效
    left_crop = max(0, min(left_crop, W-1))
    right_crop = max(left_crop + 1, min(right_crop, W))
    top_crop = max(0, min(top_crop, H-1))
    bottom_crop = max(top_crop + 1, min(bottom_crop, H))
    
    # 裁剪图像
    cropped_frames = frames[:, top_crop:bottom_crop, left_crop:right_crop]
    
    # 更新漂移信息（相对于新坐标系）
    adjusted_shifts = cumulative_shifts.copy()
    adjusted_shifts[:, 0] -= left_crop
    adjusted_shifts[:, 1] -= top_crop
    
    print(f"Cropped region: [{left_crop}:{right_crop}, {top_crop}:{bottom_crop}]")
    print(f"Original size: {W}x{H}, Cropped size: {right_crop-left_crop}x{bottom_crop-top_crop}")
    
    return cropped_frames, adjusted_shifts, (left_crop, right_crop, top_crop, bottom_crop)

def iterative_drift_correction(frames, max_iterations=10, threshold=0.5, upsample_factor=50):
    """
    迭代进行漂移校正
    
    Parameters:
    - frames: 输入图像序列 (T, H, W) - 用于计算漂移的图像（如合并通道）
    - max_iterations: 最大迭代次数
    - threshold: 停止迭代的阈值（像素）
    - upsample_factor: 相位相关算法的上采样因子
    """
    T, H, W = frames.shape
    original_frames = frames.copy()  # 保存原始用于计算漂移的图像，每次迭代都从这里开始
    
    # 初始化累积漂移
    cumulative_shifts = np.zeros((T, 2))  # [x, y] for each frame
    
    print(f"Starting iterative drift correction (max {max_iterations} iterations, threshold {threshold} pixels)")
    
    for iteration in range(max_iterations):
        print(f"Iteration {iteration + 1}:")
        
        # 使用当前累积漂移校正原始图像
        corrected_frames = np.zeros_like(original_frames, dtype=original_frames.dtype)
        corrected_frames[0] = original_frames[0]  # 第一帧不变
        
        for i in range(1, T):
            shift_x, shift_y = cumulative_shifts[i]
            corrected_frame = ndimage.shift(
                original_frames[i], 
                (-shift_y, -shift_x),  # 负号确保正确的校正方向
                order=1, 
                mode='nearest',
                cval=0
            )
            corrected_frames[i] = corrected_frame
        
        # 计算当前校正后帧序列的漂移
        shifts = [(0, 0)]  # 第一帧作为参考，偏移为(0,0)
        
        for i in range(1, T):
            shift = calculate_drift_correlation(corrected_frames[i-1], corrected_frames[i], upsample_factor)
            shifts.append((float(shift[0]), float(shift[1])))
        
        # 将新的漂移添加到累积漂移中
        new_cumulative_shifts = np.zeros((T, 2))
        cum_x, cum_y = 0.0, 0.0
        for idx, shift in enumerate(shifts):
            cum_y += shift[0]  # 注意：相位相关返回的是 (row, col) -> (y, x)
            cum_x += shift[1]
            new_cumulative_shifts[idx] = [cum_x, cum_y]
        
        # 检查是否收敛
        max_drift = max(np.max(np.abs(new_cumulative_shifts[:, 0])), np.max(np.abs(new_cumulative_shifts[:, 1])))
        print(f"  Max drift: X={np.max(np.abs(new_cumulative_shifts[:, 0])):.3f}, Y={np.max(np.abs(new_cumulative_shifts[:, 1])):.3f}, Total={max_drift:.3f}")
        
        if max_drift < threshold:
            print(f"  Converged at iteration {iteration + 1} (max drift {max_drift:.3f} < threshold {threshold})")
            cumulative_shifts += new_cumulative_shifts
            break
        
        # 更新累积漂移
        cumulative_shifts += new_cumulative_shifts
    
    else:
        print(f"Reached maximum iterations ({max_iterations})")
    
    return cumulative_shifts  # 只返回累积漂移，不返回校正后的图像

def process_image_sequence(input_path, output_dir, channel_selection='all', 
                          sample_interval=1, save_visualization=True, auto_crop=True, border=0,
                          max_iterations=10, threshold=0.5):
    """
    处理图像序列并进行漂移校正
    
    Parameters:
    - input_path: 输入文件路径 (ND2或TIF)
    - output_dir: 输出目录
    - channel_selection: 通道选择 ('all', 'first', 'last', or specific index)
    - sample_interval: 可视化采样间隔
    - save_visualization: 是否保存可视化
    - auto_crop: 是否在漂移校正后自动裁剪
    - border: 裁剪时保留的边框像素数
    - max_iterations: 最大迭代次数
    - threshold: 停止迭代的阈值（像素）
    """
    
    input_path = Path(input_path)
    print(f"Processing: {input_path}")
    
    # 读取图像数据
    if input_path.suffix.lower() == '.nd2':
        with nd2.ND2File(input_path) as ndfile:
            # 获取基本信息
            shape = dict(ndfile.sizes)
            T, C, Y, X = shape['T'], shape['C'], shape['Y'], shape['X']
            
            print(f"File shape: T={T}, C={C}, Y={Y}, X={X}")
            
            # 确定处理的通道
            if channel_selection == 'all':
                channels_to_process = list(range(C))
            elif channel_selection == 'first':
                channels_to_process = [0]
            elif channel_selection == 'last':
                channels_to_process = [C-1]
            else:
                try:
                    ch_idx = int(channel_selection)
                    if 0 <= ch_idx < C:
                        channels_to_process = [ch_idx]
                    else:
                        print(f"Invalid channel index {ch_idx}, using all channels")
                        channels_to_process = list(range(C))
                except ValueError:
                    print(f"Invalid channel selection {channel_selection}, using all channels")
                    channels_to_process = list(range(C))
            
            print(f"Processing channels: {channels_to_process}")
            
            # 读取所有帧
            all_frames = ndfile.asarray()  # Shape: (T, C, Y, X)
            
            # 用于计算漂移的图像 - 合并选定的通道
            if len(channels_to_process) == 1:
                drift_frames = all_frames[:, channels_to_process[0], :, :]
            else:
                # 合并多个通道 - 使用最大值投影
                selected_frames = all_frames[:, channels_to_process, :, :]
                drift_frames = np.max(selected_frames, axis=1)  # 沿通道轴取最大值
            
            # 用于校正的原始图像 - 保留所有通道
            original_frames = all_frames  # Shape: (T, C, Y, X)
            channels_to_save = channels_to_process  # 记录要保存的通道

    elif input_path.suffix.lower() in ['.tif', '.tiff']:
        # 读取TIF文件
        with tifffile.TiffFile(str(input_path)) as tif:
            all_frames = tif.asarray()
            
            # 处理不同维度的TIF文件
            if all_frames.ndim == 3:  # (T, H, W)
                T, Y, X = all_frames.shape
                C = 1
                drift_frames = all_frames
                original_frames = all_frames  # Shape: (T, 1, Y, X) - 保持一致的维度
                if all_frames.ndim == 3:
                    original_frames = all_frames[:, np.newaxis, :, :]  # 添加通道维度
                channels_to_save = [0]
                print(f"TIF file shape: T={T}, Y={Y}, X={X} (single channel)")
            elif all_frames.ndim == 4:  # (T, C, H, W) or (T, H, W, C)
                if all_frames.shape[1] < all_frames.shape[-1]:  # Assume (T, C, H, W)
                    T, C, Y, X = all_frames.shape
                    if channel_selection == 'all':
                        channels_to_process = list(range(C))
                    elif channel_selection == 'first':
                        channels_to_process = [0]
                    elif channel_selection == 'last':
                        channels_to_process = [C-1]
                    else:
                        try:
                            ch_idx = int(channel_selection)
                            if 0 <= ch_idx < C:
                                channels_to_process = [ch_idx]
                            else:
                                print(f"Invalid channel index {ch_idx}, using first channel")
                                channels_to_process = [0]
                        except ValueError:
                            print(f"Invalid channel selection {channel_selection}, using first channel")
                            channels_to_process = [0]
                    
                    # 用于计算漂移的图像 - 合并选定的通道
                    if len(channels_to_process) == 1:
                        drift_frames = all_frames[:, channels_to_process[0], :, :]
                    else:
                        selected_frames = all_frames[:, channels_to_process, :, :]
                        drift_frames = np.max(selected_frames, axis=1)  # 沿通道轴取最大值
                    
                    original_frames = all_frames  # Shape: (T, C, Y, X)
                    channels_to_save = channels_to_process
                    print(f"TIF file shape: T={T}, C={C}, Y={Y}, X={X}")
                    print(f"Processing channels: {channels_to_process}")
                else:  # Assume (T, H, W, C)
                    T, Y, X, C = all_frames.shape
                    print(f"TIF file shape: T={T}, Y={Y}, X={X}, C={C}")
                    if channel_selection == 'all':
                        channels_to_process = list(range(C))
                    elif channel_selection == 'first':
                        channels_to_process = [0]
                    elif channel_selection == 'last':
                        channels_to_process = [C-1]
                    else:
                        try:
                            ch_idx = int(channel_selection)
                            if 0 <= ch_idx < C:
                                channels_to_process = [ch_idx]
                            else:
                                print(f"Invalid channel index {ch_idx}, using first channel")
                                channels_to_process = [0]
                        except ValueError:
                            print(f"Invalid channel selection {channel_selection}, using first channel")
                            channels_to_process = [0]
                    
                    # 用于计算漂移的图像 - 合并选定的通道
                    if len(channels_to_process) == 1:
                        drift_frames = all_frames[:, :, :, channels_to_process[0]]
                    else:
                        selected_frames = all_frames[:, :, :, channels_to_process]
                        drift_frames = np.max(selected_frames, axis=-1)  # 沿通道轴取最大值
                    
                    # 转换为 (T, C, Y, X) 格式
                    original_frames = np.transpose(all_frames, (0, 3, 1, 2))  # (T, C, Y, X)
                    channels_to_save = channels_to_process
            else:
                raise ValueError(f"Unsupported TIF file dimensions: {all_frames.ndim}")
    else:
        raise ValueError(f"Unsupported file format: {input_path.suffix}")
    
    # 执行迭代漂移校正 - 只计算漂移，不应用
    cumulative_shifts = iterative_drift_correction(
        drift_frames, max_iterations=max_iterations, threshold=threshold
    )
    
    # 应用漂移到原始的每个通道
    print("Applying calculated drift to original frames...")
    corrected_frames = np.zeros_like(original_frames, dtype=original_frames.dtype)
    corrected_frames[:, :, :, :] = original_frames  # 复制原始数据
    
    # 对每一帧的每个通道应用漂移
    for i in range(T):
        shift_x, shift_y = cumulative_shifts[i]
        for c in range(original_frames.shape[1]):  # 遍历所有通道
            corrected_frame = ndimage.shift(
                original_frames[i, c], 
                (-shift_y, -shift_x),  # 负号确保正确的校正方向
                order=1, 
                mode='nearest',
                cval=0
            )
            corrected_frames[i, c] = corrected_frame
    
    # 如果启用自动裁剪，需要对多通道数据进行裁剪
    if auto_crop:
        print("Applying automatic cropping to keep all frames in field of view...")
        original_shape = corrected_frames.shape
        # 对每个通道应用相同的裁剪
        H, W = corrected_frames.shape[2], corrected_frames.shape[3]
        
        # 计算裁剪边界
        x_shifts = cumulative_shifts[:, 0]
        y_shifts = cumulative_shifts[:, 1]
        
        min_x_shift, max_x_shift = np.min(x_shifts), np.max(x_shifts)
        min_y_shift, max_y_shift = np.min(y_shifts), np.max(y_shifts)
        
        left_crop = int(np.ceil(max(0, -min_x_shift + border)))
        right_crop = int(np.floor(W - max(0, max_x_shift + border)))
        top_crop = int(np.ceil(max(0, -min_y_shift + border)))
        bottom_crop = int(np.floor(H - max(0, max_y_shift + border)))
        
        # 确保裁剪边界有效
        left_crop = max(0, min(left_crop, W-1))
        right_crop = max(left_crop + 1, min(right_crop, W))
        top_crop = max(0, min(top_crop, H-1))
        bottom_crop = max(top_crop + 1, min(bottom_crop, H))
        
        # 裁剪多通道图像
        corrected_frames = corrected_frames[:, :, top_crop:bottom_crop, left_crop:right_crop]
        
        # 更新漂移信息（相对于新坐标系）
        cumulative_shifts[:, 0] -= left_crop
        cumulative_shifts[:, 1] -= top_crop
        
        print(f"Auto-cropped from {original_shape} to {corrected_frames.shape}")
    
    # 创建输出目录
    output_path = Path(output_dir) / f"{input_path.stem}_corrected"
    
    # 保存漂移信息
    drift_info = {
        'file': str(input_path),
        'original_shape': drift_frames.shape,
        'corrected_shape': corrected_frames.shape,
        'channel_selection': channel_selection,
        'cumulative_shifts': [list(s) for s in cumulative_shifts],
        'processing_time': datetime.now().isoformat(),
        'sample_interval': sample_interval,
        'total_frames': T,
        'max_x_drift': float(np.max(np.abs(cumulative_shifts[:, 0]))),
        'max_y_drift': float(np.max(np.abs(cumulative_shifts[:, 1]))),
        'final_x_drift': float(cumulative_shifts[-1, 0]),
        'final_y_drift': float(cumulative_shifts[-1, 1]),
        'auto_crop_enabled': auto_crop,
        'border_pixels': border,
        'max_iterations': max_iterations,
        'threshold': threshold,
        'channels_saved': channels_to_save
    }
    
    if auto_crop:
        drift_info['crop_info'] = (int(left_crop), int(right_crop), int(top_crop), int(bottom_crop))
    
    drift_json_path = output_path.with_name(output_path.stem + "_drift_info.json")
    with open(drift_json_path, 'w') as f:
        json.dump(drift_info, f, indent=2, ensure_ascii=False)
    
    print(f"Drift info saved to: {drift_json_path}")
    
    # 创建可视化
    if save_visualization:
        create_drift_visualization(
            cumulative_shifts, 
            output_dir, 
            input_path.stem,
            sample_interval
        )
    
    # 保存校正后的数据
    corrected_tiff_path = output_path.with_name(output_path.stem + ".ome.tif")
    
    # 如果只保存选定的通道
    if len(channels_to_save) < corrected_frames.shape[1]:
        corrected_frames_to_save = corrected_frames[:, channels_to_save, :, :]
    else:
        corrected_frames_to_save = corrected_frames
    
    # 保存校正后的数据
    tifffile.imwrite(
        str(corrected_tiff_path),
        corrected_frames_to_save,
        ome=True,
        metadata={'axes': 'TCYX'}  # 指定轴顺序
    )
    
    print(f"Corrected data saved to: {corrected_tiff_path}")
    
    return cumulative_shifts, corrected_tiff_path

def create_drift_visualization(cumulative_shifts, output_dir, base_name, sample_interval=1):
    """创建漂移轨迹的可视化"""
    
    # 准备数据
    time_points = np.arange(len(cumulative_shifts))
    x_shifts = cumulative_shifts[:, 0]
    y_shifts = cumulative_shifts[:, 1]
    
    # 如果采样间隔大于1，对数据进行采样
    if sample_interval > 1:
        indices = np.arange(0, len(cumulative_shifts), sample_interval)
        time_points = time_points[indices]
        x_shifts = x_shifts[indices]
        y_shifts = y_shifts[indices]
    
    # 确保坐标轴范围至少为10个像素
    x_range = np.max(np.abs(x_shifts)) - np.min(np.abs(x_shifts))
    y_range = np.max(np.abs(y_shifts)) - np.min(np.abs(y_shifts))
    
    if x_range < 10:
        x_center = (np.max(x_shifts) + np.min(x_shifts)) / 2
        x_min = x_center - 5
        x_max = x_center + 5
    else:
        x_min = np.min(x_shifts) - 1
        x_max = np.max(x_shifts) + 1
    
    if y_range < 10:
        y_center = (np.max(y_shifts) + np.min(y_shifts)) / 2
        y_min = y_center - 5
        y_max = y_center + 5
    else:
        y_min = np.min(y_shifts) - 1
        y_max = np.max(y_shifts) + 1
    
    # 创建综合可视化图
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Drift Correction Summary - {base_name}', fontsize=16, fontweight='bold')
    
    # 1. 轨迹图 (XY平面)
    scatter = axes[0, 0].scatter(x_shifts, y_shifts, c=time_points, 
                                cmap='viridis', s=20, alpha=0.7, edgecolors='none')
    axes[0, 0].plot(x_shifts, y_shifts, 'k-', alpha=0.3, linewidth=0.5)
    axes[0, 0].set_xlabel('X Shift (pixels)')
    axes[0, 0].set_ylabel('Y Shift (pixels)')
    axes[0, 0].set_title('Drift Trajectory (XY Plane)')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_xlim(x_min, x_max)
    axes[0, 0].set_ylim(y_min, y_max)
    plt.colorbar(scatter, ax=axes[0, 0], label='Time Point')
    
    # 2. X方向时间序列
    axes[0, 1].plot(time_points, x_shifts, 'b-', linewidth=1.5, alpha=0.8)
    axes[0, 1].scatter(time_points, x_shifts, c=time_points, cmap='viridis', s=20, alpha=0.7)
    axes[0, 1].set_xlabel('Time Point')
    axes[0, 1].set_ylabel('X Shift (pixels)')
    axes[0, 1].set_title('X Drift vs Time')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim(x_min, x_max)
    
    # 3. Y方向时间序列
    axes[1, 0].plot(time_points, y_shifts, 'r-', linewidth=1.5, alpha=0.8)
    axes[1, 0].scatter(time_points, y_shifts, c=time_points, cmap='viridis', s=20, alpha=0.7)
    axes[1, 0].set_xlabel('Time Point')
    axes[1, 0].set_ylabel('Y Shift (pixels)')
    axes[1, 0].set_title('Y Drift vs Time')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_ylim(y_min, y_max)
    
    # 4. 2D直方图显示密度
    im = axes[1, 1].hexbin(x_shifts, y_shifts, gridsize=30, cmap='Blues', mincnt=1)
    axes[1, 1].set_xlabel('X Shift (pixels)')
    axes[1, 1].set_ylabel('Y Shift (pixels)')
    axes[1, 1].set_title('Drift Density Distribution')
    axes[1, 1].set_xlim(x_min, x_max)
    axes[1, 1].set_ylim(y_min, y_max)
    plt.colorbar(im, ax=axes[1, 1], label='Point Density')
    
    plt.tight_layout()
    
    # 保存可视化
    viz_path = Path(output_dir) / f"{base_name}_drift_visualization.png"
    plt.savefig(viz_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Visualization saved to: {viz_path}")

def create_summary_plot(all_shifts_data, output_dir):
    """创建所有文件的漂移轨迹总结图"""
    if not all_shifts_data:
        print("No data to create summary plot")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Summary of All Files Drift Trajectories', fontsize=16, fontweight='bold')
    
    # 收集所有文件的x、y漂移范围
    all_x_min, all_x_max = float('inf'), float('-inf')
    all_y_min, all_y_max = float('inf'), float('-inf')
    
    # 绘制所有文件的轨迹
    colors = plt.cm.tab10(np.linspace(0, 1, len(all_shifts_data)))
    
    for idx, (filename, shifts) in enumerate(all_shifts_data):
        x_shifts = shifts[:, 0]
        y_shifts = shifts[:, 1]
        time_points = np.arange(len(shifts))
        
        # 更新全局范围
        current_x_min, current_x_max = np.min(x_shifts), np.max(x_shifts)
        current_y_min, current_y_max = np.min(y_shifts), np.max(y_shifts)
        
        all_x_min = min(all_x_min, current_x_min)
        all_x_max = max(all_x_max, current_x_max)
        all_y_min = min(all_y_min, current_y_min)
        all_y_max = max(all_y_max, current_y_max)
        
        # 1. 轨迹图 (XY平面)
        axes[0, 0].plot(x_shifts, y_shifts, color=colors[idx], alpha=0.7, linewidth=1.5, 
                       label=Path(filename).stem)
        axes[0, 0].scatter(x_shifts[0], y_shifts[0], color=colors[idx], s=50, marker='o', 
                          edgecolors='black', zorder=5)  # 标记起点
        axes[0, 0].scatter(x_shifts[-1], y_shifts[-1], color=colors[idx], s=50, marker='s', 
                          edgecolors='black', zorder=5)  # 标记终点
    
    # 确保坐标轴范围至少为10个像素
    x_range = all_x_max - all_x_min
    y_range = all_y_max - all_y_min
    
    if x_range < 10:
        x_center = (all_x_max + all_x_min) / 2
        all_x_min = x_center - 5
        all_x_max = x_center + 5
    else:
        all_x_min -= 1
        all_x_max += 1
    
    if y_range < 10:
        y_center = (all_y_max + all_y_min) / 2
        all_y_min = y_center - 5
        all_y_max = y_center + 5
    else:
        all_y_min -= 1
        all_y_max += 1
    
    axes[0, 0].set_xlabel('X Shift (pixels)')
    axes[0, 0].set_ylabel('Y Shift (pixels)')
    axes[0, 0].set_title('All Files - Drift Trajectory (XY Plane)')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_xlim(all_x_min, all_x_max)
    axes[0, 0].set_ylim(all_y_min, all_y_max)
    axes[0, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # 2. X方向时间序列
    for idx, (filename, shifts) in enumerate(all_shifts_data):
        x_shifts = shifts[:, 0]
        time_points = np.arange(len(shifts))
        axes[0, 1].plot(time_points, x_shifts, color=colors[idx], alpha=0.7, linewidth=1.5, 
                       label=Path(filename).stem)
    
    axes[0, 1].set_xlabel('Time Point')
    axes[0, 1].set_ylabel('X Shift (pixels)')
    axes[0, 1].set_title('All Files - X Drift vs Time')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim(all_x_min, all_x_max)
    axes[0, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # 3. Y方向时间序列
    for idx, (filename, shifts) in enumerate(all_shifts_data):
        y_shifts = shifts[:, 1]
        time_points = np.arange(len(shifts))
        axes[1, 0].plot(time_points, y_shifts, color=colors[idx], alpha=0.7, linewidth=1.5, 
                       label=Path(filename).stem)
    
    axes[1, 0].set_xlabel('Time Point')
    axes[1, 0].set_ylabel('Y Shift (pixels)')
    axes[1, 0].set_title('All Files - Y Drift vs Time')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_ylim(all_y_min, all_y_max)
    axes[1, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # 4. 统计信息
    axes[1, 1].axis('off')
    stats_text = "Summary Statistics:\n\n"
    for idx, (filename, shifts) in enumerate(all_shifts_data):
        x_shifts = shifts[:, 0]
        y_shifts = shifts[:, 1]
        
        max_x_drift = np.max(np.abs(x_shifts))
        max_y_drift = np.max(np.abs(y_shifts))
        final_x_drift = x_shifts[-1]
        final_y_drift = y_shifts[-1]
        
        stats_text += f"{Path(filename).stem}:\n"
        stats_text += f"  Max X drift: {max_x_drift:.2f}px\n"
        stats_text += f"  Max Y drift: {max_y_drift:.2f}px\n"
        stats_text += f"  Final X drift: {final_x_drift:.2f}px\n"
        stats_text += f"  Final Y drift: {final_y_drift:.2f}px\n\n"
    
    axes[1, 1].text(0.05, 0.95, stats_text, transform=axes[1, 1].transAxes, 
                    verticalalignment='top', fontsize=10, fontfamily='monospace',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
    
    plt.tight_layout()
    
    # 保存总结图
    summary_path = Path(output_dir) / "all_files_drift_summary.png"
    plt.savefig(summary_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Summary plot saved to: {summary_path}")

def main():
    parser = argparse.ArgumentParser(description='ND2/TIF Motion Correction Tool')
    parser.add_argument('input_dir', help='Input directory containing ND2/TIF files')
    parser.add_argument('--output_dir', '-o', help='Output directory for corrected files (default: MotionCor subdirectory)')
    parser.add_argument('--channel', '-c', default='all', 
                       help='Channel to use for drift calculation (all, first, last, or specific index)')
    parser.add_argument('--sample_interval', '-s', type=int, default=5,
                       help='Sampling interval for visualization (default: 5)')
    parser.add_argument('--no_visualization', action='store_true',
                       help='Skip visualization creation')
    parser.add_argument('--no_auto_crop', action='store_true',
                       help='Disable automatic cropping to keep all frames in field of view')
    parser.add_argument('--border', type=int, default=0,
                       help='Additional border pixels to keep when auto-cropping (default: 0)')
    parser.add_argument('--max_iterations', type=int, default=10,
                       help='Maximum number of iterative corrections (default: 10)')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Threshold for stopping iteration (default: 0.5 pixels)')
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"Error: Input directory {input_dir} does not exist")
        return
    
    # 设置输出目录
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = input_dir / 'MotionCor'
    
    output_dir.mkdir(exist_ok=True)
    
    # 查找所有ND2和TIF文件
    nd2_files = list(input_dir.glob('*.nd2'))
    tif_files = list(input_dir.glob('*.tif')) + list(input_dir.glob('*.tiff'))
    all_files = nd2_files + tif_files
    
    if not all_files:
        print(f"No ND2 or TIF files found in {input_dir}")
        return
    
    print(f"Found {len(all_files)} files to process ({len(nd2_files)} ND2, {len(tif_files)} TIF)")
    
    # 存储所有文件的漂移数据
    all_shifts_data = []
    
    for file_path in all_files:
        try:
            shifts, output_path = process_image_sequence(
                str(file_path),
                str(output_dir),
                channel_selection=args.channel,
                sample_interval=args.sample_interval,
                save_visualization=not args.no_visualization,
                auto_crop=not args.no_auto_crop,
                border=args.border,
                max_iterations=args.max_iterations,
                threshold=args.threshold
            )
            
            # 添加到总结数据中
            all_shifts_data.append((str(file_path), shifts))
            
            # 打印统计信息
            total_x_drift = np.abs(shifts[-1, 0])
            total_y_drift = np.abs(shifts[-1, 1])
            max_x_drift = np.max(np.abs(shifts[:, 0]))
            max_y_drift = np.max(np.abs(shifts[:, 1]))
            
            print(f"File: {file_path.name}")
            print(f"  Final drift: X={total_x_drift:.2f}, Y={total_y_drift:.2f} pixels")
            print(f"  Max drift: X={max_x_drift:.2f}, Y={max_y_drift:.2f} pixels")
            print(f"  Max single-step drift: X={np.max(np.abs(np.diff(shifts[:, 0])))}, Y={np.max(np.abs(np.diff(shifts[:, 1])))} pixels")
            print(f"  Correction completed: {output_path}")
            print()
            
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    # 创建总结图
    if all_shifts_data and not args.no_visualization:
        create_summary_plot(all_shifts_data, output_dir)
        print(f"Created summary plot for {len(all_shifts_data)} files")

if __name__ == "__main__":
    main()
