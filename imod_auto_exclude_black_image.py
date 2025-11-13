#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import argparse
import numpy as np
import mrcfile
import matplotlib.pyplot as plt
from scipy.stats import shapiro
import warnings
warnings.filterwarnings("ignore")

def parse_adoc(adoc_path):
    """解析 adoc 文件，返回参数字典"""
    params = {}
    with open(adoc_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if '=' in line:
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip()
                params[key] = value
    return params

def write_adoc(adoc_path, params):
    """将参数写回 adoc 文件"""
    lines = []
    with open(adoc_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                lines.append(line)
                continue
            if '=' in line:
                key, _ = line.split('=', 1)
                key = key.strip()
                if key in params:
                    lines.append(f"{key} = {params[key]}")
                else:
                    lines.append(line)
            else:
                lines.append(line)

    # 如果 setupset.copyarg.skip 不存在，则添加
    if 'setupset.copyarg.skip' not in params:
        # 找到合适插入位置（比如在 setupset.copyarg.stackext 之后）
        insert_index = -1
        for i, line in enumerate(lines):
            if line.startswith('setupset.copyarg.stackext'):
                insert_index = i + 1
                break
        if insert_index == -1:
            insert_index = len(lines)
        lines.insert(insert_index, f"setupset.copyarg.skip = {params.get('setupset.copyarg.skip', '')}")

    with open(adoc_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')

def analyze_mrc_slice_stats(mrc_path, verbose=False):
    """分析 .mrc 文件中每张切片的统计信息，包括正态性检验"""
    with mrcfile.open(mrc_path, permissive=True) as mrc:
        data = mrc.data  # shape: (nz, ny, nx)

    nz = data.shape[0]
    stats = []

    for i in range(nz):
        slice_data = data[i].flatten()
        mean_val = np.mean(slice_data)
        std_val = np.std(slice_data)
        min_val = np.min(slice_data)
        max_val = np.max(slice_data)

        # 检查是否为全黑或全同值的切片
        if np.all(slice_data == slice_data[0]):  # 如果所有值都相同
            statistic = 0.0  # 全同值或全黑图片，直接设为0
        else:
            # Shapiro-Wilk 正态性检验
            try:
                statistic, _ = shapiro(slice_data)  # 获取statistic
            except Exception:
                statistic = 0.0  # 如果数据太小或其他异常，设为0

        stats.append({
            'slice': i+1,
            'mean': mean_val,
            'std': std_val,
            'min': min_val,
            'max': max_val,
            'range': (min_val, max_val),
            'statistic': statistic
        })
        if verbose:
            print(f"  Slice {i:4d}: μ={mean_val:7.2f}, σ={std_val:7.2f}, W={statistic:6.4f}, range=[{min_val:8.2f}, {max_val:8.2f}]")

    return stats

def get_slices_to_exclude_by_threshold(stats, threshold=0.95):
    """
    根据给定阈值直接判断异常切片。
    规则：
    ①只排除位于两侧的各30%的照片。
    ②一旦排除一张，那么将该张连同更靠外侧的照片一起删除。
    """
    n_slices = len(stats)
    if n_slices < 1:
        return []
    
    side_threshold = int(n_slices * 0.3)
    
    slices_to_exclude = set()
    
    # 前向检查：从开头开始找第一个低于阈值的切片
    forward_exclusion_point = None
    for i in range(n_slices // 2):
        if stats[i]['statistic'] < threshold:
            forward_exclusion_point = i
            
    
    # 后向检查：从结尾开始找第一个低于阈值的切片
    backward_exclusion_point = None
    for i in range(n_slices - 1, n_slices // 2, -1):
        if stats[i]['statistic'] < threshold:
            backward_exclusion_point = i
            break
    
    # 应用规则：只在两侧30%范围内排除，并且一旦排除就包括更外侧的所有切片
    if forward_exclusion_point is not None and forward_exclusion_point < side_threshold:
        # 排除从开头到forward_exclusion_point的所有切片
        slices_to_exclude.update(range(0, forward_exclusion_point + 1))
    
    if backward_exclusion_point is not None and backward_exclusion_point >= n_slices - side_threshold:
        # 排除从backward_exclusion_point到结尾的所有切片
        slices_to_exclude.update(range(backward_exclusion_point, n_slices))
    
    return sorted(list(slices_to_exclude))

def create_range_string(indices):
    """将索引列表转换为 range 字符串，例如 '1-3,5,7-9'"""
    if not indices:
        return ""
    ranges = []
    start = indices[0]
    end = indices[0]
    for i in range(1, len(indices)):
        if indices[i] == end + 1:
            end = indices[i]
        else:
            if start == end:
                ranges.append(str(start+1))
            else:
                ranges.append(f"{start+1}-{end+1}")
            start = indices[i]
            end = indices[i]
    if start == end:
        ranges.append(str(start+1))
    else:
        ranges.append(f"{start+1}-{end+1}")
    return ",".join(ranges)

def plot_results(mrc_path, stats, excluded_indices, output_dir):
    """生成可视化图表：statistic 曲线图 + 缩略图网格"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    slice_nums = [s['slice'] for s in stats]
    stat_values = [s['statistic'] for s in stats]

    # statistic 曲线图
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(slice_nums, stat_values, label='Shapiro-Wilk Statistic', marker='o', markersize=3, color='blue')
    ax.axhline(y=0.95, color='red', linestyle='--', label='W=0.95 (normality threshold)')
    ax.set_title(f'Shapiro-Wilk Statistic per Slice - {os.path.basename(mrc_path)}')
    ax.set_xlabel('Slice Number')
    ax.set_ylabel('W Statistic')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend()

    # 高亮被排除的切片
    if excluded_indices:
        ax.scatter(excluded_indices, [stat_values[i] for i in excluded_indices], color='red', s=50, label='Excluded Slices', zorder=5)
        ax.legend()

    plot_filename = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(mrc_path))[0]}_statistic_plot.png")
    fig.tight_layout()
    fig.savefig(plot_filename, dpi=150)
    plt.close(fig)
    print(f"  statistic plot saved to: {plot_filename}")

    # 生成缩略图网格
    plot_thumbnail_grid(mrc_path, stats, excluded_indices, output_dir)

def plot_thumbnail_grid(mrc_path, stats, excluded_indices, output_dir):
    """生成缩略图网格图，红框标出被排除的切片"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with mrcfile.open(mrc_path, permissive=True) as mrc:
        data = mrc.data

    nz = data.shape[0]
    # 计算网格大小：尽量接近正方形
    cols = int(np.ceil(np.sqrt(nz)))
    rows = int(np.ceil(nz / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(cols*2, rows*2))
    fig.suptitle(f'Thumbnail Grid - {os.path.basename(mrc_path)}', fontsize=14)

    if rows == 1 and cols == 1:
        axes = np.array([axes])

    for i in range(nz):
        row = i // cols
        col = i % cols
        ax = axes[row, col] if rows > 1 else axes[col]

        # 显示缩略图
        img = data[i]
        ax.imshow(img, cmap='gray', vmin=np.percentile(img, 1), vmax=np.percentile(img, 99))
        ax.set_title(f'Slice {i}', fontsize=8)
        ax.axis('off')

        # 如果该切片被排除，画红框
        if i in excluded_indices:
            rect = plt.Rectangle((0, 0), img.shape[1], img.shape[0],
                                 linewidth=3, edgecolor='red', facecolor='none', zorder=5)
            ax.add_patch(rect)

    # 隐藏多余的子图
    for i in range(nz, rows * cols):
        row = i // cols
        col = i % cols
        ax = axes[row, col] if rows > 1 else axes[col]
        ax.axis('off')

    fig.tight_layout()
    plot_filename = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(mrc_path))[0]}_thumbnails.png")
    fig.savefig(plot_filename, dpi=150)
    plt.close(fig)
    print(f"  Thumbnail grid saved to: {plot_filename}")

def get_mrc_filename_from_adoc(adoc_path, project_basename):
    """
    从 adoc 文件路径生成对应的 mrc 文件路径。
    规则：移除 project_basename + '_' 前缀。
    """
    filename = os.path.basename(adoc_path)
    # 移除 .adoc 后缀
    base_name_no_ext = os.path.splitext(filename)[0]
    # 移除 project_basename + '_' 前缀
    if base_name_no_ext.startswith(project_basename + '_'):
        mrc_base = base_name_no_ext[len(project_basename) + 1:]  # +1 是下划线
    else:
        mrc_base = base_name_no_ext  # 如果不匹配，保留原名（容错）
    mrc_path = os.path.join(os.path.dirname(adoc_path), mrc_base + '.mrc')
    return mrc_path

def main():
    parser = argparse.ArgumentParser(description='Automatically modify IMOD adoc files to exclude occluded images in cryoET stacks.')
    parser.add_argument('project_adoc', type=str, help='Path to the project adoc file (e.g., batchNov13-145317.adoc)')
    parser.add_argument('--no-plot', action='store_true', help='Disable visualization plot generation')
    parser.add_argument('--plot-dir', type=str, default='plots', help='Directory to save plots (default: plots)')
    parser.add_argument('--verbose', action='store_true', help='Print detailed slice statistics')

    args = parser.parse_args()

    project_path = args.project_adoc
    project_dir = os.path.dirname(project_path)
    project_basename = os.path.splitext(os.path.basename(project_path))[0]

    # 查找所有对应的 adoc 文件
    pattern = os.path.join(project_dir, f"{project_basename}_*.adoc")
    adoc_files = glob.glob(pattern)

    if not adoc_files:
        print(f"No matching adoc files found for pattern: {pattern}")
        return

    print(f"Found {len(adoc_files)} adoc files to process.")
    for adoc_file in adoc_files:
        print(f"Processing: {adoc_file}")

        mrc_file = get_mrc_filename_from_adoc(adoc_file, project_basename)

        if not os.path.exists(mrc_file):
            print(f"  Warning: MRC file not found: {mrc_file}. Skipping.")
            continue

        # 1. 分析 MRC 切片
        print(f"  Analyzing MRC file: {mrc_file}")
        stats = analyze_mrc_slice_stats(mrc_file, verbose=args.verbose)

        # 2. 直接按阈值判断需要排除的切片
        excluded_indices = get_slices_to_exclude_by_threshold(stats, threshold=0.95)
        print(f"  Slices to exclude: {excluded_indices}")

        # 3. 生成可视化
        if not args.no_plot:
            plot_results(mrc_file, stats, excluded_indices, args.plot_dir)

        # 4. 修改 adoc 文件
        adoc_params = parse_adoc(adoc_file)
        if excluded_indices:
            skip_str = create_range_string(excluded_indices)
            adoc_params['setupset.copyarg.skip'] = skip_str
            print(f"  Updating setupset.copyarg.skip to: {skip_str}")
        else:
            adoc_params['setupset.copyarg.skip'] = '' # Clear if no slices to exclude
            print(f"  No slices to exclude, clearing setupset.copyarg.skip")

        write_adoc(adoc_file, adoc_params)
        print(f"  Updated adoc file: {adoc_file}\n")

if __name__ == '__main__':
    main()
