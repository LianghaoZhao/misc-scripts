import nd2
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from scipy.optimize import curve_fit
from skimage import io
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import os
from pathlib import Path
import argparse
import glob
from natsort import natsorted

def first_order_reaction(t, A0, k, A_inf=0):
    """
    一级反应方程
    A(t) = A_inf + (A0 - A_inf) * exp(-k*t)
    其中:
    - A0: 初始值
    - k: 反应速率常数
    - A_inf: 无穷远处的值
    - t: 时间
    """
    return A_inf + (A0 - A_inf) * np.exp(-k * t)

@dataclass
class CellData:
    """存储单个细胞的数据结构"""
    cell_id: int
    channel1: np.ndarray  # 第一个通道的细胞区域数据
    channel2: np.ndarray  # 第二个通道的细胞区域数据
    file_path: str
    time_point: int
    x_coords: np.ndarray  # 细胞在原图中的x坐标
    y_coords: np.ndarray  # 细胞在原图中的y坐标
    intensity1: np.ndarray = field(init=False)  # 展平的第一个通道强度
    intensity2: np.ndarray = field(init=False)  # 展平的第二个通道强度
    pearson_corr: float = field(init=False, default=np.nan)  # Pearson相关系数
    p_value: float = field(init=False, default=np.nan)
    n_pixels: int = field(init=False)
    
    def __post_init__(self):
        """初始化后处理"""
        # 展平强度数据
        self.intensity1 = self.channel1.flatten()
        self.intensity2 = self.channel2.flatten()
        self.n_pixels = len(self.intensity1)
        
        # 计算Pearson相关系数
        if len(self.intensity1) > 1 and len(self.intensity2) > 1:
            # 移除NaN值
            mask = ~(np.isnan(self.intensity1) | np.isnan(self.intensity2))
            if np.sum(mask) > 1:
                ch1_clean = self.intensity1[mask]
                ch2_clean = self.intensity2[mask]
                if np.std(ch1_clean) > 0 and np.std(ch2_clean) > 0:
                    self.pearson_corr, self.p_value = pearsonr(ch1_clean, ch2_clean)
                else:
                    self.pearson_corr = 0.0  # 如果标准差为0，相关系数为0

class ReactionFitter:
    """反应拟合器类"""
    
    @staticmethod
    def fit_first_order_reaction(time_points, values) -> Dict[str, float]:
        """
        对数据进行一级反应方程拟合
        
        Parameters:
        - time_points: 时间点数组
        - values: 对应的数值数组
        
        Returns:
        - 包含拟合参数和反应时间的字典
        """
        # 移除NaN值
        mask = ~np.isnan(values)
        time_points_clean = time_points[mask]
        values_clean = values[mask]
        
        if len(time_points_clean) < 3:
            return {
                'A0': np.nan,
                'k': np.nan,
                'A_inf': np.nan,
                't50': np.nan,
                't90': np.nan,
                'r_squared': np.nan
            }
        
        # 确定初始值
        A0 = values_clean[0]  # 初始值
        A_inf = values_clean[-1] if len(values_clean) > 1 else A0  # 最终值
        k_guess = 0.1  # 初始k值猜测
        
        # 确保合理的初始值
        if A0 < 0:
            A0 = 0
        if A_inf < 0:
            A_inf = 0
        
        try:
            # 进行曲线拟合
            popt, pcov = curve_fit(first_order_reaction, time_points_clean, values_clean, 
                                 p0=[A0, k_guess, A_inf], maxfev=5000)
            A0_fit, k_fit, A_inf_fit = popt
            
            # 计算拟合优度
            fitted_values = first_order_reaction(time_points_clean, A0_fit, k_fit, A_inf_fit)
            ss_res = np.sum((values_clean - fitted_values) ** 2)
            ss_tot = np.sum((values_clean - np.mean(values_clean)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 1.0
            
            # 计算50%和90%反应时间
            if k_fit > 0:
                # 计算达到目标值的时间
                if A0_fit > A_inf_fit:
                    # 衰减过程
                    target_50 = A_inf_fit + 0.5 * (A0_fit - A_inf_fit)
                    target_90 = A_inf_fit + 0.1 * (A0_fit - A_inf_fit)
                else:
                    # 增长过程
                    target_50 = A0_fit + 0.5 * (A_inf_fit - A0_fit)
                    target_90 = A0_fit + 0.9 * (A_inf_fit - A0_fit)
                
                t50 = -np.log((target_50 - A_inf_fit) / (A0_fit - A_inf_fit)) / k_fit if k_fit > 0 and (A0_fit - A_inf_fit) != 0 else np.nan
                t90 = -np.log((target_90 - A_inf_fit) / (A0_fit - A_inf_fit)) / k_fit if k_fit > 0 and (A0_fit - A_inf_fit) != 0 else np.nan
            else:
                t50 = np.nan
                t90 = np.nan
            
            return {
                'A0': A0_fit,
                'k': k_fit,
                'A_inf': A_inf_fit,
                't50': t50,
                't90': t90,
                'r_squared': r_squared
            }
        except:
            return {
                'A0': np.nan,
                'k': np.nan,
                'A_inf': np.nan,
                't50': np.nan,
                't90': np.nan,
                'r_squared': np.nan
            }

@dataclass
class TimeSeriesAnalysis:
    """时间序列分析的数据结构"""
    file_path: str
    time_points: int
    cells: Dict[int, List[CellData]] = field(default_factory=dict)  # {cell_id: [CellData at time 0, time 1, ...]}
    all_cells: List[CellData] = field(default_factory=list)  # 所有时间点的所有细胞
    skip_initial_frames: int = field(default=0)  # 跳过的初始帧数
    
    def add_cell_data(self, cell_data):
        """添加细胞数据"""
        if cell_data.cell_id not in self.cells:
            self.cells[cell_data.cell_id] = []
        self.cells[cell_data.cell_id].append(cell_data)
        self.all_cells.append(cell_data)
    
    def get_correlation_over_time(self, cell_id: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """获取指定细胞的相关系数随时间的变化"""
        if cell_id not in self.cells:
            raise ValueError(f"Cell {cell_id} not found")
        
        time_points = []
        correlations = []
        p_values = []
        
        for cell_data in self.cells[cell_id]:
            if cell_data.time_point >= self.skip_initial_frames:  # 跳过初始帧
                time_points.append(cell_data.time_point)
                correlations.append(cell_data.pearson_corr)
                p_values.append(cell_data.p_value)
        
        return np.array(time_points), np.array(correlations), np.array(p_values)
    
    def get_intensity_over_time(self, cell_id: int, channel: str) -> Tuple[np.ndarray, np.ndarray]:
        """获取指定细胞的指定通道强度随时间的变化"""
        if cell_id not in self.cells:
            raise ValueError(f"Cell {cell_id} not found")
        
        time_points = []
        intensities = []
        
        for cell_data in self.cells[cell_id]:
            if cell_data.time_point >= self.skip_initial_frames:  # 跳过初始帧
                time_points.append(cell_data.time_point)
                if channel == 'channel1':
                    intensities.append(np.mean(cell_data.intensity1))
                elif channel == 'channel2':
                    intensities.append(np.mean(cell_data.intensity2))
                else:
                    raise ValueError(f"Unknown channel: {channel}")
        
        return np.array(time_points), np.array(intensities)
    
    def fit_cell_reaction(self, cell_id: int) -> Dict[str, float]:
        """
        对细胞数据进行一级反应方程拟合
        
        Parameters:
        - cell_id: 细胞ID
        
        Returns:
        - 包含拟合参数和反应时间的字典
        """
        time_points, correlations, p_values = self.get_correlation_over_time(cell_id)
        
        # 拟合相关系数变化
        correlation_fit = ReactionFitter.fit_first_order_reaction(time_points, correlations)
        
        # 拟合通道强度变化
        ch1_time, ch1_values = self.get_intensity_over_time(cell_id, 'channel1')
        ch1_fit = ReactionFitter.fit_first_order_reaction(ch1_time, ch1_values)
        
        ch2_time, ch2_values = self.get_intensity_over_time(cell_id, 'channel2')
        ch2_fit = ReactionFitter.fit_first_order_reaction(ch2_time, ch2_values)
        
        # 合并结果
        result = {
            'correlation_A0': correlation_fit['A0'],
            'correlation_k': correlation_fit['k'],
            'correlation_A_inf': correlation_fit['A_inf'],
            'correlation_t50': correlation_fit['t50'],
            'correlation_t90': correlation_fit['t90'],
            'correlation_r_squared': correlation_fit['r_squared'],
            'channel1_A0': ch1_fit['A0'],
            'channel1_k': ch1_fit['k'],
            'channel1_A_inf': ch1_fit['A_inf'],
            'channel1_t50': ch1_fit['t50'],
            'channel1_t90': ch1_fit['t90'],
            'channel1_r_squared': ch1_fit['r_squared'],
            'channel2_A0': ch2_fit['A0'],
            'channel2_k': ch2_fit['k'],
            'channel2_A_inf': ch2_fit['A_inf'],
            'channel2_t50': ch2_fit['t50'],
            'channel2_t90': ch2_fit['t90'],
            'channel2_r_squared': ch2_fit['r_squared']
        }
        
        return result
    
    def plot_cell_with_fitting(self, cell_id: int, include_scatter: bool = False, figsize=(12, 10)):
        """绘制指定细胞的完整分析图（包含拟合结果和50%、90%时间点）"""
        time_points, correlations, p_values = self.get_correlation_over_time(cell_id)
        
        # 计算拟合结果
        fit_results = self.fit_cell_reaction(cell_id)
        
        if include_scatter:
            # 如果包含散点图，创建3x2的子图
            fig, axes = plt.subplots(3, 2, figsize=figsize)
        else:
            # 如果不包含散点图，创建2x2的子图
            fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # 1. 相关系数随时间变化（包含拟合曲线）
        ax1 = axes[0, 0] if include_scatter else axes[0, 0]
        ax1.plot(time_points, correlations, 'o-', label='Correlation', linewidth=2, markersize=6)
        
        # 绘制拟合曲线
        if not np.isnan(fit_results['correlation_k']) and fit_results['correlation_k'] > 0:
            t_fit = np.linspace(time_points.min(), time_points.max(), 100)
            y_fit = first_order_reaction(t_fit, fit_results['correlation_A0'], 
                                        fit_results['correlation_k'], fit_results['correlation_A_inf'])
            ax1.plot(t_fit, y_fit, '--', label='Fitted curve', color='red', linewidth=2)
            
            # 标记50%和90%反应时间
            if not np.isnan(fit_results['correlation_t50']):
                ax1.axvline(x=fit_results['correlation_t50'], color='orange', linestyle=':', 
                           label=f't50: {fit_results["correlation_t50"]:.2f}', linewidth=2)
            if not np.isnan(fit_results['correlation_t90']):
                ax1.axvline(x=fit_results['correlation_t90'], color='purple', linestyle=':', 
                           label=f't90: {fit_results["correlation_t90"]:.2f}', linewidth=2)
        
        ax1.set_xlabel('Time Point')
        ax1.set_ylabel('Pearson Correlation')
        ax1.set_title(f'Cell {cell_id} - Correlation Over Time\nR²: {fit_results["correlation_r_squared"]:.3f}')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        ax1.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        
        # 2. P值随时间变化
        ax2 = axes[0, 1] if include_scatter else axes[0, 1]
        ax2.plot(time_points, p_values, 's-', color='red', linewidth=2, markersize=6)
        ax2.set_xlabel('Time Point')
        ax2.set_ylabel('P-value')
        ax2.set_yscale('log')
        ax2.set_title(f'Cell {cell_id} - P-value Over Time')
        ax2.grid(True, alpha=0.3)
        
        # 3. 通道1强度随时间变化
        ch1_time, ch1_values = self.get_intensity_over_time(cell_id, 'channel1')
        ax3 = axes[1, 0] if include_scatter else axes[1, 0]
        ax3.plot(ch1_time, ch1_values, 'o-', label='Channel 1', linewidth=2, markersize=6)
        
        # 绘制拟合曲线
        if not np.isnan(fit_results['channel1_k']) and fit_results['channel1_k'] > 0:
            t_fit = np.linspace(ch1_time.min(), ch1_time.max(), 100)
            y_fit = first_order_reaction(t_fit, fit_results['channel1_A0'], 
                                        fit_results['channel1_k'], fit_results['channel1_A_inf'])
            ax3.plot(t_fit, y_fit, '--', label='Fitted curve', color='red', linewidth=2)
        
        ax3.set_xlabel('Time Point')
        ax3.set_ylabel('Intensity')
        ax3.set_title(f'Cell {cell_id} - Channel 1 Intensity\nR²: {fit_results["channel1_r_squared"]:.3f}')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        # 4. 通道2强度随时间变化
        ch2_time, ch2_values = self.get_intensity_over_time(cell_id, 'channel2')
        ax4 = axes[1, 1] if include_scatter else axes[1, 1]
        ax4.plot(ch2_time, ch2_values, 'o-', label='Channel 2', linewidth=2, markersize=6, color='green')
        
        # 绘制拟合曲线
        if not np.isnan(fit_results['channel2_k']) and fit_results['channel2_k'] > 0:
            t_fit = np.linspace(ch2_time.min(), ch2_time.max(), 100)
            y_fit = first_order_reaction(t_fit, fit_results['channel2_A0'], 
                                        fit_results['channel2_k'], fit_results['channel2_A_inf'])
            ax4.plot(t_fit, y_fit, '--', label='Fitted curve', color='red', linewidth=2)
        
        ax4.set_xlabel('Time Point')
        ax4.set_ylabel('Intensity')
        ax4.set_title(f'Cell {cell_id} - Channel 2 Intensity\nR²: {fit_results["channel2_r_squared"]:.3f}')
        ax4.grid(True, alpha=0.3)
        ax4.legend()
        
        # 5. 散点图（如果需要）
        if include_scatter:
            ax5 = axes[2, 0]
            # 获取该细胞的第一个时间点的数据（或任意时间点）
            first_time_data = next((cd for cd in self.all_cells if cd.cell_id == cell_id and cd.time_point >= self.skip_initial_frames), None)
            if first_time_
                # 移除NaN值进行绘图
                mask = ~(np.isnan(first_time_data.intensity1) | np.isnan(first_time_data.intensity2))
                ch1_clean = first_time_data.intensity1[mask]
                ch2_clean = first_time_data.intensity2[mask]
                
                ax5.scatter(ch1_clean, ch2_clean, alpha=0.6, s=20)
                ax5.set_xlabel('Channel 1 Intensity')
                ax5.set_ylabel('Channel 2 Intensity')
                ax5.set_title(f'Cell {cell_id} - Scatter Plot\nCorr: {first_time_data.pearson_corr:.3f}')
                ax5.grid(True, alpha=0.3)
            
            # 6. 拟合参数摘要
            ax6 = axes[2, 1]
            ax6.axis('off')  # 关闭坐标轴
            summary_text = f"Cell {cell_id} - Fitting Summary\n\n"
            summary_text += f"Correlation:\n"
            summary_text += f"  k = {fit_results['correlation_k']:.4f}\n"
            summary_text += f"  t50 = {fit_results['correlation_t50']:.2f}\n"
            summary_text += f"  t90 = {fit_results['correlation_t90']:.2f}\n"
            summary_text += f"  R² = {fit_results['correlation_r_squared']:.3f}\n\n"
            summary_text += f"Channel1:\n"
            summary_text += f"  k = {fit_results['channel1_k']:.4f}\n"
            summary_text += f"  t50 = {fit_results['channel1_t50']:.2f}\n"
            summary_text += f"  t90 = {fit_results['channel1_t90']:.2f}\n"
            summary_text += f"  R² = {fit_results['channel1_r_squared']:.3f}\n\n"
            summary_text += f"Channel2:\n"
            summary_text += f"  k = {fit_results['channel2_k']:.4f}\n"
            summary_text += f"  t50 = {fit_results['channel2_t50']:.2f}\n"
            summary_text += f"  t90 = {fit_results['channel2_t90']:.2f}\n"
            summary_text += f"  R² = {fit_results['channel2_r_squared']:.3f}"
            
            ax6.text(0.1, 0.9, summary_text, transform=ax6.transAxes, fontsize=10,
                    verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))

        plt.tight_layout()
        return fig, axes

class FileMatcher:
    """文件匹配器类"""
    
    @staticmethod
    def calculate_filename_similarity(nd2_stem: str, mask_stem: str) -> float:
        """
        计算两个文件名的相似度
        返回0-1之间的分数，1表示完全匹配
        """
        # 移除常见的后缀
        nd2_clean = nd2_stem.replace(' -1', '').replace(' -2', '').replace(' -3', '').replace(' -4', '')
        mask_clean = mask_stem.replace(' -1', '').replace(' -2', '').replace(' -3', '').replace(' -4', '')
        
        # 移除常见的蒙版标识
        mask_clean = mask_clean.replace('_mask', '').replace('_seg', '').replace('_segmentation', '')
        mask_clean = mask_clean.replace('_channel_1_frame_0_cp_masks', '').replace('_cp_masks', '')
        
        # 如果完全匹配（考虑数字后缀）
        if nd2_stem == mask_stem or nd2_clean == mask_clean:
            return 1.0
        
        # 如果ND2文件名是蒙版文件名的前缀
        if mask_stem.startswith(nd2_stem):
            return 0.9
        
        # 如果清理后的名字匹配
        if nd2_clean == mask_clean:
            return 0.8
        
        # 计算最长公共子串的长度作为相似度
        common_length = len(os.path.commonprefix([nd2_clean.lower(), mask_clean.lower()]))
        max_length = max(len(nd2_clean), len(mask_clean))
        
        if max_length == 0:
            return 0.0
        
        similarity = common_length / max_length
        
        # 如果有很高的相似度，返回这个值，否则返回较低的值
        if similarity > 0.7:
            return similarity
        else:
            return 0.0
    
    @staticmethod
    def find_matching_mask(nd2_file_path: str, mask_pattern: Optional[str] = None) -> Optional[str]:
        """
        根据ND2文件路径找到匹配的蒙版文件
        
        Parameters:
        - nd2_file_path: ND2文件路径
        - mask_pattern: 蒙版文件的通配符模式
        
        Returns:
        - 匹配的蒙版文件路径，如果未找到则返回None
        """
        nd2_path = Path(nd2_file_path)
        nd2_stem = nd2_path.stem  # 完整文件名（不含扩展名）
        nd2_parent = nd2_path.parent
        
        # 如果提供了mask_pattern，使用glob模式查找所有蒙版
        if mask_pattern:
            mask_files = glob.glob(str(nd2_parent / mask_pattern))
            mask_files = [f for f in mask_files if Path(f).suffix.lower() != '.nd2']
        else:
            # 收集目录中所有可能的蒙版文件
            mask_extensions = ['.npy', '.tif', '.tiff', '.png', '.jpg', '.jpeg']
            mask_files = []
            for ext in mask_extensions:
                mask_files.extend(glob.glob(str(nd2_parent / f"*{ext}")))
            # 过滤掉ND2文件
            mask_files = [f for f in mask_files if Path(f).suffix.lower() != '.nd2']
        
        # 为当前ND2文件寻找最佳匹配的蒙版
        best_match = None
        best_score = -1
        
        for mask_file in mask_files:
            mask_path = Path(mask_file)
            mask_stem = mask_path.stem
            
            # 计算匹配分数
            score = FileMatcher.calculate_filename_similarity(nd2_stem, mask_stem)
            
            if score > best_score:
                best_score = score
                best_match = mask_file
        
        # 只有当匹配分数足够高时才返回匹配结果
        if best_score >= 0.5:  # 设置一个阈值
            return best_match
        else:
            return None
    
    @staticmethod
    def match_nd2_with_masks(nd2_files: List[str], mask_pattern: Optional[str] = None) -> Dict[str, Optional[str]]:
        """
        为所有ND2文件匹配蒙版
        
        Returns:
        - 字典，键为ND2文件路径，值为对应的蒙版路径（或None）
        """
        matches = {}
        
        # 收集所有蒙版文件
        if mask_pattern:
            # 如果提供了特定的mask_pattern，只使用该模式匹配的蒙版
            parent_dir = Path(nd2_files[0]).parent if nd2_files else Path('.')
            all_mask_files = glob.glob(str(parent_dir / mask_pattern))
            all_mask_files = [f for f in all_mask_files if Path(f).suffix.lower() != '.nd2']
        else:
            # 否则收集所有可能的蒙版文件
            parent_dir = Path(nd2_files[0]).parent if nd2_files else Path('.')
            mask_extensions = ['.npy', '.tif', '.tiff', '.png', '.jpg', '.jpeg']
            all_mask_files = []
            for ext in mask_extensions:
                all_mask_files.extend(glob.glob(str(parent_dir / f"*{ext}")))
            all_mask_files = [f for f in all_mask_files if Path(f).suffix.lower() != '.nd2']
        
        print(f"Found {len(all_mask_files)} potential mask files:")
        for mask_file in all_mask_files:
            print(f"  {mask_file}")
        
        # 为每个ND2文件找到最佳匹配的蒙版
        for nd2_file in nd2_files:
            nd2_path = Path(nd2_file)
            nd2_stem = nd2_path.stem
            
            best_match = None
            best_score = -1
            
            for mask_file in all_mask_files:
                mask_path = Path(mask_file)
                mask_stem = mask_path.stem
                
                score = FileMatcher.calculate_filename_similarity(nd2_stem, mask_stem)
                
                if score > best_score:
                    best_score = score
                    best_match = mask_file
            
            # 只有当匹配分数足够高时才认为匹配成功
            if best_score >= 0.5:
                matches[nd2_file] = best_match
                print(f"Matched {nd2_file} with {best_match} (score: {best_score:.2f})")
            else:
                matches[nd2_file] = None
                print(f"No good match found for {nd2_file} (best score: {best_score:.2f})")
        
        return matches

def load_mask(mask_path: str) -> np.ndarray:
    """
    加载蒙版并确保其为整数类型
    
    Parameters:
    - mask_path: 蒙版文件路径
    
    Returns:
    - 整数类型的蒙版数组
    """
    if mask_path.endswith(('.npy', '.npz')):
        mask = np.load(mask_path)
    else:
        # 使用skimage读取图像
        mask = io.imread(mask_path)
        
        # 如果是RGB图像，转换为灰度
        if len(mask.shape) == 3:
            mask = np.mean(mask, axis=2).astype(mask.dtype)
        
        # 检查是否为整数类型
        if mask.dtype.kind not in ['u', 'i']:  # 不是无符号整数或有符号整数
            raise ValueError(f"Mask file {mask_path} is not integer type. Got dtype: {mask.dtype}")
    
    # 确保是整数类型
    if mask.dtype.kind not in ['u', 'i']:  # 不是无符号整数或有符号整数
        raise ValueError(f"Mask file {mask_path} is not integer type. Got dtype: {mask.dtype}")
    
    return mask

class FluorescenceAnalyzer:
    """荧光共聚焦图像分析器"""
    
    def __init__(self):
        self.analyses: List[TimeSeriesAnalysis] = []
    
    def load_nd2_file(self, file_path: str, mask_path: Optional[str] = None, skip_initial_frames: int = 0) -> TimeSeriesAnalysis:
        """
        加载ND2文件并分析
        
        Parameters:
        - file_path: ND2文件路径
        - mask_path: 蒙版文件路径
        - skip_initial_frames: 跳过的初始帧数
        """
        print(f"Loading file: {file_path}")
        
        if mask_path is None:
            print(f"Warning: No mask provided for {file_path}, skipping this file")
            return None
        
        print(f"Using mask: {mask_path}")
        
        # 读取蒙版
        try:
            mask = load_mask(mask_path)
            print(f"Mask shape: {mask.shape}, dtype: {mask.dtype}, unique values: {np.unique(mask[:100, :100]) if mask.size > 10000 else np.unique(mask)}")
        except Exception as e:
            print(f"Error reading mask {mask_path}: {e}")
            return None
        
        # 使用nd2.imread读取整个文件到numpy数组
        with nd2.ND2File(file_path) as nd2_file:
            img_array = nd2_file.asarray()  # 读取整个图像数组
        
        # 获取图像信息
        shape = img_array.shape
        if len(shape) == 4:
            time_points, channels, height, width = shape
        elif len(shape) == 3:
            if 'c' in nd2_file.sizes and nd2_file.sizes['C'] > 1:
                channels, height, width = shape
                time_points = 1
            else:
                time_points, channels, height, width = 1, 1, shape[0], shape[1]
        else:
            raise ValueError(f"Unexpected image shape: {shape}")
        
        print(f"Image shape: {shape}, Time points: {time_points}, Channels: {channels}")
        
        # 验证图像和蒙版尺寸匹配
        if (height, width) != mask.shape:
            raise ValueError(f"Image shape {(height, width)} doesn't match mask shape {mask.shape}")
        
        # 创建分析对象
        analysis = TimeSeriesAnalysis(
            file_path=file_path,
            time_points=time_points,
            skip_initial_frames=skip_initial_frames
        )
        
        # 获取细胞ID - 确保是整数类型
        unique_cells = np.unique(mask)
        unique_cells = unique_cells[unique_cells > 0]  # 排除背景（0）
        
        # 确保所有cell_id都是整数
        unique_cells = unique_cells.astype(int)
        
        print(f"Found {len(unique_cells)} cells in mask: {unique_cells[:10]}...")  # 只显示前10个
        
        # 处理每个时间点
        for t in range(time_points):
            print(f"Processing time point {t+1}/{time_points}")
            
            # 获取当前时间点的图像
            if time_points > 1:
                current_img = img_array[t]  # shape: (channels, height, width)
            else:
                current_img = img_array  # 如果只有一个时间点，可能是 (channels, height, width)
                if len(current_img.shape) == 2:  # 如果确实是2D，添加通道维度
                    current_img = current_img[np.newaxis, :, :]
            
            # 验证通道数
            if current_img.shape[0] < 2:
                print(f"Warning: Not enough channels for correlation analysis in {file_path}")
                continue
            
            # 处理每个细胞
            for cell_id in unique_cells:
                # 获取细胞在图像中的坐标
                cell_mask = (mask == cell_id)
                y_coords, x_coords = np.where(cell_mask)
                
                if len(y_coords) == 0:
                    continue
                
                # 提取通道数据
                channel1 = current_img[0][cell_mask]  # 第一个通道
                channel2 = current_img[1][cell_mask]  # 第二个通道
                
                # 创建CellData对象
                cell_data = CellData(
                    cell_id=int(cell_id),  # 确保是整数
                    channel1=channel1,
                    channel2=channel2,
                    file_path=file_path,
                    time_point=t,
                    x_coords=x_coords,
                    y_coords=y_coords
                )
                
                analysis.add_cell_data(cell_data)
        
        self.analyses.append(analysis)
        return analysis
    
    def process_files_with_masks(self, nd2_files: List[str], mask_pattern: Optional[str] = None, skip_initial_frames: int = 0) -> List[TimeSeriesAnalysis]:
        """处理多个ND2文件，自动匹配蒙版"""
        # 首先匹配所有ND2文件和蒙版
        matches = FileMatcher.match_nd2_with_masks(nd2_files, mask_pattern)
        
        results = []
        
        for nd2_file in nd2_files:
            mask_path = matches[nd2_file]
            if mask_path:
                print(f"Processing {nd2_file} with mask {mask_path}")
                analysis = self.load_nd2_file(nd2_file, mask_path, skip_initial_frames)
                if analysis:
                    results.append(analysis)
            else:
                print(f"Skipping {nd2_file} - no matching mask found")
        
        return results
    
    def get_summary_dataframe(self) -> pd.DataFrame:
        """获取所有细胞的汇总数据"""
        data = []
        for analysis in self.analyses:
            for cell_data in analysis.all_cells:
                # 只包含在skip_initial_frames之后的数据
                if cell_data.time_point >= analysis.skip_initial_frames:
                    data.append({
                        'file_path': cell_data.file_path,
                        'cell_id': cell_data.cell_id,
                        'time_point': cell_data.time_point,
                        'pearson_corr': cell_data.pearson_corr,
                        'p_value': cell_data.p_value,
                        'n_pixels': cell_data.n_pixels,
                        'mean_ch1': np.mean(cell_data.intensity1),
                        'mean_ch2': np.mean(cell_data.intensity2),
                        'std_ch1': np.std(cell_data.intensity1),
                        'std_ch2': np.std(cell_data.intensity2)
                    })
        
        return pd.DataFrame(data)
    
    def get_reaction_fitting_results(self) -> pd.DataFrame:
        """获取一级反应拟合结果"""
        data = []
        for analysis in self.analyses:
            for cell_id in analysis.cells.keys():
                # 拟合相关系数变化
                fit_results = analysis.fit_cell_reaction(cell_id)
                
                data.append({
                    'file_path': analysis.file_path,
                    'cell_id': cell_id,
                    **fit_results  # 展开fit_results字典
                })
        
        return pd.DataFrame(data)

def main():
    parser = argparse.ArgumentParser(description='Analyze fluorescence co-localization in ND2 files')
    parser.add_argument('nd2_pattern', type=str, 
                       help='Pattern for ND2 files (e.g., "data/*.nd2" or "data/**/*.nd2")')
    parser.add_argument('--mask-pattern', type=str, default=None,
                       help='Pattern for mask files (e.g., "*.npy", "*_mask.npy"). If not provided, will try to auto-match based on filename similarity.')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory for results. If not specified, creates a directory based on input pattern.')
    parser.add_argument('--save-results', action='store_true',
                       help='Save results to CSV file')
    parser.add_argument('--skip-initial-frames', type=int, default=0,
                       help='Number of initial frames to skip (not used for fitting)')
    parser.add_argument('--include-scatter', action='store_true',
                       help='Include scatter plot in cell analysis figures')
    
    args = parser.parse_args()
    
    # 使用glob查找所有匹配的ND2文件
    nd2_files = glob.glob(args.nd2_pattern)
    nd2_files = natsorted(nd2_files)  # 自然排序，确保文件顺序正确
    
    if not nd2_files:
        print(f"No ND2 files found matching pattern: {args.nd2_pattern}")
        return
    
    print(f"Found {len(nd2_files)} ND2 files:")
    for f in nd2_files:
        print(f"  {Path(f).name}")  # 只显示文件名，不显示完整路径
    
    # 设置输出目录
    if args.output_dir is None:
        # 基于输入模式创建输出目录名
        input_pattern_path = Path(args.nd2_pattern)
        if '*' in args.nd2_pattern:
            base_name = input_pattern_path.parent.name or "analysis"
        else:
            base_name = input_pattern_path.stem
        output_dir = f"output_{base_name}"
    else:
        output_dir = args.output_dir
    
    # 自动创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # 创建分析器
    analyzer = FluorescenceAnalyzer()
    
    # 处理文件
    analyses = analyzer.process_files_with_masks(nd2_files, args.mask_pattern, args.skip_initial_frames)
    
    if not analyses:
        print("No files were successfully processed (no matching masks found)")
        return
    
    # 获取汇总数据
    summary_df = analyzer.get_summary_dataframe()
    print(f"\nProcessed {len(summary_df)} cell-time points")
    print("First few rows (showing key information):")
    
    # 显示更友好的格式
    display_cols = ['file_path', 'cell_id', 'time_point', 'pearson_corr', 'p_value', 'n_pixels']
    display_df = summary_df[display_cols].copy()
    display_df['file_path'] = display_df['file_path'].apply(lambda x: Path(x).name)  # 只显示文件名
    print(display_df.head())
    
    # 获取反应拟合结果
    print("\nPerforming first-order reaction fitting...")
    reaction_df = analyzer.get_reaction_fitting_results()
    print(f"Reaction fitting results for {len(reaction_df)} cells:")
    print("First few rows of reaction fitting:")
    reaction_display = reaction_df[['file_path', 'cell_id', 'correlation_k', 'correlation_t50', 'correlation_t90', 'correlation_r_squared']].copy()
    reaction_display['file_path'] = reaction_display['file_path'].apply(lambda x: Path(x).name)
    print(reaction_display.head())
    
    # 保存结果
    if args.save_results:
        # 保存基础汇总数据
        output_file = os.path.join(output_dir, 'correlation_analysis_results.csv')
        summary_df.to_csv(output_file, index=False)
        print(f"Basic results saved to: {output_file}")
        
        # 保存反应拟合结果
        reaction_output_file = os.path.join(output_dir, 'reaction_fitting_results.csv')
        reaction_df.to_csv(reaction_output_file, index=False)
        print(f"Reaction fitting results saved to: {reaction_output_file}")
    
    # 可视化示例
    if analyses:
        print("\nGenerating visualizations...")
        
        # 为每个文件中每个细胞绘制完整的分析图（包含拟合和50%、90%时间点）
        for i, analysis in enumerate(analyses):
            print(f"Processing file {i+1}: {Path(analysis.file_path).name}")
            for cell_id in analysis.cells.keys():
                print(f"  Plotting analysis for cell {cell_id}")
                fig, axes = analysis.plot_cell_with_fitting(cell_id, include_scatter=args.include_scatter)
                
                # 保存图片
                fig.savefig(os.path.join(output_dir, f'file_{i+1}_cell_{cell_id}_analysis.png'), 
                            dpi=300, bbox_inches='tight')
        
        plt.show()

if __name__ == "__main__":
    main()
