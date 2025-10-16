import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import argparse
import os
import glob

# --- 模型函数 ---
# Langmuir 吸附动力学模型 / 一级增长模型
# y = y0 + (y_max - y0) * (1 - exp(-k_app * t))
# y0: 初始值 (t=0 时的值, 拟合的初始值)
# k_app: 表观速率常数 (k_app > 0 for growth)
# y_max: 长时间反应后的平衡值/最大值 (拟合的最大值)
def reaction_model(t, y0, k_app, y_max):
    """
    Langmuir吸附动力学模型函数 (一级增长)。
    """
    return y0 + (y_max - y0) * (1.0 - np.exp(-k_app * t))

def negative_log_likelihood(params, t_data, y_data):
    """
    计算负对数似然函数 (等价于最小二乘)。
    params: [y0, k_app, y_max]
    """
    y0, k_app, y_max = params
    y_pred = reaction_model(t_data, y0, k_app, y_max)

    if not np.all(np.isfinite(y_pred)):
        return np.inf

    residuals = y_data - y_pred
    if not np.all(np.isfinite(residuals)):
         print(f"Warning: Invalid residuals found with params {params}")
         return np.inf

    rss = np.sum(residuals**2)
    return rss

def fit_data(t_data, y_data, initial_guess=None):
    """
    使用 scipy.minimize 执行拟合。
    """
    if initial_guess is None:
        y0_guess = y_data.iloc[0] if not y_data.empty else 0.5
        y_max_guess = y_data.iloc[-1] if not y_data.empty else 1.0 # 假设最终值接近最大值
        k_app_guess = 0.1 # 默认值
        if len(t_data) > 1 and y_max_guess > y0_guess:
             if y_data.iloc[-1] < y_max_guess and t_data.iloc[-1] > 0:
                 ratio = (y_max_guess - y_data.iloc[-1]) / (y_max_guess - y0_guess)
                 if 0 < ratio < 1:
                     k_app_guess = -np.log(ratio) / t_data.iloc[-1]
        initial_guess = [y0_guess, k_app_guess, y_max_guess]

    # 定义参数边界
    y_data_max = y_data.max()
    y_data_min = y_data.min()
    # 确保 y0_guess 和 y_max_guess 在合理范围内
    y0_min = min(y_data_min, initial_guess[0])
    y0_max = max(y_data_min, initial_guess[0])
    y_max_min = min(y_data_max, initial_guess[2])
    y_max_max = max(y_data_max, initial_guess[2], y0_max) # y_max should be >= y0_max
    bounds = [(y0_min, y0_max), (1e-6, 10), (y_max_min, y_max_max)] # (y0, k_app, y_max)

    result = minimize(negative_log_likelihood, initial_guess, args=(t_data.values, y_data.values), method='L-BFGS-B', bounds=bounds)

    if not result.success:
        print(f"Warning: Optimization failed for initial guess {initial_guess}. Message: {result.message}")
        return None, None, None, result.message, np.nan, np.nan

    y0_fit, k_app_fit, y_max_fit = result.x

    # --- 拟合效果评估 ---
    y_pred_fit = reaction_model(t_data.values, y0_fit, k_app_fit, y_max_fit)
    residuals = y_data.values - y_pred_fit
    rss = np.sum(residuals**2)
    tss = np.sum((y_data.values - y_data.values.mean())**2) # Total Sum of Squares
    if tss == 0:
        r_squared = np.nan # If all y_data is the same, R^2 is undefined
    else:
        r_squared = 1 - (rss / tss)

    return y0_fit, k_app_fit, y_max_fit, result.message, rss, r_squared

def calculate_times(y0, k_app, y_max, target_fraction_50=0.5, target_fraction_90=0.9):
    """
    基于 Langmuir 模型拟合参数计算 T50 和 T90 时间。
    T50: 达到 (y_max - y0) * 0.5 + y0 的时间
    T90: 达到 (y_max - y0) * 0.9 + y0 的时间
    y_target = y0 + (y_max - y0) * (1 - exp(-k_app*T))
    (y_target - y0) / (y_max - y0) = 1 - exp(-k_app*T)
    exp(-k_app*T) = 1 - (y_target - y0) / (y_max - y0)
    -k_app*T = ln( 1 - (y_target - y0) / (y_max - y0) )
    T = -ln( (y_max - y_target) / (y_max - y0) ) / k_app
    """
    if k_app <= 0:
        print("Warning: Rate constant k_app is not positive, cannot calculate meaningful T50/T90 for growth.")
        return np.nan, np.nan

    y_range = y_max - y0
    if y_range <= 0:
        print("Warning: y_max is not greater than y0, no growth occurs or initial parameters are invalid.")
        return np.nan, np.nan

    y_target_50 = y0 + y_range * target_fraction_50
    y_target_90 = y0 + y_range * target_fraction_90

    if not (y0 <= y_target_50 <= y_max and y0 <= y_target_90 <= y_max):
        print(f"Warning: Calculated target values ({y_target_50}, {y_target_90}) are outside the range [{y0}, {y_max}] defined by y0 ({y0}) and y_max ({y_max}). Cannot calculate T50/T90.")
        return np.nan, np.nan

    try:
        # 使用推导出的公式
        t50 = -np.log((y_max - y_target_50) / y_range) / k_app
        t90 = -np.log((y_max - y_target_90) / y_range) / k_app
    except (ValueError, ZeroDivisionError) as e:
        print(f"Error calculating T50/T90: {e}. y0={y0}, y_max={y_max}, k_app={k_app}, y_target_50={y_target_50}, y_target_90={y_target_90}")
        return np.nan, np.nan

    t50 = max(0, t50)
    t90 = max(0, t90)

    return t50, t90

def plot_single_fit(df, time_col, y_col, y0_fit, k_app_fit, y_max_fit, t50, t90, r_squared, filename, output_dir, baseline_value):
    """生成并保存单个文件的拟合图"""
    fig, ax = plt.subplots(figsize=(8, 6))
    # 绘制原始数据点
    ax.scatter(df[time_col], df[y_col], label='Data', color='blue', alpha=0.6)
    t_smooth = np.linspace(df[time_col].min(), df[time_col].max(), 200)
    y_smooth = reaction_model(t_smooth, y0_fit, k_app_fit, y_max_fit)
    # 在图例中加入 R_squared
    ax.plot(t_smooth, y_smooth, label=f'Fit: y = {y0_fit:.3f} + ({y_max_fit:.3f} - {y0_fit:.3f})*(1 - exp(-{k_app_fit:.3f}*t)), R² = {r_squared:.4f}', color='red')

    if not np.isnan(t50):
        ax.axvline(x=t50, color='green', linestyle='--', label=f'T50 = {t50:.2f}s')
    if not np.isnan(t90):
        ax.axvline(x=t90, color='orange', linestyle='--', label=f'T90 = {t90:.2f}s')

    ax.set_xlabel('Time (s)')
    ax.set_ylabel(y_col)
    ax.set_title(f'Fitted Curve for {os.path.basename(filename)} (Langmuir Adsorption)')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)

    output_path = os.path.join(output_dir, f"fit_{os.path.basename(filename)}.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved single fit plot: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Fit colocalization data (growing trend) to Langmuir adsorption kinetics using glob pattern. Baseline frame is included in baseline calculation but fitting starts after it.')
    parser.add_argument('--input', type=str, required=True, help='Glob pattern to match input CSV files (e.g., "*.csv", "data_*.csv", "/path/to/files/*.csv")')
    parser.add_argument('--frame-to-sec', type=float, required=True, help='Conversion factor from frame to seconds (e.g., seconds per frame)')
    parser.add_argument('--baseline', type=int, required=True, help='Frame number used to calculate the baseline. Data up to and including this frame will be used to calculate the baseline. Fitting starts from the frame after this one.')
    parser.add_argument('--output_dir', type=str, default='output_fits', help='Directory to save output plots and results (default: output_fits)')
    parser.add_argument('--plot_indices', type=int, nargs='*', help='Indices of matched files to include in the combined plot (0-based). If not provided, all matched files are included.')

    args = parser.parse_args()

    # 使用 glob 查找匹配的文件
    input_files = glob.glob(args.input)
    if not input_files:
        print(f"Error: No files found matching pattern '{args.input}'")
        return # 退出脚本

    print(f"Found {len(input_files)} file(s) matching pattern '{args.input}':")
    for f in input_files:
        print(f"  - {f}")

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    results_summary = []
    all_fitted_curves = {}
    all_fitted_curves_normalized = {}

    for i, filename in enumerate(input_files):
        print(f"\nProcessing file {i+1}/{len(input_files)}: {filename}")
        try:
            df = pd.read_csv(filename)
        except FileNotFoundError:
            print(f"Error: File not found: {filename}")
            continue
        except pd.errors.EmptyDataError:
            print(f"Error: File is empty: {filename}")
            continue
        except Exception as e:
            print(f"Error reading file {filename}: {e}")
            continue

        if 'frame' not in df.columns or 'pearson_above' not in df.columns:
            print(f"Error: Required columns 'frame' or 'pearson_above' not found in {filename}")
            continue

        # 计算基线值 (包含 baseline 帧)
        baseline_mask = df['frame'] <= args.baseline # <--- 修改：包含 baseline 帧
        if baseline_mask.any():
            baseline_value = df.loc[baseline_mask, 'pearson_above'].mean()
            print(f"  Baseline calculated from frames <= {args.baseline}: {baseline_value:.6f}")
        else:
            print(f"  Warning: No frames found before or at baseline frame {args.baseline}. Using 0 as baseline.")
            baseline_value = 0.0

        # 转换帧到时间
        df['time_seconds'] = df['frame'] * args.frame_to_sec

        # 拟合数据 (不包含 baseline 帧及之前的数据)
        # 拟合从 baseline 帧的下一帧开始
        fit_mask = df['frame'] > args.baseline # <--- 修改：从 baseline 帧的下一帧开始拟合
        if not fit_mask.any():
             print(f"  Warning: No frames found after baseline frame {args.baseline}. Skipping fit.")
             continue

        t_data_fit = df.loc[fit_mask, 'time_seconds']
        y_data_fit = df.loc[fit_mask, 'pearson_above'] # <--- 修改：使用原始 pearson_above

        y0_fit, k_app_fit, y_max_fit, fit_message, rss, r_squared = fit_data(t_data_fit, y_data_fit)
        if y0_fit is None:
            print(f"Skipping {filename} due to fit failure: {fit_message}")
            continue

        # 计算 T50, T90
        t50, t90 = calculate_times(y0_fit, k_app_fit, y_max_fit)

        # 保存结果到摘要
        # y0_fit, y_max_fit 是拟合得到的原始值
        results_summary.append({
            'file': os.path.basename(filename),
            'y0_fitted': y0_fit, # 拟合的初始值 (对应拟合起始时间点)
            'k_app_fitted': k_app_fit,
            'y_max_fitted': y_max_fit, # 拟合的最大值
            'T50_seconds': t50,
            'T90_seconds': t90,
            'baseline_used': baseline_value, # 记录使用的基线值
            'R_squared': r_squared, # 记录 R_squared
            'RSS': rss, # 记录残差平方和 (可选)
            'fit_message': fit_message
        })

        # 生成单个文件的拟合图 (使用原始数据)
        plot_single_fit(df, 'time_seconds', 'pearson_above', y0_fit, k_app_fit, y_max_fit, t50, t90, r_squared, filename, args.output_dir, baseline_value)

        # 为多文件比较图准备数据 (使用原始数据的拟合曲线)
        # 使用所有时间点，但曲线基于拟合参数，代表原始数据的模型
        time_smooth = np.linspace(df['time_seconds'].min(), df['time_seconds'].max(), 200)
        y_smooth = reaction_model(time_smooth, y0_fit, k_app_fit, y_max_fit)
        all_fitted_curves[os.path.basename(filename)] = (time_smooth, y_smooth)
        # 归一化: (y - y0) / (y_max - y0) (基于拟合的原始值)
        y_range = y_max_fit - y0_fit
        if y_range != 0: # Avoid division by zero
            y_smooth_norm = (y_smooth - y0_fit) / y_range
        else:
            y_smooth_norm = y_smooth # If no range, normalization is identity
        all_fitted_curves_normalized[os.path.basename(filename)] = (time_smooth, y_smooth_norm)

    # 保存结果摘要到 CSV
    if results_summary:
        summary_df = pd.DataFrame(results_summary)
        summary_path = os.path.join(args.output_dir, "fitting_summary.csv")
        summary_df.to_csv(summary_path, index=False)
        print(f"\nSaved fitting summary to: {summary_path}")
    else:
        print("\nNo files were successfully fitted. Skipping summary CSV creation.")
        return # Exit if no data to plot combined graphs

    # 生成多文件比较图
    all_keys = list(all_fitted_curves.keys())
    if args.plot_indices is not None:
        # 检查索引是否有效
        valid_indices = [i for i in args.plot_indices if 0 <= i < len(all_keys)]
        if len(valid_indices) != len(args.plot_indices):
            print(f"Warning: Some indices in --plot_indices are out of range. Valid range: 0 to {len(all_keys)-1}. Using valid indices only.")
        selected_keys = [all_keys[i] for i in valid_indices]
    else:
        selected_keys = all_keys # All keys if no indices specified

    if not selected_keys:
        print("No valid files selected for combined plot.")
        return

    # 1. 未归一化图 (原始值)
    fig, ax = plt.subplots(figsize=(10, 7))
    for key in selected_keys:
        if key in all_fitted_curves:
            t_data, y_data = all_fitted_curves[key]
            ax.plot(t_data, y_data, label=f'Fit: {key}', marker='', linestyle='-', linewidth=2)
        else:
            print(f"Warning: Key {key} not found in fitted curves for combined plot.")

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('pearson_above')
    ax.set_title('Fitted Curves Comparison (Unnormalized)')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    combined_path_unnorm = os.path.join(args.output_dir, "combined_fits_unnormalized.png")
    plt.savefig(combined_path_unnorm, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved combined unnormalized fit plot: {combined_path_unnorm}")

    # 2. 归一化图 (原始值)
    fig, ax = plt.subplots(figsize=(10, 7))
    for key in selected_keys:
        if key in all_fitted_curves_normalized:
            t_data, y_data_norm = all_fitted_curves_normalized[key]
            ax.plot(t_data, y_data_norm, label=f'Fit Norm: {key}', marker='', linestyle='-', linewidth=2)
        else:
            print(f"Warning: Key {key} not found in fitted curves normalized for combined plot.")

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('(pearson_above - y0) / (y_max - y0)')
    ax.set_title('Fitted Curves Comparison (Normalized)')
    ax.set_ylim(-0.05, 1.05)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    combined_path_norm = os.path.join(args.output_dir, "combined_fits_normalized.png")
    plt.savefig(combined_path_norm, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved combined normalized fit plot: {combined_path_norm}")


if __name__ == "__main__":
    main()
