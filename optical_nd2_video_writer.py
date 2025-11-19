import cv2
import nd2
import numpy as np
from typing import Dict, List, Tuple, Optional
import argparse

def milliseconds_to_hmsms(ms: float) -> str:
    """将毫秒转换为 HH:MM:SS.SS 格式。"""
    total_seconds = ms / 1000.0
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = int(total_seconds % 60)
    subseconds = int((total_seconds - int(total_seconds)) * 100) # 取小数点后两位作为百分之一秒
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{subseconds:02d}"

def get_channel_color_map(num_channels: int) -> List[Tuple[int, int, int]]:
    """为不同通道分配颜色 (B, G, R)。"""
    # 定义常用颜色，可以扩展
    colors = [
        (0, 0, 255),   # Red
        (0, 255, 0),   # Green
        (255, 0, 0),   # Blue
        (0, 255, 255), # Yellow
        (255, 0, 255), # Magenta
        (255, 255, 0), # Cyan
    ]
    if num_channels <= len(colors):
        return colors[:num_channels]
    else:
        # 如果通道数超过预定义颜色数，可以循环使用或生成更多颜色
        print(f"警告: 通道数 ({num_channels}) 超过预设颜色数。将循环使用颜色。")
        extended_colors = []
        for i in range(num_channels):
            extended_colors.append(colors[i % len(colors)])
        return extended_colors

def add_scale_bar(image: np.ndarray, pixel_size_um: float, target_scale_length_um: float = 10):
    """
    在图像上添加比例尺。
    :param image: OpenCV 图像 (BGR)
    :param pixel_size_um: 每个像素的物理尺寸 (微米)
    :param target_scale_length_um: 目标比例尺长度 (微米)
    """
    height, width = image.shape[:2]
    length_px = int(target_scale_length_um / pixel_size_um)
    
    # 确保比例尺不会太长
    if length_px > width * 0.3: # 比例尺不超过图像宽度的30%
        length_px = int(width * 0.3)
        actual_length_um = length_px * pixel_size_um
    else:
        actual_length_um = target_scale_length_um

    # 位置 (图像右下角)
    start_point = (width - length_px - 10, height - 20)
    end_point = (width - 10, height - 20)
    # 线条和文字颜色 (白色)
    color = (255, 255, 255)
    # 绘制线段
    cv2.line(image, start_point, end_point, color, 3)
    # 添加文字
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 3
    thickness = 2
    text = f"{actual_length_um:.1f} um"
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    text_x = width - 10 - length_px // 2 - text_size[0] // 2
    text_y = height - 30
    cv2.putText(image, text, (text_x, text_y), font, font_scale, color, thickness, cv2.LINE_AA)

def add_timestamp(image: np.ndarray, time_str: str):
    """在图像上添加时间戳。"""
    height, width = image.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 3
    thickness = 2
    color = (255, 255, 255) # 白色
    # 获取文字大小以便定位
    text_size = cv2.getTextSize(time_str, font, font_scale, thickness)[0]
    # 位置 (左上角)
    text_x = 10
    text_y = text_size[1] + 10
    cv2.putText(image, time_str, (text_x, text_y), font, font_scale, color, thickness, cv2.LINE_AA)

def create_video_from_nd2(
    nd2_file_path: str,
    output_video_path: str,
    selected_channels: List[int],
    frame_range: Optional[Tuple[int, int]] = None,
    fps: int = 10,
    target_scale_length_um: float = 10.0
):
    """
    从 ND2 文件创建视频。
    :param nd2_file_path: 输入 ND2 文件路径
    :param output_video_path: 输出视频文件路径
    :param selected_channels: 要合并的通道索引列表，例如 [0] 或 [0, 1]
    :param frame_range: 要处理的帧范围 (start, end)，例如 (0, 10) 表示处理前10帧
    :param fps: 输出视频的帧率
    :param target_scale_length_um: 比例尺的目标长度 (微米)
    """
    with nd2.ND2File(nd2_file_path) as nd2_file:
        # 1. 获取图像信息
        attrs = nd2_file.sizes # 例如 {'T': 38, 'C': 2, 'Y': 1200, 'X': 1200}
        print(f"图像尺寸: {attrs}")
        
        # 2. 获取元数据
        try:
            voxel_size_um = nd2_file.voxel_size()[0] # Z轴步长, 通常ZYX顺序, 取Y方向步长
            # nd2 库的 voxel_size() 可能返回 (dz, dy, dx) 或 (dy, dx)。我们假设 dy 是 Y 方向的物理尺寸。
            # 如果 voxel_size() 返回 None 或不准确，可能需要从其他元数据获取。
            if voxel_size_um is None or voxel_size_um <= 0:
                # 尝试从 metadata 中获取
                try:
                    # 这是常见的路径，具体路径可能因 ND2 版本而异
                    pixel_microns = nd2_file.metadata.contents.channelLoopPfsState[0].positionX[1] - nd2_file.metadata.contents.channelLoopPfsState[0].positionX[0]
                    if pixel_microns == 0: # 如果X方向没有步长信息，尝试Y方向
                        pixel_microns = nd2_file.metadata.contents.channelLoopPfsState[0].positionY[1] - nd2_file.metadata.contents.channelLoopPfsState[0].positionY[0]
                    voxel_size_um = abs(pixel_microns) if pixel_microns != 0 else 0.1 # 默认值
                except:
                    print("警告: 无法获取像素物理尺寸，使用默认值 0.1 微米。")
                    voxel_size_um = 0.1
            print(f"估算的像素物理尺寸 (Y方向): {voxel_size_um} 微米")
        except Exception as e:
            print(f"警告: 无法获取像素物理尺寸，使用默认值 0.1 微米。错误: {e}")
            voxel_size_um = 0.1

        # 3. 解析帧范围
        total_frames = attrs.get('T', 1)
        if frame_range:
            start_frame, end_frame = frame_range
            start_frame = max(0, min(start_frame, total_frames - 1))
            end_frame = max(start_frame + 1, min(end_frame, total_frames))
        else:
            start_frame = 0
            end_frame = total_frames
        print(f"处理帧范围: {start_frame} 到 {end_frame} (不包含)")

        # 4. 获取图像数据
        # nd2_file.asarray() 返回形状为 (T, C, Y, X) 的 numpy 数组
        all_data = nd2_file.asarray()
        print(f"数据数组形状: {all_data.shape}")

        # 5. 初始化视频写入器
        height, width = all_data.shape[2], all_data.shape[3]
        fourcc = cv2.VideoWriter_fourcc(*'H264')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height), isColor=True)

        # 6. 获取通道颜色映射
        num_channels = attrs.get('C', 1)
        color_map = get_channel_color_map(num_channels)
        print(f"可用通道数: {num_channels}, 选定通道: {selected_channels}")
        print(f"通道颜色映射: {dict(enumerate(color_map))}")

        # 7. 循环处理选定帧
        for t_idx in range(start_frame, end_frame):
            # 创建一个空的RGB图像
            combined_frame = np.zeros((height, width, 3), dtype=np.uint8)

            # --- 新增: 存储时间戳，使用第一个处理通道的时间 ---
            timestamp_str = f"Frame {t_idx}" # 默认值
            first_channel_processed = False

            # 遍历选定的通道
            for c_idx in selected_channels:
                if c_idx >= num_channels:
                    print(f"警告: 通道索引 {c_idx} 超出范围 (0-{num_channels-1})，跳过。")
                    continue

                # 获取当前帧当前通道的图像 (Y, X)
                channel_image = all_data[t_idx, c_idx, :, :]

                # 归一化并转换为8位
                if channel_image.dtype != np.uint8:
                    # 假设输入是整数类型，先归一化到0-1，再转换到0-255
                    max_val = channel_image.max()
                    if max_val > 0:
                         normalized = (channel_image.astype(np.float32) / max_val) * 255.0
                         channel_image_8bit = np.clip(normalized, 0, 255).astype(np.uint8)
                    else:
                         channel_image_8bit = channel_image.astype(np.uint8)
                else:
                     channel_image_8bit = channel_image

                # 获取该通道对应的颜色 (B, G, R)
                color = color_map[c_idx]

                # 创建一个单色通道的彩色图像
                colored_channel = np.zeros((height, width, 3), dtype=np.uint8)
                for i in range(3): # B, G, R
                    colored_channel[:, :, i] = cv2.convertScaleAbs(channel_image_8bit, alpha=(color[i] / 255.0))

                # 将彩色通道累加到合并帧上
                combined_frame = cv2.add(combined_frame, colored_channel)

                # --- 获取并存储时间戳，只获取一次 ---
                if not first_channel_processed:
                    seq_index = t_idx
                    print(f"正在获取第{seq_index}帧")
                    try:
                        # 调用 frame_metadata 方法并传入 seq_index
                        metadata = nd2_file.frame_metadata(seq_index)
                        time_ms = metadata.channels[c_idx].time.relativeTimeMs
                        timestamp_str = milliseconds_to_hmsms(time_ms)
                    except KeyError:
                        print(f"警告: 帧 {t_idx}, 通道 {c_idx} 的元数据中未找到 'relativeTimeMs'")
                    except Exception as e:
                        print(f"警告: 获取帧 {t_idx}, 通道 {c_idx} 元数据时出错: {e}")
                    
                    first_channel_processed = True # 标记已处理第一个通道

            # 8. 添加比例尺
            add_scale_bar(combined_frame, voxel_size_um, target_scale_length_um)

            # 9. 添加时间戳 (使用第一个处理通道获取的时间)
            add_timestamp(combined_frame, timestamp_str)

            # 10. 写入视频帧
            out.write(combined_frame)
            print(f"已处理帧: {t_idx + 1} / {end_frame}")

        # 11. 释放资源
        out.release()
        print(f"视频已保存至: {output_video_path}")

def main():
    parser = argparse.ArgumentParser(description='将 ND2 文件中的 timelapse 共聚焦图片转换为视频。')
    parser.add_argument('input_nd2', help='输入的 ND2 文件路径')
    parser.add_argument('output_video', help='输出的视频文件路径 (例如 output.mp4)')
    parser.add_argument('--channels', nargs='+', type=int, required=True,
                        help='要包含在视频中的通道索引 (从0开始)，例如 --channels 0 1')
    parser.add_argument('--frames', nargs=2, type=int, metavar=('START', 'END'),
                        help='要处理的帧范围 [START, END)，例如 --frames 0 10')
    parser.add_argument('--fps', type=int, default=30,
                        help='输出视频的帧率 (默认: 10)')
    parser.add_argument('--scale_um', type=float, default=10.0,
                        help='视频上显示的比例尺长度 (微米) (默认: 10.0)')

    args = parser.parse_args()

    print(f"输入 ND2: {args.input_nd2}")
    print(f"输出视频: {args.output_video}")
    print(f"选定通道: {args.channels}")
    print(f"帧范围: {args.frames}")
    print(f"输出FPS: {args.fps}")
    print(f"比例尺长度: {args.scale_um} µm")

    create_video_from_nd2(
        nd2_file_path=args.input_nd2,
        output_video_path=args.output_video,
        selected_channels=args.channels,
        frame_range=args.frames,
        fps=args.fps,
        target_scale_length_um=args.scale_um
    )

if __name__ == "__main__":
    main()
