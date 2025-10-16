import os
import glob
import shutil
import argparse

def collect_files_to_single_folder(source_dir, target_base_dir=None):
    """
    将源目录中的相关文件（除mrc外）收集到一个文件夹中，不创建子文件夹
    
    Args:
        source_dir: 源目录路径，包含各dataset的子文件夹
        target_base_dir: 目标基础目录路径，存放整合后的文件。如果为None，则在source_dir下创建
    """
    
    # 如果没有指定target_base_dir，则在source_dir下创建一个
    if target_base_dir is None:
        target_base_dir = os.path.join(source_dir, "collected_files")
    
    # 确保目标目录存在
    os.makedirs(target_base_dir, exist_ok=True)
    
    # 获取源目录中的所有子文件夹
    subfolders = [f for f in os.listdir(source_dir) if 
                  os.path.isdir(os.path.join(source_dir, f))]
    
    if not subfolders:
        print(f"在 {source_dir} 中没有找到子文件夹")
        return
    
    for folder in subfolders:
        folder_path = os.path.join(source_dir, folder)
        
        # 查找该文件夹中的edf文件
        edf_files = glob.glob(os.path.join(folder_path, "*.edf"))
        
        if not edf_files:
            print(f"在 {folder_path} 中没有找到edf文件，跳过")
            continue
        
        if len(edf_files) > 1:
            print(f"在 {folder_path} 中找到多个edf文件，使用第一个: {edf_files[0]}")
        
        edf_file = edf_files[0]
        
        # 从edf文件中提取dataset名称
        dataset_name = extract_dataset_name(edf_file)
        
        if not dataset_name:
            print(f"无法从 {edf_file} 中提取dataset名称，跳过")
            continue
        
        # 如果dataset_name不匹配文件夹名，给出警告
        if dataset_name != folder:
            print(f"警告: DatasetName ({dataset_name}) 与文件夹名 ({folder}) 不匹配")
        
        # 定义需要复制的文件类型（除了mrc）
        file_extensions = ['.xf', '.tlt', '.xtilt']
        
        # 复制相关文件到目标文件夹（直接在目标目录下，不创建子文件夹）
        for ext in file_extensions:
            source_file = os.path.join(folder_path, dataset_name + ext)
            
            if os.path.exists(source_file):
                # 目标文件直接放在目标目录下，保持原文件名
                target_file = os.path.join(target_base_dir, dataset_name + ext)
                
                # 检查源文件和目标文件是否是同一个文件
                if os.path.abspath(source_file) == os.path.abspath(target_file):
                    print(f"源文件和目标文件相同，跳过: {source_file}")
                    continue
                
                shutil.copy2(source_file, target_file)
                print(f"已复制: {source_file} -> {target_file}")
            else:
                print(f"警告: 文件不存在: {source_file}")
        
        # 复制edf文件
        target_edf = os.path.join(target_base_dir, dataset_name + '.edf')
        
        # 检查源edf文件和目标edf文件是否是同一个文件
        if os.path.abspath(edf_file) != os.path.abspath(target_edf):
            shutil.copy2(edf_file, target_edf)
            print(f"已复制: {edf_file} -> {target_edf}")
        else:
            print(f"源EDF文件和目标EDF文件相同，跳过: {edf_file}")
        
        print(f"完成处理dataset: {dataset_name}")
        print("-" * 50)

def extract_dataset_name(edf_file):
    """
    从edf文件中提取DatasetName
    """
    try:
        with open(edf_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        lines = content.split('\n')
        for line in lines:
            if 'Setup.DatasetName' in line and '=' in line:
                dataset_name = line.split('=')[1].strip()
                return dataset_name
    except Exception as e:
        print(f"读取edf文件失败: {e}")
        return None
    
    return None

def verify_collected_files(target_dir):
    """
    验证目标目录中是否包含所有必需的文件（除mrc外）
    """
    all_files = os.listdir(target_dir)
    
    # 获取所有唯一的dataset名称
    datasets = set()
    for file in all_files:
        name, ext = os.path.splitext(file)
        if ext in ['.edf', '.xf', '.tlt', '.xtilt']:
            datasets.add(name)
    
    print(f"\n找到 {len(datasets)} 个dataset:")
    for dataset in sorted(datasets):
        print(f"\n验证 {dataset}:")
        required_files = [f"{dataset}.edf", f"{dataset}.xf", 
                         f"{dataset}.tlt", f"{dataset}.xtilt"]
        
        for file in required_files:
            file_path = os.path.join(target_dir, file)
            if os.path.exists(file_path):
                print(f"  ✓ {file}")
            else:
                print(f"  ✗ {file} (缺失)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='将各dataset子文件夹中的文件收集到一个文件夹中')
    parser.add_argument('input', type=str, 
                        help='源目录路径，包含各dataset的子文件夹')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='目标目录路径，存放整合后的文件。如果未指定，则在source目录下创建collected_files文件夹')
    
    args = parser.parse_args()
    
    # 收集文件（不包括mrc）
    collect_files_to_single_folder(args.input, args.output)
    
    # 验证结果
    target_dir = args.output if args.output else os.path.join(args.input, "collected_files")
    verify_collected_files(target_dir)
