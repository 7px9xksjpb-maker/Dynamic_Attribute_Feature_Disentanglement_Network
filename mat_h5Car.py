import os
import scipy.io as sio
import numpy as np
import h5py
import random
import sys
import io
from sklearn.utils import shuffle  # 引入shuffle用于打乱最终数据

# 修复 Windows 系统中文乱码问题
if sys.platform.startswith('win'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# 设置随机种子
seed = 42
np.random.seed(seed)
random.seed(seed)

# ================= 1. 配置区域 =================

# 路径设置
data_dir = r'D:\HRRP\A02\SupConLearning\HRRP_Car10'
output_dir = r'D:\HRRP\A02\SupConLearning\h5_car2'
os.makedirs(output_dir, exist_ok=True)

# 数据参数
original_step_angle = 0.0625  # 原始步长 0.0625度
target_step_angle = 0.0625     # 目标步长
stride = int(target_step_angle / original_step_angle)  

# === 自定义角度范围 ===
# 格式：[(起始角度, 结束角度), (起始角度, 结束角度), ...]
train_angle_ranges = [
   (75, 81)
]

test_angle_ranges = [
   (84, 87)
]

# 类别定义
classes = {
    'HondaCivic4dr': 0,
    'Mitsubishi': 1,
    'Sentra': 2,
    'Camry': 3,
    'Jeep93': 4,
    'Jeep99': 5,
    'MazdaMPV': 6,
    'Mitsubishi': 7,
    'ToyotaAvalon': 8,
    'ToyotaTacoma': 9,
}
file_names = [
    'HondaCivic4dr_el30.0000.mat',
    'Mitsubishi_el30.0000.mat',
    'Sentra_el30.0000.mat',
    'Camry_el30.0000.mat',
    'Jeep93_el30.0000.mat',
    'Jeep99_el30.0000.mat',
    'MazdaMPV_el30.0000.mat',
    'Mitsubishi_el30.0000.mat',
    'ToyotaAvalon_el30.0000.mat',
    'ToyotaTacoma_el30.0000.mat'
]

print(f"--- 配置确认 ---")
print(f"原始步长: {original_step_angle}° -> 目标步长: {target_step_angle}° (Stride={stride})")
print(f"训练集角度范围: {train_angle_ranges}")
print(f"测试集角度范围: {test_angle_ranges}")
print(f"----------------\n")

# ================= 2. 核心处理函数 =================

def extract_data_by_ranges(hrrp_matrix, angle_ranges, label, original_step, stride_val):
    """
    根据给定的角度范围，从原始矩阵中提取数据
    """
    x_list, y_list, z_list = [], [], []
    
    for start_deg, end_deg in angle_ranges:
        # 1. 将角度转换为索引
        # 公式: Index = Angle / 0.0625
        start_idx = int(start_deg / original_step)
        end_idx = int(end_deg / original_step)
        
        # 边界保护
        if start_idx >= hrrp_matrix.shape[1]: continue
        if end_idx > hrrp_matrix.shape[1]: end_idx = hrrp_matrix.shape[1]
        
        # 2. 切片提取 (包含降采样 stride)
        # hrrp_matrix 形状通常是 (HRRP长度, 角度点数)
        chunk_data = hrrp_matrix[:, start_idx:end_idx:stride_val]
        
        if chunk_data.shape[1] == 0:
            continue
            
        # 3. 处理每一列 (HRRP样本)
        for col in range(chunk_data.shape[1]):
            # 取模
            mag = np.abs(chunk_data[:, col])
            
            # 归一化 (L2范数)
            norm = np.linalg.norm(mag)
            if norm != 0:
                mag = mag / norm
            
            x_list.append(mag)
            
            # 计算对应的真实角度
            # 当前列在 chunk 中的索引 * 步长 + 起始角度
            current_angle = start_deg + (col * target_step_angle)
            z_list.append(current_angle)
            y_list.append(label)
            
    return x_list, y_list, z_list

# ================= 3. 主循环 =================

# 初始化容器
train_data = {'x': [], 'y': [], 'z': []}
test_data  = {'x': [], 'y': [], 'z': []}

for file_name, label in zip(file_names, classes.values()):
    file_path = os.path.join(data_dir, file_name)
    print(f"正在读取: {file_name} (类别: {label})")
    
    try:
        mat_data = sio.loadmat(file_path)
        # 获取原始复数矩阵
        hrrp_complex = mat_data['data']['hv'][0, 0] 
        
        # --- 提取训练集部分 ---
        x_tr, y_tr, z_tr = extract_data_by_ranges(
            hrrp_complex, train_angle_ranges, label, original_step_angle, stride
        )
        train_data['x'].extend(x_tr)
        train_data['y'].extend(y_tr)
        train_data['z'].extend(z_tr)
        print(f"  -> 加入训练集: {len(x_tr)} 个样本")

        # --- 提取测试集部分 ---
        x_te, y_te, z_te = extract_data_by_ranges(
            hrrp_complex, test_angle_ranges, label, original_step_angle, stride
        )
        test_data['x'].extend(x_te)
        test_data['y'].extend(y_te)
        test_data['z'].extend(z_te)
        print(f"  -> 加入测试集: {len(x_te)} 个样本")
        
    except Exception as e:
        print(f"处理文件 {file_name} 时出错: {e}")

# ================= 4. 数据整合与保存 =================

def process_and_save(data_dict, filename):
    if not data_dict['x']:
        print(f"警告: {filename} 数据为空，跳过保存。")
        return

    # 转换为 NumPy 数组
    x = np.array(data_dict['x'], dtype=np.float32)
    y = np.array(data_dict['y'], dtype=np.int32)
    z = np.array(data_dict['z'], dtype=np.float32)

    # *** 打乱数据 (Shuffle) ***
    # 因为我们是按文件顺序读取的，必须打乱，否则训练会有问题
    x, y, z = shuffle(x, y, z, random_state=seed)

    # 保存
    path = os.path.join(output_dir, filename)
    with h5py.File(path, 'w') as f:
        f.create_dataset('x_data', data=x)
        f.create_dataset('y_data', data=y)
        f.create_dataset('z_data', data=z)
        
    print(f"\n已保存: {path}")
    print(f"  最终形状: {x.shape}")
    print(f"  角度范围示例: {z.min():.2f}° - {z.max():.2f}°")

print("-" * 30)
process_and_save(train_data, 'train_data.h5')
process_and_save(test_data, 'test_data.h5')
print("-" * 30)
print("处理完成！")