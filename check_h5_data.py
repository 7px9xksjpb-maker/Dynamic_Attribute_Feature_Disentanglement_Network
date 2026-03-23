import h5py
import numpy as np
import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 导入 3D 绘图支持
import sys
import io

# 修复 Windows 系统中文乱码问题
if sys.platform.startswith('win'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

def inspect_h5_file(file_path: str, show_plot: bool = True):
    print(f"正在读取文件: {file_path}\n" + "="*40)
    
    try:
        with h5py.File(file_path, 'r') as f:
            # 1. 检查全局属性 (Attributes)
            print("【全局属性 (Attributes)】")
            for key, val in f.attrs.items():
                if key == 'angle_ranges_json':
                    val = json.loads(val)
                print(f"  - {key:<20}: {val}")
            
            print("\n" + "="*40)
            
            # 2. 检查数据集结构 (Datasets Structure)
            print("【数据集结构 (Datasets)】")
            for key in f.keys():
                dataset = f[key]
                print(f"  - {key:<10}: 维度={dataset.shape}, 类型={dataset.dtype}")
                
            print("\n" + "="*40)
            
            # 3. 严谨性数据统计 (Sanity Checks)
            print("【数据严谨性检查 (Sanity Checks)】")
            x_data = f['x_data'][:]
            y_data = f['y_data'][:]
            z_data = f['z_data'][:]
            
            # 标签检查
            unique_labels = np.unique(y_data)
            print(f"  - 包含的标签种类 : {unique_labels}")
            
            # 角度范围检查
            print(f"  - 角度 (Z) 范围  : {np.min(z_data):.4f}° ~ {np.max(z_data):.4f}°")
            
            # 异常值与极值检查 (L2 归一化后的数据范围应在 [0, 1] 之间)
            print(f"  - X 数据包含 NaN : {np.isnan(x_data).any()}")
            print(f"  - X 数据最大值   : {np.max(x_data):.4f}")
            print(f"  - X 数据最小值   : {np.min(x_data):.4f}")
            
            # 4. 发散性扩展：随机抽样可视化 (2D Waterfall + 3D Surface)
            if show_plot and len(x_data) > 0:
                plot_random_hrrp_sequence(x_data, y_data, z_data)

    except FileNotFoundError:
        print(f"错误: 找不到文件 {file_path}。请检查路径是否正确。")
    except Exception as e:
        print(f"发生错误: {e}")

def plot_random_hrrp_sequence(x_data, y_data, z_data):
    """
    发散性扩展：绘制一个随机采样的 HRRP 序列
    同时包含 2D 瀑布图和 3D 曲面图，以便从不同维度评估信号特征。
    """
    sample_idx = np.random.randint(0, len(x_data))
    seq_matrix = x_data[sample_idx]  # 形状: (sequence_length, num_bins)
    label = y_data[sample_idx]
    angles = z_data[sample_idx]
    
    sequence_length, num_bins = seq_matrix.shape
    
    # 创建一个 1行2列 的宽幅画布
    fig = plt.figure(figsize=(16, 6))
    
    # ================== 子图 1: 2D 瀑布图 (Waterfall) ==================
    ax1 = fig.add_subplot(1, 2, 1)
    im1 = ax1.imshow(seq_matrix, aspect='auto', cmap='jet', origin='lower')
    fig.colorbar(im1, ax=ax1, label='Normalized Magnitude')
    
    ax1.set_title(f"2D HRRP Waterfall | Sample: {sample_idx} | Label: {label}\n"
                  f"Angle Range: {angles[0]:.2f}° to {angles[-1]:.2f}°")
    ax1.set_xlabel("Range Bins")
    ax1.set_ylabel("Sequence Steps (Angle Variation)")
    
    # ================== 子图 2: 3D 曲面图 (Surface) ==================
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    
    # 动态生成与 seq_matrix 维度完全匹配的 X/Y 网格
    x_vec = np.arange(num_bins)         # X轴: 距离库
    y_vec = np.arange(sequence_length)  # Y轴: 序列索引
    X, Y = np.meshgrid(x_vec, y_vec)
    
    # 绘制 3D 表面
    # 使用 cmap='jet' 保持与雷达领域常用的伪彩图一致
    surf = ax2.plot_surface(X, Y, seq_matrix, cmap='jet', edgecolor='none')
    
    # 添加 3D 图的颜色条
    cbar2 = fig.colorbar(surf, ax=ax2, shrink=0.5, aspect=10)
    cbar2.set_label('Normalized Magnitude', rotation=270, labelpad=15)
    
    ax2.set_title(f"3D HRRP Surface | Label: {label}")
    ax2.set_xlabel('Range Bins')
    ax2.set_ylabel('Sequence Steps')
    ax2.set_zlabel('Amplitude')
    
    # 根据实际 L2 归一化后的数值范围动态设置 Z 轴刻度，避免图形扁平或溢出
    z_min, z_max = np.min(seq_matrix), np.max(seq_matrix)
    ax2.set_zlim(0, z_max * 1.1) 
    
    # 设置合理的初始观察视角 (仰角 30度，方位角 -120度)
    ax2.view_init(elev=30, azim=-120)
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    # 替换为你实际的输出路径
    h5_file_path = r'D:\HRRP\A02\DA_Feature\h5_car_seq\seq_data.h5'
    inspect_h5_file(h5_file_path, show_plot=True)