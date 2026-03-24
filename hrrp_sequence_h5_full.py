import json
import os
import random
from typing import Dict, List, Sequence, Tuple

import h5py
import numpy as np
import scipy.io as sio

seed = 42
np.random.seed(seed)
random.seed(seed)

# ================= 1. Configuration =================

data_dir = r'D:\HRRP\A02\SupConLearning\HRRP_Car10'
output_dir = r'D:\HRRP\A02\DA_Feature\h5_car_seq'
os.makedirs(output_dir, exist_ok=True)

original_step_angle = 0.0625
target_step_angle = 1
stride = int(round(target_step_angle / original_step_angle))
if stride <= 0:
    raise ValueError(f'Invalid stride={stride}. Check target_step_angle/original_step_angle.')

# 统一角度范围 
angle_ranges = [
    (0, 180), 
]

file_specs: List[Tuple[str, int]] = [
   # 类别 0: HondaCivic4dr 的多个俯仰角
    ('HondaCivic4dr_el30.0000.mat', 0),
    ('HondaCivic4dr_el40.0000.mat', 0),
    ('HondaCivic4dr_el50.0000.mat', 0),
    ('HondaCivic4dr_el60.0000.mat', 0),
    
    # 类别 1: Mitsubishi 的多个俯仰角
    ('Mitsubishi_el30.0000.mat', 1),
    ('Mitsubishi_el40.0000.mat', 1),
    ('Mitsubishi_el50.0000.mat', 1),
    ('Mitsubishi_el60.0000.mat', 1),

    # 类别 2: Sentra 的多个俯仰角
    ('Sentra_el30.0000.mat', 2),
    ('Sentra_el40.0000.mat', 2),
    ('Sentra_el50.0000.mat', 2),
    ('Sentra_el60.0000.mat', 2),

    # 类别 3: Camry 的多个俯仰角
    ('Camry_el30.0000.mat', 3),
    ('Camry_el40.0000.mat', 3),
    ('Camry_el50.0000.mat', 3),
    ('Camry_el60.0000.mat', 3),

]
    ## 其他类别
    # ('Jeep93_el30.0000.mat', 4),
    # ('Jeep99_el30.0000.mat', 5),
    # ('Maxima_el30.0000.mat', 6),
    # ('MazdaMPV_el30.0000.mat', 7),
    # ('ToyotaAvalon_el30.0000.mat', 8),
    # ('ToyotaTacoma_el30.0000.mat', 9),

# Sequence settings expected by the training code
sequence_length = 30
slide_step = 1
save_mask = True
train_ratio = 0.7  # 新增：训练集比例

print('--- Configuration ---')
print(f'original_step_angle = {original_step_angle}')
print(f'target_step_angle   = {target_step_angle}')
print(f'stride              = {stride}')
print(f'angle_ranges        = {angle_ranges}')
print(f'sequence_length     = {sequence_length}')
print(f'slide_step          = {slide_step}')
print(f'train_ratio         = {train_ratio}')
print('---------------------\n')


# ================= 2. Utility functions =================

def validate_file_specs(specs: Sequence[Tuple[str, int]]) -> None:
    filename_to_labels: Dict[str, set] = {}
    for filename, label in specs:
        filename_to_labels.setdefault(filename, set()).add(label)

    conflicts = {k: sorted(v) for k, v in filename_to_labels.items() if len(v) > 1}
    if conflicts:
        msg = (
            'The following file names are mapped to multiple labels, which will corrupt the dataset:\n'
            + json.dumps(conflicts, indent=2, ensure_ascii=False)
            + '\nPlease fix file_specs before running.'
        )
        raise ValueError(msg)

# 修改：计算整体矩阵的 Frobenius 范数进行统一缩放
def global_normalize(chunk_data: np.ndarray) -> np.ndarray:
    mag = np.abs(chunk_data).astype(np.float32)
    norm_val = np.linalg.norm(mag)  # 计算整个矩阵的 L2 范数
    if norm_val == 0.0:
        return mag
    return mag / norm_val


def extract_snapshots_by_ranges(
    hrrp_matrix: np.ndarray,
    angle_ranges: Sequence[Tuple[float, float]],
    label: int,
    original_step: float,
    stride_val: int,
    target_step: float,
) -> Tuple[List[np.ndarray], List[int], List[float]]:
    x_list: List[np.ndarray] = []
    y_list: List[int] = []
    z_list: List[float] = []

    num_bins, num_angle_points = hrrp_matrix.shape

    for start_deg, end_deg in angle_ranges:
        start_idx = int(round(start_deg / original_step))
        end_idx = int(round(end_deg / original_step))

        if start_idx >= num_angle_points:
            continue
        end_idx = min(end_idx, num_angle_points)
        if end_idx <= start_idx:
            continue

        chunk_data = hrrp_matrix[:, start_idx:end_idx:stride_val]
        if chunk_data.shape[1] == 0:
            continue
            
        # 替换为整体归一化
        chunk_data = global_normalize(chunk_data)
        angles = start_deg + np.arange(chunk_data.shape[1], dtype=np.float32) * target_step

        for col in range(chunk_data.shape[1]):
            x_list.append(chunk_data[:, col].astype(np.float32))
            y_list.append(int(label))
            z_list.append(float(angles[col]))

    return x_list, y_list, z_list


def build_sliding_sequences(
    x_list: Sequence[np.ndarray],
    y_list: Sequence[int],
    z_list: Sequence[float],
    seq_len: int,
    step: int,
) -> Tuple[List[np.ndarray], List[int], List[np.ndarray], List[np.ndarray]]:
    if len(x_list) != len(y_list) or len(x_list) != len(z_list):
        raise ValueError('x_list, y_list, z_list must have the same length.')

    X_seq_list: List[np.ndarray] = []
    Y_seq_list: List[int] = []
    Z_seq_list: List[np.ndarray] = []
    M_seq_list: List[np.ndarray] = []

    if len(x_list) < seq_len:
        return X_seq_list, Y_seq_list, Z_seq_list, M_seq_list

    num_bins = x_list[0].shape[0]
    for start in range(0, len(x_list) - seq_len + 1, step):
        end = start + seq_len
        seq_x = np.stack(x_list[start:end], axis=0).astype(np.float32)   
        seq_y = int(y_list[start])
        seq_z = np.asarray(z_list[start:end], dtype=np.float32)          
        seq_m = np.ones((seq_len, num_bins), dtype=np.float32)

        X_seq_list.append(seq_x)
        Y_seq_list.append(seq_y)
        Z_seq_list.append(seq_z)
        M_seq_list.append(seq_m)

    return X_seq_list, Y_seq_list, Z_seq_list, M_seq_list


def process_file_to_sequences(
    file_path: str,
    label: int,
    angle_ranges: Sequence[Tuple[float, float]],
    original_step: float,
    stride_val: int,
    target_step: float,
    seq_len: int,
    slide_step_val: int,
) -> Tuple[List[np.ndarray], List[int], List[np.ndarray], List[np.ndarray]]:
    mat_data = sio.loadmat(file_path)
    hrrp_complex = mat_data['data']['hv'][0, 0]

    X_all: List[np.ndarray] = []
    Y_all: List[int] = []
    Z_all: List[np.ndarray] = []
    M_all: List[np.ndarray] = []

    for start_deg, end_deg in angle_ranges:
        x_list, y_list, z_list = extract_snapshots_by_ranges(
            hrrp_matrix=hrrp_complex,
            angle_ranges=[(start_deg, end_deg)],
            label=label,
            original_step=original_step,
            stride_val=stride_val,
            target_step=target_step,
        )

        x_seq, y_seq, z_seq, m_seq = build_sliding_sequences(
            x_list=x_list,
            y_list=y_list,
            z_list=z_list,
            seq_len=seq_len,
            step=slide_step_val,
        )

        X_all.extend(x_seq)
        Y_all.extend(y_seq)
        Z_all.extend(z_seq)
        M_all.extend(m_seq)

    return X_all, Y_all, Z_all, M_all


def shuffle_in_unison(*arrays: np.ndarray) -> Tuple[np.ndarray, ...]:
    if not arrays:
        return tuple()
    n = len(arrays[0])
    indices = np.random.permutation(n)
    return tuple(arr[indices] for arr in arrays)


def save_h5(x: np.ndarray, y: np.ndarray, z: np.ndarray, m: np.ndarray, filename: str) -> None:
    if len(x) == 0:
        print(f'Warning: {filename} is empty, skip saving.')
        return

    save_path = os.path.join(output_dir, filename)
    with h5py.File(save_path, 'w') as f:
        f.create_dataset('x_data', data=x, compression='gzip')
        f.create_dataset('y_data', data=y, compression='gzip')
        f.create_dataset('z_data', data=z, compression='gzip')
        if save_mask:
            f.create_dataset('mask_data', data=m, compression='gzip')

        f.attrs['sequence_length'] = int(x.shape[1])
        f.attrs['num_bins'] = int(x.shape[2])
        f.attrs['num_samples'] = int(x.shape[0])
        f.attrs['original_step_angle'] = float(original_step_angle)
        f.attrs['target_step_angle'] = float(target_step_angle)
        f.attrs['slide_step'] = int(slide_step)
        f.attrs['class_count'] = int(len(np.unique(y)))
        f.attrs['angle_ranges_json'] = json.dumps(angle_ranges)

    print(f'\nSaved: {save_path}')
    print(f'  Samples        = {x.shape[0]}')


# ================= 3. Main loop =================

def main() -> None:
    validate_file_specs(file_specs)

    dataset = {'x': [], 'y': [], 'z': [], 'mask': []}

    for file_name, label in file_specs:
        file_path = os.path.join(data_dir, file_name)
        print(f'Reading: {file_name} (label={label})')

        try:
            x_all, y_all, z_all, m_all = process_file_to_sequences(
                file_path=file_path,
                label=label,
                angle_ranges=angle_ranges,
                original_step=original_step_angle,
                stride_val=stride,
                target_step=target_step_angle,
                seq_len=sequence_length,
                slide_step_val=slide_step,
            )
            dataset['x'].extend(x_all)
            dataset['y'].extend(y_all)
            dataset['z'].extend(z_all)
            dataset['mask'].extend(m_all)

        except Exception as e:
            print(f'Error processing {file_name}: {e}')

    print('-' * 40)
    
    # 转换为 numpy 数组
    x = np.asarray(dataset['x'], dtype=np.float32)
    y = np.asarray(dataset['y'], dtype=np.int64)
    z = np.asarray(dataset['z'], dtype=np.float32)
    m = np.asarray(dataset['mask'], dtype=np.float32)

    # 统一打乱
    x, y, z, m = shuffle_in_unison(x, y, z, m)

    # 训练/测试划分 (7:3)
    split_idx = int(len(x) * train_ratio)
    
    x_train, y_train, z_train, m_train = x[:split_idx], y[:split_idx], z[:split_idx], m[:split_idx]
    x_test, y_test, z_test, m_test = x[split_idx:], y[split_idx:], z[split_idx:], m[split_idx:]

    print(f'Total sequences: {len(x)}')
    save_h5(x_train, y_train, z_train, m_train, 'train_seq_data.h5')
    save_h5(x_test, y_test, z_test, m_test, 'test_seq_data.h5')
    print('-' * 40)
    print('Done.')

if __name__ == '__main__':
    main()