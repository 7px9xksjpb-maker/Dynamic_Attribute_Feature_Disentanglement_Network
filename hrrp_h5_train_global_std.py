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
output_dir = r'D:\HRRP\A02\DA_Feature\h5_car_seq3'
os.makedirs(output_dir, exist_ok=True)

original_step_angle = 0.0625
target_step_angle = 1
stride = int(round(target_step_angle / original_step_angle))
if stride <= 0:
    raise ValueError(f'Invalid stride={stride}. Check target_step_angle/original_step_angle.')

angle_ranges = [
    (0, 360),
]

file_specs: List[Tuple[str, int]] = [
    ('HondaCivic4dr_el30.0000.mat', 0),
    ('HondaCivic4dr_el40.0000.mat', 0),
    ('HondaCivic4dr_el50.0000.mat', 0),
    ('HondaCivic4dr_el60.0000.mat', 0),

    ('Mitsubishi_el30.0000.mat', 1),
    ('Mitsubishi_el40.0000.mat', 1),
    ('Mitsubishi_el50.0000.mat', 1),
    ('Mitsubishi_el60.0000.mat', 1),

    ('Sentra_el30.0000.mat', 2),
    ('Sentra_el40.0000.mat', 2),
    ('Sentra_el50.0000.mat', 2),
    ('Sentra_el60.0000.mat', 2),

    ('Camry_el30.0000.mat', 3),
    ('Camry_el40.0000.mat', 3),
    ('Camry_el50.0000.mat', 3),
    ('Camry_el60.0000.mat', 3),
]

sequence_length = 30
slide_step = 1
save_mask = True
train_ratio = 0.7

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


def snapshot_magnitude(chunk_data: np.ndarray) -> np.ndarray:
    """只取幅值，不做逐帧归一化，后续统一用训练集 mean/std 标准化。"""
    return np.abs(chunk_data).astype(np.float32)


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

    _, num_angle_points = hrrp_matrix.shape

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

        chunk_data = snapshot_magnitude(chunk_data)
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


def stratified_train_test_split(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    m: np.ndarray,
    ratio: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_idx_list: List[np.ndarray] = []
    test_idx_list: List[np.ndarray] = []

    for cls in np.unique(y):
        cls_idx = np.where(y == cls)[0]
        cls_idx = cls_idx[np.random.permutation(len(cls_idx))]
        n_train = int(round(len(cls_idx) * ratio))
        n_train = min(max(n_train, 1), len(cls_idx) - 1) if len(cls_idx) > 1 else len(cls_idx)
        train_idx_list.append(cls_idx[:n_train])
        test_idx_list.append(cls_idx[n_train:])

    train_idx = np.concatenate(train_idx_list)
    test_idx = np.concatenate(test_idx_list)
    train_idx = train_idx[np.random.permutation(len(train_idx))]
    test_idx = test_idx[np.random.permutation(len(test_idx))]

    return (
        x[train_idx], y[train_idx], z[train_idx], m[train_idx],
        x[test_idx], y[test_idx], z[test_idx], m[test_idx],
    )


def fit_global_mean_std(x_train: np.ndarray) -> Tuple[float, float]:
    mean = float(x_train.mean())
    std = float(x_train.std())
    if std < 1e-6:
        std = 1e-6
    return mean, std


def apply_global_standardization(x: np.ndarray, mean: float, std: float) -> np.ndarray:
    return ((x - mean) / std).astype(np.float32)


def save_h5(x: np.ndarray, y: np.ndarray, z: np.ndarray, m: np.ndarray, filename: str, mean: float, std: float) -> None:
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
        f.attrs['normalization'] = 'train_global_mean_std'
        f.attrs['train_global_mean'] = float(mean)
        f.attrs['train_global_std'] = float(std)

    print(f'\nSaved: {save_path}')
    print(f'  x_data shape   = {x.shape}')
    print(f'  y_data shape   = {y.shape}')
    print(f'  z_data shape   = {z.shape}')
    print(f'  mask_data shape= {m.shape}')
    print(f'  angle example  = {z.min():.4f}° ~ {z.max():.4f}°')


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
            print(f'  -> Extracted sequences: {len(x_all)}')
        except Exception as e:
            print(f'Error processing {file_name}: {e}')

    x = np.asarray(dataset['x'], dtype=np.float32)
    y = np.asarray(dataset['y'], dtype=np.int64)
    z = np.asarray(dataset['z'], dtype=np.float32)
    m = np.asarray(dataset['mask'], dtype=np.float32)

    x_train, y_train, z_train, m_train, x_test, y_test, z_test, m_test = stratified_train_test_split(
        x, y, z, m, train_ratio
    )

    mean, std = fit_global_mean_std(x_train)
    print(f'\nTrain global mean = {mean:.8f}')
    print(f'Train global std  = {std:.8f}')

    x_train = apply_global_standardization(x_train, mean, std)
    x_test = apply_global_standardization(x_test, mean, std)

    save_h5(x_train, y_train, z_train, m_train, 'train_seq_data.h5', mean, std)
    save_h5(x_test, y_test, z_test, m_test, 'test_seq_data.h5', mean, std)

    stats_path = os.path.join(output_dir, 'train_global_mean_std.json')
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump({'train_global_mean': mean, 'train_global_std': std}, f, ensure_ascii=False, indent=2)
    print(f'\nSaved: {stats_path}')
    print('Done.')


if __name__ == '__main__':
    main()
