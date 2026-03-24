import os
from collections import Counter

import h5py
import numpy as np

# ======== 在这里直接填写输入参数 ========
H5_PATH = os.path.join('h5_car_seq', 'train_seq_data.h5')
# =====================================


def inspect_h5(h5_path: str):
    with h5py.File(h5_path, 'r') as f:
        print('=== Datasets ===')
        for k in f.keys():
            print(f'{k}: shape={f[k].shape}, dtype={f[k].dtype}')

        required = ['x_data', 'y_data']
        missing = [k for k in required if k not in f]
        if missing:
            print(f'\n[ERROR] Missing required datasets: {missing}')
            return

        x = f['x_data'][:]
        y = f['y_data'][:]
        z = f['z_data'][:] if 'z_data' in f else None
        m = f['mask_data'][:] if 'mask_data' in f else None

    print('\n=== Basic checks ===')
    print(f'x ndim: {x.ndim}')
    if x.ndim == 3:
        print(f'x shape = [num_seq, T, M] = {x.shape}')
    elif x.ndim == 2:
        print(f'x shape = [num_samples, M] = {x.shape}')
        print('[WARN] This looks like single-frame HRRP, not sequence HRRP expected by UDAFD.')
        return
    else:
        print('[ERROR] Unexpected x_data ndim.')
        return

    if len(y) != len(x):
        print('[ERROR] y_data length does not match x_data.')
    else:
        print('[OK] y_data length matches x_data.')

    if z is not None:
        print(f'z shape = {z.shape}')
        if z.ndim != 2 or z.shape[0] != x.shape[0] or z.shape[1] != x.shape[1]:
            print('[WARN] For sequence HRRP, z_data is recommended to be [num_seq, T].')

    if m is not None:
        print(f'mask shape = {m.shape}')
        if m.shape != x.shape:
            print('[WARN] mask_data shape should match x_data exactly.')
        else:
            print('[OK] mask_data shape matches x_data.')

    print('\n=== Statistics ===')
    print(f'x min / max / mean / std = {x.min():.6f} / {x.max():.6f} / {x.mean():.6f} / {x.std():.6f}')
    print(f'finite ratio = {np.isfinite(x).mean():.6f}')

    counter = Counter(y.tolist())
    print('\n=== Label counts ===')
    for k in sorted(counter.keys()):
        print(f'class {k}: {counter[k]}')

    norms = np.linalg.norm(x.reshape(x.shape[0], -1), axis=1)
    print('\n=== Sequence norms ===')
    print(f'norm min / max / mean = {norms.min():.6f} / {norms.max():.6f} / {norms.mean():.6f}')

    if z is not None and z.ndim == 2 and z.shape[1] == x.shape[1]:
        delta_angles = np.diff(z, axis=1)
        print('\n=== Angle step (from z_data) ===')
        print(f'mean step = {delta_angles.mean():.6f}')
        print(f'min  step = {delta_angles.min():.6f}')
        print(f'max  step = {delta_angles.max():.6f}')

    print('\nInspection complete.')


def main():
    inspect_h5(H5_PATH)


if __name__ == '__main__':
    main()
