# UDAFD 一套可直接运行的脚本

## 文件
- `udafd_pytorch_flexible.py`：模型、数据集与训练框架
- `train_udafd.py`：训练脚本
- `test_udafd.py`：测试脚本
- `inspect_h5.py`：检查 h5 数据格式

## 1) 先检查 h5
```bash
python inspect_h5.py train_seq_data.h5
python inspect_h5.py test_seq_data.h5
```

期望至少包含：
- `x_data`: `[num_seq, T, M]`
- `y_data`: `[num_seq]`

建议还包含：
- `z_data`: `[num_seq, T]`
- `mask_data`: `[num_seq, T, M]`

## 2) 训练
```bash
python train_udafd.py \
  --train_h5 train_seq_data.h5 \
  --num_classes 10 \
  --output_dir runs/car10_case1 \
  --batch_size 128 \
  --epochs_phase1 70 \
  --epochs_phase2 40
```

输出：
- `runs/car10_case1/udafd_checkpoint.pt`
- `runs/car10_case1/train_summary.json`

## 3) 测试
```bash
python test_udafd.py \
  --test_h5 test_seq_data.h5 \
  --checkpoint runs/car10_case1/udafd_checkpoint.pt \
  --output_json runs/car10_case1/test_result.json
```

## 4) 关于距离单元数 M
`M` 不再写死为 101。只要同一个 h5 文件里的所有样本 `M` 一致，就可以直接训练和测试。

## 5) 说明
- 论文默认参数仍建议用：`T=30, N=10, l=9, d_D=d_A=8`
- 若想尽量贴论文仿真设置，建议先把方位步长下采样到接近 3°
