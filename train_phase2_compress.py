import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from udafd_compress128 import AttributeEncoder, HRRP_RecNet, RangeCompressor, UDAFDConfig, create_h5_dataloader, extract_attribute_feature

# ======== 在这里直接填写输入/输出参数 ========
TRAIN_H5 = 'h5_car_seq3/train_seq_data.h5'
PRETRAINED_CHECKPOINT = 'outputs3_compress/udafd_checkpoint.pt'
OUTPUT_DIR = 'outputs3_compress/phase2_only'  # 将重新训练的结果保存在新目录，防止覆盖原文件
NUM_CLASSES = 4
BATCH_SIZE = 128
EPOCHS_PHASE2 = 40
LR_2 = 1e-3
SEED = 42
NUM_WORKERS = 0
DEVICE = 'auto'      # 可选: 'auto', 'cpu', 'cuda'
# ==========================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main():
    set_seed(SEED)
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    if DEVICE == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(DEVICE)

    # 读取之前（全量训练过的）模型权重
    print(f"Loading checkpoint from: {PRETRAINED_CHECKPOINT}")
    ckpt = torch.load(PRETRAINED_CHECKPOINT, map_location=device)
    cfg = UDAFDConfig(**ckpt['cfg'])
    raw_input_dim = int(ckpt['input_dim'])
    
    # 1. 重新构建并在内存中加载 enc_A（即一阶段训练好的特征提取器）
    feature_input_dim = raw_input_dim
    range_compressor = None
    if getattr(cfg, 'use_range_compressor', False) and raw_input_dim > getattr(cfg, 'compressed_bins', 128):
        range_compressor = RangeCompressor(
            input_bins=raw_input_dim,
            output_bins=cfg.compressed_bins,
            hidden_channels=cfg.compressor_hidden_channels,
            kernel_size=cfg.compressor_kernel_size,
            use_skip=cfg.compressor_use_skip,
        )
        feature_input_dim = cfg.compressed_bins

    enc_A = AttributeEncoder(input_dim=feature_input_dim, cfg=cfg).to(device)
    if range_compressor is not None:
        enc_A.range_compressor = range_compressor.to(device)

    # 放心加载一阶段参数。由于原代码在二阶段时冻结了 enc_A，这部分参数绝对是原汁原味未受二阶段污染的
    enc_A.load_state_dict(ckpt['enc_A'])

    # 冻结第一阶段由于 enc_A 的参数，以供二阶段直接提取特征
    if range_compressor is not None:
        for p in enc_A.range_compressor.parameters():
            p.requires_grad = False
    for p in enc_A.parameters():
        p.requires_grad = False
    
    enc_A.eval()

    # 2. 从头开始初始化一个新的分类网络 (Phase 2 模型)
    rec_net = HRRP_RecNet(num_classes=NUM_CLASSES, cfg=cfg).to(device)
    optimizer_RecNet = optim.Adam(rec_net.parameters(), lr=LR_2)
    criterion = nn.CrossEntropyLoss()

    dataset, dataloader = create_h5_dataloader(
        h5_path=TRAIN_H5,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
    )

    print('Starting Phase 2 training only...')
    
    # 3. 运行Phase 2 训练
    for epoch in range(EPOCHS_PHASE2):
        rec_net.train()
        running_ce = 0.0

        for batch in dataloader:
            if len(batch) == 3:
                batch_X, batch_Y, _ = batch
            else:
                batch_X, batch_Y = batch[:2]

            batch_X = batch_X.to(device)
            batch_Y = batch_Y.to(device)

            optimizer_RecNet.zero_grad()
            
            # 提取一阶段特征 (不需要计算梯度)
            with torch.no_grad():
                F_A_feat = extract_attribute_feature(
                    enc_A,
                    batch_X,
                    cfg=cfg,
                    crop_mode='center',
                    range_compressor=enc_A.range_compressor if hasattr(enc_A, 'range_compressor') else None,
                )

            # 更新二阶段的分类器
            logits = rec_net(F_A_feat)
            loss_ce = criterion(logits, batch_Y)
            loss_ce.backward()
            optimizer_RecNet.step()
            running_ce += float(loss_ce.detach().cpu())

        num_batches = max(1, len(dataloader))
        print(f'[Phase2][{epoch+1:03d}/{EPOCHS_PHASE2}] ce={running_ce/num_batches:.4f}')

    # 4. 保存新的 checkpoint (复用原有的一阶段参数，仅替换成新训练好的 rec_net)
    new_ckpt = ckpt.copy()
    new_ckpt['rec_net'] = rec_net.state_dict()
    
    ckpt_path = output_dir / 'udafd_checkpoint.pt'
    torch.save(new_ckpt, ckpt_path)

    summary = {
        'checkpoint': str(ckpt_path),
        'train_h5': TRAIN_H5,
        'num_classes': NUM_CLASSES,
        'input_dim': raw_input_dim,
        'num_samples': int(len(dataset)),
        'cfg': cfg.__dict__,
        'seed': SEED,
        'device': str(device),
        'phase2_only': True
    }
    summary_path = output_dir / 'train_summary.json'
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')

    print('Phase 2 training finished.')
    print(f'New checkpoint saved to: {ckpt_path}')

if __name__ == '__main__':
    main()
