import json
import random
from pathlib import Path

import numpy as np
import torch

from udafd_pytorch_flexible import UDAFDConfig, create_h5_dataloader, train_framework

# ======== 在这里直接填写输入/输出参数 ========
TRAIN_H5 = 'h5_car_seq/seq_data.h5'
NUM_CLASSES = 10
OUTPUT_DIR = 'outputs/train_run'
BATCH_SIZE = 128
EPOCHS_PHASE1 = 70
EPOCHS_PHASE2 = 40
LR_1 = 1e-3
LR_2 = 1e-3
T = 30
N = 10
L = 9
D_D = 8
D_A = 8
BETA = 0.1
LAMBDA_REG = 0.5
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

    cfg = UDAFDConfig(
        T=T,
        N=N,
        l=L,
        d_D=D_D,
        d_A=D_A,
        beta=BETA,
        lambda_reg=LAMBDA_REG,
        lr_1=LR_1,
        lr_2=LR_2,
        batch_size=BATCH_SIZE,
    )

    dataset, dataloader = create_h5_dataloader(
        h5_path=TRAIN_H5,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
    )

    enc_D, enc_A, decoder, rec_net = train_framework(
        dataloader=dataloader,
        cfg=cfg,
        input_dim=dataset.num_bins,
        num_classes=NUM_CLASSES,
        epochs_phase1=EPOCHS_PHASE1,
        epochs_phase2=EPOCHS_PHASE2,
        device=device,
    )

    ckpt = {
        'enc_D': enc_D.state_dict(),
        'enc_A': enc_A.state_dict(),
        'decoder': decoder.state_dict(),
        'rec_net': rec_net.state_dict(),
        'cfg': cfg.__dict__,
        'num_classes': NUM_CLASSES,
        'input_dim': int(dataset.num_bins),
        'train_h5': TRAIN_H5,
        'seed': SEED,
    }

    ckpt_path = output_dir / 'udafd_checkpoint.pt'
    torch.save(ckpt, ckpt_path)

    summary = {
        'checkpoint': str(ckpt_path),
        'train_h5': TRAIN_H5,
        'num_classes': NUM_CLASSES,
        'input_dim': int(dataset.num_bins),
        'num_samples': int(len(dataset)),
        'cfg': cfg.__dict__,
        'seed': SEED,
        'device': str(device),
    }
    summary_path = output_dir / 'train_summary.json'
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')

    print('Training finished.')
    print(f'Checkpoint saved to: {ckpt_path}')
    print(f'Summary saved to:    {summary_path}')


if __name__ == '__main__':
    main()
