import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from udafd_pytorch_flexible import AttributeEncoder, HRRP_RecNet, UDAFDConfig, create_h5_dataloader, extract_attribute_feature

# ======== 在这里直接填写输入/输出参数 ========
TEST_H5 = 'h5_car_seq/seq_data.h5'
CHECKPOINT = 'outputs/train_run/udafd_checkpoint.pt'
OUTPUT_JSON = 'outputs/test_results.json'  # 不想保存可设为 ''
BATCH_SIZE = 256
NUM_WORKERS = 0
DEVICE = 'auto'          # 可选: 'auto', 'cpu', 'cuda'
CROP_MODE = 'center'     # 可选: 'center', 'random_avg'
NUM_RANDOM_CROPS = 5
# ==========================================


@torch.no_grad()
def evaluate(enc_A, rec_net, dataloader, cfg, device, crop_mode='center', num_random_crops=1):
    y_true_all = []
    y_pred_all = []

    for batch in dataloader:
        if len(batch) == 3:
            batch_X, batch_Y, _ = batch
        else:
            batch_X, batch_Y = batch[:2]

        batch_X = batch_X.to(device)
        batch_Y = batch_Y.to(device)

        F_A_feat = extract_attribute_feature(
            enc_A,
            batch_X,
            cfg=cfg,
            crop_mode=crop_mode,
            num_random_crops=num_random_crops,
        )
        logits = rec_net(F_A_feat)
        preds = logits.argmax(dim=1)

        y_true_all.append(batch_Y.cpu().numpy())
        y_pred_all.append(preds.cpu().numpy())

    y_true = np.concatenate(y_true_all, axis=0)
    y_pred = np.concatenate(y_pred_all, axis=0)
    acc = float(accuracy_score(y_true, y_pred))
    cm = confusion_matrix(y_true, y_pred).tolist()
    report = classification_report(y_true, y_pred, digits=4, output_dict=True)
    return {
        'accuracy': acc,
        'confusion_matrix': cm,
        'classification_report': report,
        'num_samples': int(len(y_true)),
    }


def main():
    if DEVICE == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(DEVICE)

    ckpt = torch.load(CHECKPOINT, map_location=device)
    cfg = UDAFDConfig(**ckpt['cfg'])
    input_dim = int(ckpt['input_dim'])
    num_classes = int(ckpt['num_classes'])

    dataset, dataloader = create_h5_dataloader(
        h5_path=TEST_H5,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    if dataset.num_bins != input_dim:
        raise ValueError(f'Checkpoint input_dim={input_dim}, but test h5 has M={dataset.num_bins}.')

    enc_A = AttributeEncoder(input_dim=input_dim, cfg=cfg).to(device)
    rec_net = HRRP_RecNet(num_classes=num_classes, cfg=cfg).to(device)
    enc_A.load_state_dict(ckpt['enc_A'])
    rec_net.load_state_dict(ckpt['rec_net'])
    enc_A.eval()
    rec_net.eval()

    results = evaluate(
        enc_A=enc_A,
        rec_net=rec_net,
        dataloader=dataloader,
        cfg=cfg,
        device=device,
        crop_mode=CROP_MODE,
        num_random_crops=NUM_RANDOM_CROPS,
    )
    results['checkpoint'] = CHECKPOINT
    results['test_h5'] = TEST_H5
    results['num_classes'] = num_classes
    results['input_dim'] = input_dim
    results['cfg'] = cfg.__dict__

    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"Num samples: {results['num_samples']}")

    if OUTPUT_JSON:
        output_path = Path(OUTPUT_JSON)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')
        print(f'Saved results to: {output_path}')


if __name__ == '__main__':
    main()
