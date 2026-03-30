import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Union

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from typing import Optional

@dataclass
class UDAFDConfig:
    T: int = 30
    N: int = 10
    l: int = 9
    d_D: int = 8
    d_A: int = 8
    beta: float = 0.1
    lambda_reg: float = 0.5
    lr_1: float = 1e-3
    lr_2: float = 1e-3
    batch_size: int = 128
    dynamic_hidden_dim: int = 64
    attribute_hidden_dim: int = 64
    decoder_c1: int = 128
    decoder_c2: int = 128

    # ---------- Front-end range compression ----------
    # If the raw HRRP range-cell dimension M is larger than compressed_bins,
    # a learnable 1D-CNN front-end is inserted before both encoders.
    use_sparse_attention: bool = True
    compressed_bins: int = 128
    attention_hidden_channels: int = 32

    # ---------- Reconstruction target ----------
    # 'original'   : encoders see compressed HRRP, decoder reconstructs raw HRRP [T, M_raw].
    # 'compressed' : encoders see compressed HRRP, decoder reconstructs compressed HRRP [T, M_comp].
    # For 512 -> 128 compression, 'original' is safer if you still want the generative objective
    # to be anchored in the original observation space.
    reconstruction_mode: str = 'original'

    @property
    def delta(self) -> int:
        if self.T % self.N != 0:
            raise ValueError(f'T must be divisible by N, got T={self.T}, N={self.N}.')
        return self.T // self.N


class HRRPSequenceH5Dataset(Dataset):
    """
    H5 format expected by this class:
      x_data:    [num_samples, T, M]
      y_data:    [num_samples]
      z_data:    [num_samples, T]              optional metadata
      mask_data: [num_samples, T, M]           optional
    """

    def __init__(
        self,
        h5_path: Union[str, Path],
        x_key: str = 'x_data',
        y_key: str = 'y_data',
        mask_key: str = 'mask_data',
        load_into_memory: bool = True,
    ):
        self.h5_path = str(h5_path)
        self.x_key = x_key
        self.y_key = y_key
        self.mask_key = mask_key
        self.load_into_memory = load_into_memory

        with h5py.File(self.h5_path, 'r') as f:
            self.shape = tuple(f[self.x_key].shape)
            self.num_samples, self.seq_len, self.num_bins = self.shape
            if self.y_key not in f:
                raise KeyError(f'Missing dataset key: {self.y_key}')
            self.has_mask = self.mask_key in f
            self.num_classes = int(np.max(f[self.y_key][:])) + 1

            if self.load_into_memory:
                self.x_data = f[self.x_key][:].astype(np.float32)
                self.y_data = f[self.y_key][:].astype(np.int64)
                self.mask_data = f[self.mask_key][:].astype(np.float32) if self.has_mask else None
            else:
                self.x_data = None
                self.y_data = None
                self.mask_data = None

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int):
        if self.load_into_memory:
            x = torch.from_numpy(self.x_data[idx])
            y = torch.tensor(self.y_data[idx], dtype=torch.long)
            if self.has_mask:
                m = torch.from_numpy(self.mask_data[idx])
                return x, y, m
            return x, y

        with h5py.File(self.h5_path, 'r') as f:
            x = torch.tensor(f[self.x_key][idx], dtype=torch.float32)
            y = torch.tensor(f[self.y_key][idx], dtype=torch.long)
            if self.has_mask:
                m = torch.tensor(f[self.mask_key][idx], dtype=torch.float32)
                return x, y, m
            return x, y



def create_h5_dataloader(
    h5_path: Union[str, Path],
    batch_size: int = 128,
    shuffle: bool = True,
    num_workers: int = 0,
    load_into_memory: bool = True,
) -> Tuple[HRRPSequenceH5Dataset, DataLoader]:
    dataset = HRRPSequenceH5Dataset(h5_path, load_into_memory=load_into_memory)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return dataset, loader



def split_windows(X: torch.Tensor, cfg: UDAFDConfig) -> torch.Tensor:
    """
    X: [B, T, M]
    return: [B, N, delta, M]
    """
    B, T_, M_ = X.shape
    if T_ != cfg.T:
        raise ValueError(f'Expected sequence length T={cfg.T}, got {T_}.')
    return X.view(B, cfg.N, cfg.delta, M_)



def crop_subsequence(
    X: torch.Tensor,
    crop_len: int,
    starts: Optional[torch.Tensor] = None,
    random_crop: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    X: [B, T, M]
    return: cropped [B, crop_len, M], starts [B]
    """
    B, T_, _ = X.shape
    if crop_len > T_:
        raise ValueError(f'crop_len={crop_len} cannot exceed T={T_}.')

    if starts is None:
        if random_crop:
            starts = torch.randint(0, T_ - crop_len + 1, (B,), device=X.device)
        else:
            start = (T_ - crop_len) // 2
            starts = torch.full((B,), start, device=X.device, dtype=torch.long)

    chunks = []
    for i, s in enumerate(starts.tolist()):
        chunks.append(X[i : i + 1, s : s + crop_len, :])
    return torch.cat(chunks, dim=0), starts


from SparseAttention import SparseAttentionCompressor


def maybe_compress_sequence(
    X: torch.Tensor,
    range_compressor: Optional[SparseAttentionCompressor] = None,
) -> torch.Tensor:
    if range_compressor is None:
        return X
    return range_compressor(X)



def compress_mask_sequence(mask: torch.Tensor, output_bins: int) -> torch.Tensor:
    """
    Compress mask on the range dimension with average pooling.
    The result remains in [0, 1] and can be interpreted as confidence/coverage
    inside each compressed range interval.
    """
    if mask.size(-1) == output_bins:
        return mask
    B, T_, M_ = mask.shape
    pooled = F.adaptive_avg_pool1d(mask.reshape(B * T_, 1, M_), output_bins)
    return pooled.view(B, T_, output_bins).clamp(0.0, 1.0)



def adapt_sequence_for_attribute_encoder(
    X: torch.Tensor,
    attribute_encoder: 'AttributeEncoder',
    range_compressor: Optional[SparseAttentionCompressor] = None,
) -> torch.Tensor:
    if X.size(-1) == attribute_encoder.input_dim:
        return X
    if range_compressor is None:
        raise ValueError(
            'Decoder output dimension does not match AttributeEncoder input_dim, '
            'but no range_compressor was provided to adapt the sequence.'
        )
    X_feat = range_compressor(X)
    if X_feat.size(-1) != attribute_encoder.input_dim:
        raise ValueError(
            f'Compressed sequence dim={X_feat.size(-1)} does not match '
            f'AttributeEncoder input_dim={attribute_encoder.input_dim}.'
        )
    return X_feat



def build_gp_covariance(
    N_: int,
    d_D_: int,
    epsilon: float = 1e-5,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Multi-scale multi-kernel GP prior covariance.
    return: [d_D, N, N]
    """
    S = d_D_ // 2
    n = torch.arange(N_, dtype=torch.float32, device=device)
    dist_sq = (n[:, None] - n[None, :]).pow(2)

    kernels = []
    for s in range(S):
        lam = 2.0 / (2 ** s)
        k_rbf = torch.exp(-dist_sq / (2.0 * lam * lam))
        k_cauchy = 1.0 / (1.0 + dist_sq / (lam * lam))
        kernels.extend([k_rbf, k_cauchy])

    Sigma_G = torch.stack(kernels, dim=0)
    Sigma_G = Sigma_G + epsilon * torch.eye(N_, device=device).unsqueeze(0)
    return Sigma_G


class DynamicEncoder(nn.Module):
    """
    Input:  [B, N, delta, M_feat]
    Output: mu_D [B, N, d_D], gamma_D [B, N, d_D, 2]
    """

    def __init__(self, input_dim: int, cfg: UDAFDConfig):
        super().__init__()
        self.input_dim = input_dim
        self.n_windows = cfg.N
        self.delta = cfg.delta
        self.d_D = cfg.d_D

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=cfg.dynamic_hidden_dim,
            batch_first=True,
        )
        self.fc1 = nn.Linear(cfg.dynamic_hidden_dim, cfg.dynamic_hidden_dim)
        self.fc2 = nn.Linear(cfg.dynamic_hidden_dim, cfg.dynamic_hidden_dim)
        self.fc_mu = nn.Linear(cfg.dynamic_hidden_dim, cfg.d_D)
        self.fc_gamma = nn.Linear(cfg.dynamic_hidden_dim, 2 * cfg.d_D)

    def forward(
        self,
        x_windows: torch.Tensor,
        m_windows: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, N_, delta_, M_ = x_windows.shape
        if N_ != self.n_windows or delta_ != self.delta or M_ != self.input_dim:
            raise ValueError(
                f'Expected [B,{self.n_windows},{self.delta},{self.input_dim}], '
                f'got {tuple(x_windows.shape)}.'
            )

        if m_windows is None:
            m_windows = torch.ones_like(x_windows)

        x = (x_windows * m_windows).reshape(B * N_, delta_, M_)
        s, _ = self.lstm(x)
        s = torch.sigmoid(s[:, -1, :])

        r = F.relu(self.fc2(F.relu(self.fc1(s))))
        mu_D = self.fc_mu(r).view(B, N_, self.d_D)
        gamma_D = F.softplus(self.fc_gamma(r)).view(B, N_, self.d_D, 2) + 1e-4
        return mu_D, gamma_D


class AttributeEncoder(nn.Module):
    """
    Input: [B, l, M_feat]
    Output: mu_A [B, d_A]
    Posterior q(F_A|X) = N(mu_A, I)
    """

    def __init__(self, input_dim: int, cfg: UDAFDConfig):
        super().__init__()
        self.input_dim = input_dim
        self.d_A = cfg.d_A

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=cfg.attribute_hidden_dim,
            batch_first=True,
        )
        self.fc1 = nn.Linear(cfg.attribute_hidden_dim, cfg.attribute_hidden_dim)
        self.fc2 = nn.Linear(cfg.attribute_hidden_dim, cfg.attribute_hidden_dim)
        self.fc_mu = nn.Linear(cfg.attribute_hidden_dim, cfg.d_A)

    def forward(
        self,
        x_subseq: torch.Tensor,
        m_subseq: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if m_subseq is None:
            m_subseq = torch.ones_like(x_subseq)

        if x_subseq.size(-1) != self.input_dim:
            raise ValueError(f'Expected last dim={self.input_dim}, got {x_subseq.size(-1)}.')

        x = x_subseq * m_subseq
        s, _ = self.lstm(x)
        s = torch.sigmoid(s[:, -1, :])

        r = F.relu(self.fc2(F.relu(self.fc1(s))))
        mu_A = self.fc_mu(r)
        return mu_A



def build_precision_cholesky(
    gamma_D: torch.Tensor,
    epsilon: float = 1e-4,
) -> torch.Tensor:
    """
    gamma_D: [B, N, d_D, 2]
    return U: [B, d_D, N, N], upper-triangular band matrix
    """
    B, N_, d_D_, two = gamma_D.shape
    if two != 2:
        raise ValueError(f'Expected gamma_D last dim = 2, got {two}.')

    gamma = gamma_D.permute(0, 2, 1, 3)  # [B, d_D, N, 2]

    U = torch.zeros(B, d_D_, N_, N_, device=gamma.device, dtype=gamma.dtype)
    idx = torch.arange(N_, device=gamma.device)

    U[..., idx, idx] = gamma[..., 0] + epsilon
    if N_ > 1:
        U[..., idx[:-1], idx[1:]] = gamma[..., :-1, 1]

    return U



def sample_dynamic_features(
    mu_D: torch.Tensor,
    gamma_D: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    mu_D: [B, N, d_D]
    gamma_D: [B, N, d_D, 2]

    returns:
        F_D:   [B, N, d_D]
        cov_q: [B, d_D, N, N]
        U:     [B, d_D, N, N]
    """
    B, N_, d_D_ = mu_D.shape
    U = build_precision_cholesky(gamma_D)
    
    # 1. 极其稳定的三角矩阵求逆 (替代 torch.linalg.inv)
    # 计算 U_inv = U^{-1}
    I = torch.eye(N_, device=U.device, dtype=U.dtype).expand_as(U)
    U_inv = torch.linalg.solve_triangular(U, I, upper=True)
    
    # 2. 计算协方差矩阵用于 KL 散度计算 (V @ V^T 物理上绝对保证半正定)
    cov_q = U_inv @ U_inv.transpose(-1, -2)
    
    # 3. 直接使用 U_inv 进行重参数化采样，彻底避开 cholesky！
    eps = torch.randn(B, d_D_, N_, 1, device=mu_D.device, dtype=mu_D.dtype)
    
    mu = mu_D.permute(0, 2, 1).unsqueeze(-1)  # [B, d_D, N, 1]
    z = mu + U_inv @ eps
    F_D = z.squeeze(-1).permute(0, 2, 1)      # [B, N, d_D]
    
    return F_D, cov_q, U



def compute_kl_dynamic(
    mu_D: torch.Tensor,
    cov_q: torch.Tensor,
    U: torch.Tensor,
    Sigma_G: torch.Tensor,
) -> torch.Tensor:
    """
    KL(q(F_D|X) || p(F_D)) with a GP prior over the N windows.
    """
    B, N_, d_D_ = mu_D.shape
    mu = mu_D.permute(0, 2, 1)  # [B, d_D, N]

    logdet_q = -2.0 * torch.log(torch.diagonal(U, dim1=-2, dim2=-1)).sum(dim=-1)  # [B, d_D]

    kl_all = []
    for d in range(d_D_):
        Sigma_p = Sigma_G[d]  # [N, N]
        L_p = torch.linalg.cholesky(Sigma_p)
        Sigma_p_inv = torch.cholesky_inverse(L_p)
        logdet_p = 2.0 * torch.log(torch.diagonal(L_p)).sum()

        trace_term = torch.einsum('ij,bji->b', Sigma_p_inv, cov_q[:, d])
        maha_term = torch.einsum('bi,ij,bj->b', mu[:, d], Sigma_p_inv, mu[:, d])

        kl_d = 0.5 * (trace_term + maha_term - N_ + logdet_p - logdet_q[:, d])
        kl_all.append(kl_d)

    return torch.stack(kl_all, dim=1).sum(dim=1).mean()



def compute_kl_attribute(mu_A: torch.Tensor) -> torch.Tensor:
    return 0.5 * mu_A.pow(2).sum(dim=1).mean()


class Decoder(nn.Module):
    """
    Output: mu_X, sigma_X in R^{T x M_recon}
    M_recon can be either the raw range-cell dimension M_raw or the compressed
    dimension M_comp, depending on cfg.reconstruction_mode.
    """

    def __init__(self, output_dim: int, cfg: UDAFDConfig):
        super().__init__()
        self.output_dim = output_dim
        self.seq_len = cfg.T
        self.n_windows = cfg.N
        self.delta = cfg.delta

        self.fc1 = nn.Linear(cfg.d_D + cfg.d_A, cfg.decoder_c1)
        self.bn = nn.BatchNorm1d(cfg.decoder_c1)
        self.lstm = nn.LSTM(input_size=cfg.decoder_c1, hidden_size=cfg.decoder_c1, batch_first=True)
        self.fc2 = nn.Linear(cfg.decoder_c1, cfg.decoder_c2)
        self.fc_mu_x = nn.Linear(cfg.decoder_c2, output_dim)
        self.fc_sigma_x = nn.Linear(cfg.decoder_c2, output_dim)

    def forward(self, F_D: torch.Tensor, F_A_rep: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = torch.cat([F_D, F_A_rep], dim=-1)     # [B, N, d_D + d_A]
        h = self.fc1(h)                           # [B, N, c1]
        h = self.bn(h.transpose(1, 2)).transpose(1, 2)
        h = torch.tanh(h)

        if h.size(1) != self.seq_len:
            h = h.repeat_interleave(self.delta, dim=1)
            if h.size(1) > self.seq_len:
                h = h[:, : self.seq_len, :]
            elif h.size(1) < self.seq_len:
                pad = h[:, -1:, :].expand(-1, self.seq_len - h.size(1), -1)
                h = torch.cat([h, pad], dim=1)

        L_out, _ = self.lstm(h)
        h2 = F.relu(self.fc2(L_out))
        mu_x = self.fc_mu_x(h2)
        sigma_x = torch.sigmoid(self.fc_sigma_x(h2)) + 1e-4
        return mu_x, sigma_x



def gaussian_nll(X_true: torch.Tensor, mu_x: torch.Tensor, sigma_x: torch.Tensor) -> torch.Tensor:
    sigma_x = sigma_x.clamp_min(1e-4)
    nll = 0.5 * (
        ((X_true - mu_x) / sigma_x).pow(2)
        + 2.0 * torch.log(sigma_x)
        + math.log(2.0 * math.pi)
    )
    return nll.sum(dim=(1, 2)).mean()



def compute_elbo_loss(
    X_true: torch.Tensor,
    mu_x: torch.Tensor,
    sigma_x: torch.Tensor,
    mu_D: torch.Tensor,
    cov_q_D: torch.Tensor,
    U_D: torch.Tensor,
    mu_A: torch.Tensor,
    Sigma_G: torch.Tensor,
    cfg: UDAFDConfig,
    current_beta: Optional[float] = None,
):
    beta_val = current_beta if current_beta is not None else cfg.beta
    nll = gaussian_nll(X_true, mu_x, sigma_x)
    kl_D = compute_kl_dynamic(mu_D, cov_q_D, U_D, Sigma_G)
    kl_A = compute_kl_attribute(mu_A)
    beta_A = beta_val * 0.1
    elbo = nll + beta_val * kl_D + beta_A * kl_A
    stats = {
        'nll': float(nll.detach().cpu()),
        'kl_D': float(kl_D.detach().cpu()),
        'kl_A': float(kl_A.detach().cpu()),
        'elbo': float(elbo.detach().cpu()),
    }
    return elbo, stats



def compute_counterfactual_reg(
    attribute_encoder: AttributeEncoder,
    decoder: Decoder,
    F_D: torch.Tensor,
    F_A_original: torch.Tensor,
    cfg: UDAFDConfig,
    range_compressor: Optional[SparseAttentionCompressor] = None,
) -> torch.Tensor:
    B = F_D.size(0)
    device = F_D.device
    dtype = F_D.dtype

    F_A_cf = torch.randn(B, cfg.d_A, device=device, dtype=dtype)
    F_A_cf_rep = F_A_cf.unsqueeze(1).expand(-1, cfg.N, -1)

    mu_x_cf, _ = decoder(F_D, F_A_cf_rep)
    mu_x_cf_for_attr = adapt_sequence_for_attribute_encoder(
        mu_x_cf,
        attribute_encoder=attribute_encoder,
        range_compressor=range_compressor,
    )
    X_cf_crop, _ = crop_subsequence(mu_x_cf_for_attr, crop_len=cfg.l, random_crop=True)

    mu_A_cf_post = attribute_encoder(X_cf_crop)
    log_q_orig = -0.5 * (F_A_original - mu_A_cf_post).pow(2).sum(dim=1)
    log_q_cf = -0.5 * (F_A_cf - mu_A_cf_post).pow(2).sum(dim=1)

    log_ratio = (log_q_orig - log_q_cf).clamp(min=-30.0, max=30.0)
    ratio = torch.exp(log_ratio)
    return ratio.mean()


class HRRP_RecNet(nn.Module):
    def __init__(self, num_classes: int, cfg: UDAFDConfig):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(cfg.d_A, 32),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(16, num_classes),
        )

    def forward(self, F_A_feat: torch.Tensor) -> torch.Tensor:
        return self.classifier(F_A_feat)


@torch.no_grad()
def extract_attribute_feature(
    enc_A: AttributeEncoder,
    X: torch.Tensor,
    cfg: UDAFDConfig,
    crop_mode: str = 'center',
    num_random_crops: int = 1,
    range_compressor: Optional[SparseAttentionCompressor] = None,
) -> torch.Tensor:
    if range_compressor is None:
        range_compressor = getattr(enc_A, 'range_compressor', None)

    X_feat = maybe_compress_sequence(X, range_compressor)

    if crop_mode == 'center':
        X_sub, _ = crop_subsequence(X_feat, crop_len=cfg.l, random_crop=False)
        return enc_A(X_sub)

    if crop_mode == 'random_avg':
        feats = []
        for _ in range(num_random_crops):
            X_sub, _ = crop_subsequence(X_feat, crop_len=cfg.l, random_crop=True)
            feats.append(enc_A(X_sub))
        return torch.stack(feats, dim=0).mean(dim=0)

    raise ValueError(f'Unknown crop_mode: {crop_mode}')



def infer_input_dim_and_num_classes(dataloader: DataLoader) -> Tuple[int, int]:
    batch = next(iter(dataloader))
    X = batch[0]
    input_dim = int(X.shape[-1])

    dataset = getattr(dataloader, 'dataset', None)
    num_classes = getattr(dataset, 'num_classes', None)
    if num_classes is None:
        Y = batch[1]
        num_classes = int(torch.max(Y).item()) + 1
    return input_dim, int(num_classes)



def _resolve_frontend_and_decoder_dims(
    raw_input_dim: int,
    cfg: UDAFDConfig,
) -> Tuple[Optional[SparseAttentionCompressor], int, int]:
    range_compressor: Optional[SparseAttentionCompressor] = None
    feature_input_dim = raw_input_dim

    if cfg.use_sparse_attention and raw_input_dim > cfg.compressed_bins:
        range_compressor = SparseAttentionCompressor(
            input_bins=raw_input_dim,
            output_bins=cfg.compressed_bins,
            hidden_channels=cfg.attention_hidden_channels,
        )
        feature_input_dim = cfg.compressed_bins

    if cfg.reconstruction_mode == 'original':
        decoder_output_dim = raw_input_dim
    elif cfg.reconstruction_mode == 'compressed':
        decoder_output_dim = feature_input_dim
    else:
        raise ValueError(
            f"cfg.reconstruction_mode must be 'original' or 'compressed', got {cfg.reconstruction_mode!r}."
        )

    return range_compressor, feature_input_dim, decoder_output_dim



def train_framework(
    dataloader: DataLoader,
    cfg: UDAFDConfig,
    input_dim: Optional[int] = None,
    num_classes: Optional[int] = None,
    epochs_phase1: int = 70,
    epochs_phase2: int = 40,
    device: Optional[torch.device] = None,
):
    """
    Expected dataloader outputs:
      either (X, Y)
      or     (X, Y, X_mask)

    X:      [B, T, M_raw]
    Y:      [B]
    X_mask: [B, T, M_raw], optional

    When cfg.use_sparse_attention=True and M_raw > cfg.compressed_bins,
    the encoders work on [B, T, M_comp] while the decoder reconstructs either:
      - raw [B, T, M_raw]     if cfg.reconstruction_mode == 'original'
      - compressed [B, T, M_comp] if cfg.reconstruction_mode == 'compressed'
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    inferred_input_dim, inferred_num_classes = infer_input_dim_and_num_classes(dataloader)
    if input_dim is None:
        input_dim = inferred_input_dim
    if num_classes is None:
        num_classes = inferred_num_classes

    range_compressor, feature_input_dim, decoder_output_dim = _resolve_frontend_and_decoder_dims(
        raw_input_dim=input_dim,
        cfg=cfg,
    )

    if range_compressor is not None:
        range_compressor = range_compressor.to(device)

    enc_D = DynamicEncoder(input_dim=feature_input_dim, cfg=cfg).to(device)
    enc_A = AttributeEncoder(input_dim=feature_input_dim, cfg=cfg).to(device)
    decoder = Decoder(output_dim=decoder_output_dim, cfg=cfg).to(device)
    rec_net = HRRP_RecNet(num_classes=num_classes, cfg=cfg).to(device)

    optimizer_params = list(enc_D.parameters()) + list(enc_A.parameters()) + list(decoder.parameters())
    if range_compressor is not None:
        optimizer_params = list(range_compressor.parameters()) + optimizer_params

    optimizer_UDAFD = optim.Adam(optimizer_params, lr=cfg.lr_1)
    optimizer_RecNet = optim.Adam(rec_net.parameters(), lr=cfg.lr_2)

    Sigma_G = build_gp_covariance(N_=cfg.N, d_D_=cfg.d_D, device=device)

    warmup_epochs = 30

    print(
        f'[Init] raw_bins={input_dim} feature_bins={feature_input_dim} '
        f'recon_bins={decoder_output_dim} compressor={'on' if range_compressor is not None else 'off'} '
        f'recon_mode={cfg.reconstruction_mode}'
    )

    for epoch in range(epochs_phase1):
        if range_compressor is not None:
            range_compressor.train()
        enc_D.train()
        enc_A.train()
        decoder.train()

        running = {'total': 0.0, 'elbo': 0.0, 'reg': 0.0, 'nll': 0.0, 'kl_D': 0.0, 'kl_A': 0.0}
        current_beta = cfg.beta * min(1.0, epoch / warmup_epochs)

        for batch in dataloader:
            if len(batch) == 3:
                batch_X, _, batch_mask = batch
                batch_mask = batch_mask.to(device)
            else:
                batch_X, _ = batch[:2]
                batch_mask = torch.ones_like(batch_X)

            batch_X = batch_X.to(device)
            batch_mask = batch_mask.to(device)

            if batch_X.size(1) != cfg.T:
                raise ValueError(f'Expected T={cfg.T}, got {batch_X.size(1)}.')
            if batch_X.size(2) != input_dim:
                raise ValueError(f'Expected M={input_dim}, got {batch_X.size(2)}.')

            optimizer_UDAFD.zero_grad()

            batch_X_feat = maybe_compress_sequence(batch_X, range_compressor)
            batch_mask_feat = compress_mask_sequence(batch_mask, feature_input_dim)

            X_windows = split_windows(batch_X_feat, cfg)
            M_windows = split_windows(batch_mask_feat, cfg)

            X_subseq, starts = crop_subsequence(batch_X_feat, crop_len=cfg.l, random_crop=True)
            M_subseq, _ = crop_subsequence(batch_mask_feat, crop_len=cfg.l, starts=starts, random_crop=False)

            mu_D, gamma_D = enc_D(X_windows, M_windows)
            mu_A = enc_A(X_subseq, M_subseq)

            F_D, cov_q_D, U_D = sample_dynamic_features(mu_D, gamma_D)
            F_A = mu_A + torch.randn_like(mu_A)

            F_A_rep = F_A.unsqueeze(1).expand(-1, cfg.N, -1)
            mu_x, sigma_x = decoder(F_D, F_A_rep)

            if cfg.reconstruction_mode == 'original':
                X_target = batch_X
            else:
                # Keep the reconstruction target fixed with respect to the likelihood term.
                # Otherwise the learnable front-end could move the target itself and weaken the constraint.
                X_target = batch_X_feat.detach()

            loss_elbo, stats = compute_elbo_loss(
                X_true=X_target,
                mu_x=mu_x,
                sigma_x=sigma_x,
                mu_D=mu_D,
                cov_q_D=cov_q_D,
                U_D=U_D,
                mu_A=mu_A,
                Sigma_G=Sigma_G,
                cfg=cfg,
                current_beta=current_beta,
            )

            loss_reg = compute_counterfactual_reg(
                attribute_encoder=enc_A,
                decoder=decoder,
                F_D=F_D,
                F_A_original=F_A,
                cfg=cfg,
                range_compressor=range_compressor,
            )

            loss = loss_elbo + cfg.lambda_reg * loss_reg
            loss.backward()
            optimizer_UDAFD.step()

            running['total'] += float(loss.detach().cpu())
            running['elbo'] += float(loss_elbo.detach().cpu())
            running['reg'] += float(loss_reg.detach().cpu())
            running['nll'] += stats['nll']
            running['kl_D'] += stats['kl_D']
            running['kl_A'] += stats['kl_A']

        num_batches = max(1, len(dataloader))
        print(
            f'[Phase1][{epoch+1:03d}/{epochs_phase1}] '
            f'beta={current_beta:.4f} '
            f'total={running["total"]/num_batches:.4f} '
            f'elbo={running["elbo"]/num_batches:.4f} '
            f'reg={running["reg"]/num_batches:.4f} '
            f'nll={running["nll"]/num_batches:.4f} '
            f'klD={running["kl_D"]/num_batches:.4f} '
            f'klA={running["kl_A"]/num_batches:.4f}'
        )

    if range_compressor is not None:
        for p in range_compressor.parameters():
            p.requires_grad = False
        range_compressor.eval()

    for p in enc_A.parameters():
        p.requires_grad = False
    enc_A.eval()

    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs_phase2):
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
            with torch.no_grad():
                F_A_feat = extract_attribute_feature(
                    enc_A,
                    batch_X,
                    cfg=cfg,
                    crop_mode='center',
                    range_compressor=range_compressor,
                )

            logits = rec_net(F_A_feat)
            loss_ce = criterion(logits, batch_Y)
            loss_ce.backward()
            optimizer_RecNet.step()
            running_ce += float(loss_ce.detach().cpu())

        num_batches = max(1, len(dataloader))
        print(f'[Phase2][{epoch+1:03d}/{epochs_phase2}] ce={running_ce/num_batches:.4f}')

    # Attach the front-end to enc_A so downstream inference can stay API-compatible:
    # extract_attribute_feature(enc_A, raw_X, cfg) will automatically use it.
    if range_compressor is not None:
        enc_A.range_compressor = range_compressor

    return enc_D, enc_A, decoder, rec_net


__all__ = [
    'UDAFDConfig',
    'HRRPSequenceH5Dataset',
    'create_h5_dataloader',
    'SparseAttentionCompressor',
    'maybe_compress_sequence',
    'compress_mask_sequence',
    'adapt_sequence_for_attribute_encoder',
    'build_gp_covariance',
    'DynamicEncoder',
    'AttributeEncoder',
    'Decoder',
    'HRRP_RecNet',
    'split_windows',
    'crop_subsequence',
    'sample_dynamic_features',
    'compute_kl_dynamic',
    'compute_kl_attribute',
    'gaussian_nll',
    'compute_elbo_loss',
    'compute_counterfactual_reg',
    'extract_attribute_feature',
    'train_framework',
]
