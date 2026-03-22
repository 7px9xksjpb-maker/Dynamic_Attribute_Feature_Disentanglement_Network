
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# -----------------------------
# Paper-aligned default settings
# -----------------------------
T = 30
M = 101
N = 10
delta = 3
l = 9
d_D = 8
d_A = 8
beta = 0.1
lambda_reg = 0.5
lr_1 = 1e-3
lr_2 = 1e-3
batch_size = 128


def split_windows(X: torch.Tensor) -> torch.Tensor:
    """
    X: [B, T, M]
    return: [B, N, delta, M]
    """
    B, T_, M_ = X.shape
    assert T_ == T and M_ == M and T == N * delta
    return X.view(B, N, delta, M)


def crop_subsequence(
    X: torch.Tensor,
    crop_len: int = l,
    starts: Optional[torch.Tensor] = None,
    random_crop: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    X: [B, T, M]
    return: cropped [B, crop_len, M], starts [B]
    """
    B, T_, _ = X.shape
    assert crop_len <= T_

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


def build_gp_covariance(
    N_: int = N,
    d_D_: int = d_D,
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
        # NOTE:
        # The PDF rendering around Eq.(11) is a bit ambiguous.
        # This is the common, numerically stable implementation.
        k_cauchy = 1.0 / (1.0 + dist_sq / (lam * lam))
        kernels.extend([k_rbf, k_cauchy])

    Sigma_G = torch.stack(kernels, dim=0)
    Sigma_G = Sigma_G + epsilon * torch.eye(N_, device=device).unsqueeze(0)
    return Sigma_G


class DynamicEncoder(nn.Module):
    """
    Input:  [B, N, delta, M]
    Output: mu_D [B, N, d_D], gamma_D [B, N, d_D, 2]
    gamma_D[..., 0] -> diagonal of U
    gamma_D[..., 1] -> first super-diagonal of U
    """

    def __init__(self, input_dim: int = M, hidden_dim: int = 64, d_D_: int = d_D):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.d_D = d_D_

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            batch_first=True,
        )
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, d_D_)
        self.fc_gamma = nn.Linear(hidden_dim, 2 * d_D_)

    def forward(
        self,
        x_windows: torch.Tensor,
        m_windows: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B = x_windows.size(0)
        if m_windows is None:
            m_windows = torch.ones_like(x_windows)

        x = (x_windows * m_windows).reshape(B * N, delta, M)
        s, _ = self.lstm(x)
        s = torch.sigmoid(s[:, -1, :])

        r = F.relu(self.fc2(F.relu(self.fc1(s))))
        mu_D = self.fc_mu(r).view(B, N, self.d_D)
        gamma_D = F.softplus(self.fc_gamma(r)).view(B, N, self.d_D, 2) + 1e-4
        return mu_D, gamma_D


class AttributeEncoder(nn.Module):
    """
    Input: [B, l, M]
    Output: mu_A [B, d_A]
    Posterior is q(F_A|X) = N(mu_A, I)
    """

    def __init__(self, input_dim: int = M, hidden_dim: int = 64, d_A_: int = d_A):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            batch_first=True,
        )
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, d_A_)

    def forward(
        self,
        x_subseq: torch.Tensor,
        m_subseq: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if m_subseq is None:
            m_subseq = torch.ones_like(x_subseq)

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
    B = gamma_D.size(0)
    gamma = gamma_D.permute(0, 2, 1, 3)  # [B, d_D, N, 2]

    U = torch.zeros(B, d_D, N, N, device=gamma.device, dtype=gamma.dtype)
    idx = torch.arange(N, device=gamma.device)

    # diagonal
    U[..., idx, idx] = gamma[..., 0] + epsilon

    # first super-diagonal
    if N > 1:
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
    U = build_precision_cholesky(gamma_D)
    precision_q = U.transpose(-1, -2) @ U
    cov_q = torch.linalg.inv(precision_q)

    L_q = torch.linalg.cholesky(cov_q)  # [B, d_D, N, N]
    eps = torch.randn(mu_D.size(0), d_D, N, 1, device=mu_D.device, dtype=mu_D.dtype)

    mu = mu_D.permute(0, 2, 1).unsqueeze(-1)  # [B, d_D, N, 1]
    z = mu + L_q @ eps
    F_D = z.squeeze(-1).permute(0, 2, 1)  # [B, N, d_D]
    return F_D, cov_q, U


def compute_kl_dynamic(
    mu_D: torch.Tensor,
    cov_q: torch.Tensor,
    U: torch.Tensor,
    Sigma_G: torch.Tensor,
) -> torch.Tensor:
    """
    KL( q(F_D|X) || p(F_D) )
    q: product over d of N(mu_d, Sigma_q_d)
    p: product over d of N(0, Sigma_G_d)
    """
    mu = mu_D.permute(0, 2, 1)  # [B, d_D, N]

    # logdet(Sigma_q) = -logdet(U^T U) = -2 * sum(log diag(U))
    logdet_q = -2.0 * torch.log(torch.diagonal(U, dim1=-2, dim2=-1)).sum(dim=-1)  # [B, d_D]

    kl_all = []
    for d in range(d_D):
        Sigma_p = Sigma_G[d]  # [N, N]
        L_p = torch.linalg.cholesky(Sigma_p)
        Sigma_p_inv = torch.cholesky_inverse(L_p)
        logdet_p = 2.0 * torch.log(torch.diagonal(L_p)).sum()

        trace_term = torch.einsum("ij,bji->b", Sigma_p_inv, cov_q[:, d])
        maha_term = torch.einsum("bi,ij,bj->b", mu[:, d], Sigma_p_inv, mu[:, d])

        kl_d = 0.5 * (trace_term + maha_term - N + logdet_p - logdet_q[:, d])
        kl_all.append(kl_d)

    kl = torch.stack(kl_all, dim=1).sum(dim=1).mean()
    return kl


def compute_kl_attribute(mu_A: torch.Tensor) -> torch.Tensor:
    """
    q(F_A|X)=N(mu_A, I), p(F_A)=N(0, I)
    KL = 0.5 * ||mu_A||^2
    """
    return 0.5 * mu_A.pow(2).sum(dim=1).mean()


class Decoder(nn.Module):
    """
    Paper says:
      F' in R^{N x c1} -> LSTM -> L in R^{T x c1} -> F'' in R^{T x c2}
      -> mu_X, gamma_X in R^{T x M}

    Since the paper does not fully specify the N->T transition inside the decoder,
    we use a conservative surrogate:
      repeat_interleave(delta) to expand N steps to T steps, then feed LSTM.
    """

    def __init__(
        self,
        d_D_: int = d_D,
        d_A_: int = d_A,
        c1: int = 128,
        c2: int = 128,
        output_dim: int = M,
    ):
        super().__init__()
        self.fc1 = nn.Linear(d_D_ + d_A_, c1)
        self.bn = nn.BatchNorm1d(c1)
        self.lstm = nn.LSTM(input_size=c1, hidden_size=c1, batch_first=True)
        self.fc2 = nn.Linear(c1, c2)
        self.fc_mu_x = nn.Linear(c2, output_dim)
        self.fc_sigma_x = nn.Linear(c2, output_dim)

    def forward(
        self,
        F_D: torch.Tensor,
        F_A_rep: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # [B, N, d_D + d_A]
        h = torch.cat([F_D, F_A_rep], dim=-1)

        # FC -> BN -> Tanh
        h = self.fc1(h)  # [B, N, c1]
        h = self.bn(h.transpose(1, 2)).transpose(1, 2)
        h = torch.tanh(h)

        # conservative N -> T expansion
        if h.size(1) != T:
            h = h.repeat_interleave(delta, dim=1)
            if h.size(1) > T:
                h = h[:, :T, :]
            elif h.size(1) < T:
                pad = h[:, -1:, :].expand(-1, T - h.size(1), -1)
                h = torch.cat([h, pad], dim=1)

        L_out, _ = self.lstm(h)         # [B, T, c1]
        h2 = F.relu(self.fc2(L_out))    # [B, T, c2]

        mu_x = self.fc_mu_x(h2)                         # [B, T, M]
        sigma_x = torch.sigmoid(self.fc_sigma_x(h2))   # [B, T, M]
        sigma_x = sigma_x + 1e-4
        return mu_x, sigma_x


def gaussian_nll(
    X_true: torch.Tensor,
    mu_x: torch.Tensor,
    sigma_x: torch.Tensor,
) -> torch.Tensor:
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
    beta_: float = beta,
):
    nll = gaussian_nll(X_true, mu_x, sigma_x)
    kl_D = compute_kl_dynamic(mu_D, cov_q_D, U_D, Sigma_G)
    kl_A = compute_kl_attribute(mu_A)
    elbo = nll + beta_ * (kl_D + kl_A)
    stats = {
        "nll": float(nll.detach().cpu()),
        "kl_D": float(kl_D.detach().cpu()),
        "kl_A": float(kl_A.detach().cpu()),
        "elbo": float(elbo.detach().cpu()),
    }
    return elbo, stats


def compute_counterfactual_reg(
    attribute_encoder: AttributeEncoder,
    decoder: Decoder,
    F_D: torch.Tensor,
    F_A_original: torch.Tensor,
    crop_len: int = l,
) -> torch.Tensor:
    """
    Paper Eq.(39) uses a likelihood ratio under q(F_A | X*).
    Since q(F_A | X*) = N(mu_A(X*), I), we can compute the ratio
    up to constants using the encoder output mu_A(X*).

    We build:
        ratio = q(F_A_original | X*) / q(F_A_cf | X*)
    and minimize its expectation.
    """
    B = F_D.size(0)
    device = F_D.device
    dtype = F_D.dtype

    F_A_cf = torch.randn(B, d_A, device=device, dtype=dtype)
    F_A_cf_rep = F_A_cf.unsqueeze(1).expand(-1, N, -1)

    mu_x_cf, _ = decoder(F_D, F_A_cf_rep)   # deterministic surrogate of X*
    X_cf_crop, _ = crop_subsequence(mu_x_cf, crop_len=crop_len, random_crop=True)

    mu_A_cf_post = attribute_encoder(X_cf_crop)

    # q(f | X*) = N(mu_A_cf_post, I), constants cancel in ratio
    log_q_orig = -0.5 * (F_A_original - mu_A_cf_post).pow(2).sum(dim=1)
    log_q_cf = -0.5 * (F_A_cf - mu_A_cf_post).pow(2).sum(dim=1)

    log_ratio = (log_q_orig - log_q_cf).clamp(min=-30.0, max=30.0)
    ratio = torch.exp(log_ratio)
    return ratio.mean()


class HRRP_RecNet(nn.Module):
    def __init__(self, d_A_: int = d_A, num_classes: int = 3):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(d_A_, 32),
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
    crop_mode: str = "center",
    num_random_crops: int = 1,
) -> torch.Tensor:
    """
    In phase 2 / inference:
    - 'center' is deterministic and stable
    - 'random_avg' averages multiple random crops
    """
    if crop_mode == "center":
        X_sub, _ = crop_subsequence(X, crop_len=l, random_crop=False)
        return enc_A(X_sub)

    if crop_mode == "random_avg":
        feats = []
        for _ in range(num_random_crops):
            X_sub, _ = crop_subsequence(X, crop_len=l, random_crop=True)
            feats.append(enc_A(X_sub))
        return torch.stack(feats, dim=0).mean(dim=0)

    raise ValueError(f"Unknown crop_mode: {crop_mode}")


def train_framework(
    dataloader,
    num_classes: int = 3,
    epochs_phase1: int = 70,
    epochs_phase2: int = 40,
    device: Optional[torch.device] = None,
):
    """
    Expected dataloader outputs:
        either (X, Y)
        or     (X, Y, X_mask)

    X:      [B, 30, 101]
    Y:      [B]
    X_mask: [B, 30, 101], optional
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    enc_D = DynamicEncoder().to(device)
    enc_A = AttributeEncoder().to(device)
    decoder = Decoder().to(device)
    rec_net = HRRP_RecNet(num_classes=num_classes).to(device)

    optimizer_UDAFD = optim.Adam(
        list(enc_D.parameters()) +
        list(enc_A.parameters()) +
        list(decoder.parameters()),
        lr=lr_1,
    )
    optimizer_RecNet = optim.Adam(rec_net.parameters(), lr=lr_2)

    Sigma_G = build_gp_covariance(device=device)

    # -----------------------------
    # Phase 1: train UDAFD-Net
    # -----------------------------
    for epoch in range(epochs_phase1):
        enc_D.train()
        enc_A.train()
        decoder.train()

        running = {"total": 0.0, "elbo": 0.0, "reg": 0.0, "nll": 0.0, "kl_D": 0.0, "kl_A": 0.0}

        for batch in dataloader:
            if len(batch) == 3:
                batch_X, _, batch_mask = batch
                batch_mask = batch_mask.to(device)
            else:
                batch_X, _ = batch[:2]
                batch_mask = torch.ones_like(batch_X)

            batch_X = batch_X.to(device)
            batch_mask = batch_mask.to(device)

            optimizer_UDAFD.zero_grad()

            X_windows = split_windows(batch_X)
            M_windows = split_windows(batch_mask)

            X_subseq, starts = crop_subsequence(batch_X, crop_len=l, random_crop=True)
            M_subseq, _ = crop_subsequence(batch_mask, crop_len=l, starts=starts, random_crop=False)

            mu_D, gamma_D = enc_D(X_windows, M_windows)
            mu_A = enc_A(X_subseq, M_subseq)

            F_D, cov_q_D, U_D = sample_dynamic_features(mu_D, gamma_D)
            F_A = mu_A + torch.randn_like(mu_A)

            F_A_rep = F_A.unsqueeze(1).expand(-1, N, -1)
            mu_x, sigma_x = decoder(F_D, F_A_rep)

            loss_elbo, stats = compute_elbo_loss(
                X_true=batch_X,
                mu_x=mu_x,
                sigma_x=sigma_x,
                mu_D=mu_D,
                cov_q_D=cov_q_D,
                U_D=U_D,
                mu_A=mu_A,
                Sigma_G=Sigma_G,
                beta_=beta,
            )

            loss_reg = compute_counterfactual_reg(
                attribute_encoder=enc_A,
                decoder=decoder,
                F_D=F_D,
                F_A_original=F_A,
                crop_len=l,
            )

            loss = loss_elbo + lambda_reg * loss_reg
            loss.backward()
            optimizer_UDAFD.step()

            running["total"] += float(loss.detach().cpu())
            running["elbo"] += float(loss_elbo.detach().cpu())
            running["reg"] += float(loss_reg.detach().cpu())
            running["nll"] += stats["nll"]
            running["kl_D"] += stats["kl_D"]
            running["kl_A"] += stats["kl_A"]

        num_batches = max(1, len(dataloader))
        print(
            f"[Phase1][{epoch+1:03d}/{epochs_phase1}] "
            f"total={running['total']/num_batches:.4f} "
            f"elbo={running['elbo']/num_batches:.4f} "
            f"reg={running['reg']/num_batches:.4f} "
            f"nll={running['nll']/num_batches:.4f} "
            f"klD={running['kl_D']/num_batches:.4f} "
            f"klA={running['kl_A']/num_batches:.4f}"
        )

    # -----------------------------
    # Phase 2: freeze attribute encoder, train HRRP-RecNet
    # -----------------------------
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
                    crop_mode="center",
                )

            logits = rec_net(F_A_feat)
            loss_ce = criterion(logits, batch_Y)
            loss_ce.backward()
            optimizer_RecNet.step()

            running_ce += float(loss_ce.detach().cpu())

        num_batches = max(1, len(dataloader))
        print(f"[Phase2][{epoch+1:03d}/{epochs_phase2}] ce={running_ce/num_batches:.4f}")

    return enc_D, enc_A, decoder, rec_net


__all__ = [
    "T", "M", "N", "delta", "l", "d_D", "d_A",
    "build_gp_covariance",
    "DynamicEncoder", "AttributeEncoder", "Decoder", "HRRP_RecNet",
    "sample_dynamic_features",
    "compute_kl_dynamic", "compute_kl_attribute",
    "gaussian_nll", "compute_elbo_loss", "compute_counterfactual_reg",
    "extract_attribute_feature", "train_framework",
]
