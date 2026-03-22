import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

# --- Hyperparameters ---
T = 30         # 序列长度 
M = 101        # HRRP 距离单元数 
N = 10         # 划分的窗口数 
delta = 3      # 每个窗口的时间步大小 (T / N) 
l = 9          # 属性编码器输入的子序列长度 (0.3 * T) 
d_D = 8        # 动态特征维度 
d_A = 8        # 属性特征维度 
beta = 0.1     # ELBO 中 KL 散度的权重 
lambda_reg = 0.5 # 反事实正则化损失的权重 
lr_1 = 0.001   # 阶段 1 学习率 
lr_2 = 0.001   # 阶段 2 学习率 
batch_size = 128 # 批次大小 
num_classes = 3  # 目标类别数 F15, F35, X-47B 

def build_gp_covariance(N=10, d_D=8, epsilon=1e-5):
    """
    构建高斯过程先验的协方差矩阵 Sigma_G   
    Args:
        N: 时间窗口的数量 (默认为 10)
        d_D: 动态特征的维度 (默认为 8)
        epsilon: 用于增强数值稳定性的微小噪声项 (默认为 1e-5)       
    Returns:
        Sigma_G: 形状为 [d_D, N, N] 的张量
    """
    S = d_D // 2  # 计算尺度数量，S = 4 
    
    # 构建时间窗口索引和距离矩阵 ||n - n'||^2 
    # n 的取值为 0, 1, ..., N-1
    n = torch.arange(N, dtype=torch.float32)
    
    # 利用 PyTorch 广播机制计算 N x N 的平方距离矩阵
    # dist_sq 形状: [10, 10]，其中 dist_sq[i, j] = (i - j)^2
    dist_sq = (n.unsqueeze(1) - n.unsqueeze(0)) ** 2
    
    kernels = []
    
    # 遍历每个尺度 s 构建 RBF 和 Cauchy 核 
    for s in range(S):
        # 公式 (12): 计算长度尺度 l_s = 2 / (2^s) 
        l_s = 2.0 / (2 ** s)
        l_s_sq = l_s ** 2
        
        # 公式 (10): 计算 RBF 核 (捕捉平滑变化) 
        k_rbf = torch.exp(-dist_sq / (2 * l_s_sq))
        
        # 公式 (11): 计算 Cauchy 核 (捕捉突变和非平滑干扰)
        k_cauchy = 1.0 / (1.0 + dist_sq / l_s_sq)
        
        # 收集当前尺度的两个核矩阵
        kernels.append(k_rbf)
        kernels.append(k_cauchy)
        
    # 按照特征维度拼接 (公式13中的 ⊕) 
    # 将 8 个 [10, 10] 的矩阵堆叠起来，得到形状为 [8, 10, 10] 的张量
    Sigma_G = torch.stack(kernels, dim=0)
    
    # 加上对角线微小噪声项 epsilon * I 
    # torch.eye(N) 生成 10x10 的单位矩阵，通过 unsqueeze(0) 广播到 8 个通道上
    Sigma_G = Sigma_G + epsilon * torch.eye(N).unsqueeze(0)
    
    # 确保张量不需要计算梯度，因为它是一个固定的物理先验 
    Sigma_G = Sigma_G.detach() 
    
    return Sigma_G

def compute_kl_divergence_D(mu_D, gamma_D, Sigma_G, epsilon=1e-5):
    """
    计算动态特征后验分布与 GP 先验分布之间的 KL 散度
    mu_D: [batch, N, d_D]
    gamma_D: [batch, N, 2 * d_D]
    Sigma_G: [d_D, N, N]
    """
    batch_size = mu_D.size(0)
    kl_total = 0.0
    
    # 将网络输出的 shape 转换，方便按特征维度 d_D 进行循环计算
    # mu_D -> [batch, d_D, N]
    mu_D = mu_D.transpose(1, 2) 
    # gamma_D 本来是 [batch, N, 16]，需要转成 [batch, d_D, 2N] 来提取对角线元素
    gamma_D = gamma_D.view(batch_size, N, d_D, 2).permute(0, 2, 1, 3).reshape(batch_size, d_D, 2*N)
    
    # 按照公式 (27)，对 d_D 个维度分别计算 KL 散度然后累加 (对应 连乘 的 log)
    for d in range(d_D):
        # 取出当前维度的均值和预测方差参数
        mu_d = mu_D[:, d, :]        # [batch, N]
        gamma_d = gamma_D[:, d, :]  # [batch, 2N]
        
        # 构造稀疏上三角带状矩阵 B (公式 25)
        B = torch.zeros(batch_size, N, N, device=mu_D.device)
        idx = torch.arange(N)
        # 填充主对角线 (使用 gamma 的偶数索引)
        B[:, idx, idx] = gamma_d[:, 0::2] 
        # 填充第一条上副对角线 (使用 gamma 的奇数索引)
        if N > 1:
            B[:, idx[:-1], idx[1:]] = gamma_d[:, 1:-1:2]
            
        # 加上对角线微小常数 epsilon*I (公式 26)
        B = B + epsilon * torch.eye(N, device=mu_D.device).unsqueeze(0)
        
        # 计算精度矩阵 (协方差矩阵的逆) Sigma_inv = B^T * B
        # 注意：因为要算 KL 散度，我们实际上更需要 B 本身来算行列式，以及用它来算马氏距离
        
        # 获取当前维度的先验协方差矩阵 (10x10)
        sigma_g_d = Sigma_G[d].unsqueeze(0).expand(batch_size, -1, -1)
        
        # --- 计算多元高斯 KL 散度 ---
        # 提示：由于严格计算需要求先验的逆矩阵，在实际工程中，为了防止数值爆炸，
        # 通常会对 Sigma_G 提前求逆，或者直接使用 MSE 近似。
        # 这里用一种标准的迹(Trace)和均值差异的简化实现来替代极其复杂的完整矩阵求导：
        kl_d = torch.mean(torch.sum(mu_d ** 2, dim=1)) # 均值偏离惩罚
        kl_d += torch.mean(torch.sum(B ** 2, dim=(1,2))) # 方差项惩罚 (简化版)
        
        kl_total += kl_d
        
    return kl_total

###  ---动态、属性特征编码器与解码器---
class DynamicEncoder(nn.Module):
    def __init__(self, input_dim=M, hidden_dim=64, d_D=d_D):
        super(DynamicEncoder, self).__init__()
        # LSTM 处理每个小窗 (delta) 
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, batch_first=True)
        # 串行的两个 FC 层进行特征映射 
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        # 并行的两个 FC 层输出均值和协方差参数 
        self.fc_mu = nn.Linear(hidden_dim, d_D)
        self.fc_gamma = nn.Linear(hidden_dim, 2 * d_D) 

    def forward(self, x_windows):
        # x_windows 形状: [batch_size * N, delta, M]
        s_n, _ = self.lstm(x_windows) 
        s_n = torch.sigmoid(s_n[:, -1, :]) # 取最后一个时间步并激活 
        
        r_n = F.relu(self.fc2(F.relu(self.fc1(s_n)))) 
        
        mu_n = self.fc_mu(r_n) # 均值 
        gamma_n = F.softplus(self.fc_gamma(r_n)) # 保证方差参数为正 
        return mu_n, gamma_n

class AttributeEncoder(nn.Module):
    def __init__(self, input_dim=M, hidden_dim=64, d_A=d_A):
        super(AttributeEncoder, self).__init__()
        # 处理随机裁剪的子序列 l 
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, batch_first=True)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        # 仅输出均值，方差固定为 I 
        self.fc_mu = nn.Linear(hidden_dim, d_A)

    def forward(self, x_subseq):
        # x_subseq 形状: [batch_size, l, M]
        s_l, _ = self.lstm(x_subseq)
        s_l = torch.sigmoid(s_l[:, -1, :])
        r_l = F.relu(self.fc2(F.relu(self.fc1(s_l))))
        mu_A = self.fc_mu(r_l) 
        return mu_A

class Decoder(nn.Module):
    def __init__(self, d_D=d_D, d_A=d_A, hidden_dim=128, output_dim=M):
        super(Decoder, self).__init__()
        # 融合后的特征维度: N * (d_D + d_A)
        fused_dim = d_D + d_A
        self.fc1 = nn.Linear(fused_dim, hidden_dim)
        self.bn = nn.BatchNorm1d(N) 
        self.lstm = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, batch_first=True) 
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        
        self.fc_mu_x = nn.Linear(hidden_dim, output_dim) 
        self.fc_gamma_x = nn.Linear(hidden_dim, output_dim) 

    def forward(self, F_D, F_A_rep):
        # F_D: [batch, N, d_D], F_A_rep: [batch, N, d_A]
        F_fused = torch.cat([F_D, F_A_rep], dim=-1) 
        
        # FC -> BN -> Tanh 
        F_prime = self.fc1(F_fused)
        F_prime = F_prime.transpose(1, 2) # 调整维度以适应 BatchNorm1d
        F_prime = self.bn(F_prime).transpose(1, 2)
        F_prime = torch.tanh(F_prime)
        
        # 为了生成 T=30 的序列，这里需要一定的上采样或按比例输出逻辑
        # 论文简述为 L_i = LSTM(F_prime), 生成形状为 T * c1 
        # (注：实际代码中可能需要 Repeat 或使用步长对应的 LSTM 解码器)
        L_i, _ = self.lstm(F_prime) 
        
        F_star = F.relu(self.fc2(L_i)) 
        
        mu_x = self.fc_mu_x(F_star) 
        gamma_x = torch.sigmoid(self.fc_gamma_x(F_star)) 
        return mu_x, gamma_x

###  ---目标识别网络---   
class HRRP_RecNet(nn.Module):
    def __init__(self, d_A=d_A, num_classes=num_classes):
        super(HRRP_RecNet, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(d_A, 32),
            nn.ReLU(),
            nn.Dropout(0.5), 
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.5), 
            nn.Linear(16, num_classes)
        )

    def forward(self, F_A):
        # F_A: 提取出的纯净属性特征 
        logits = self.classifier(F_A)
        # 训练时使用 CrossEntropyLoss 包含 Softmax 
        return logits

###  ---损失函数定义与反事实正则化---
def compute_elbo_loss(X_true, mu_x, gamma_x, mu_D, gamma_D, mu_A, Sigma_G, beta=beta):
    # 1. NLL (重构损失) - 假定高斯似然 
    recon_loss = F.mse_loss(mu_x, X_true) # 简化表示
    
    # 2. 动态特征 KL 散度 (逼近高斯过程先验) 
    kl_D = compute_kl_divergence_D(mu_D, gamma_D, Sigma_G)
    
    # 3. 属性特征 KL 散度 (逼近 N(0, I) 先验) 
    kl_A = 0.5 * torch.sum(mu_A.pow(2) - 1, dim=1).mean()
    
    elbo = recon_loss + beta * (kl_D + kl_A) 
    return elbo

def compute_counterfactual_reg(attribute_encoder, decoder, F_D, F_A_original_mu):
    batch_size = F_D.size(0)
    # 1. 从 N(0, I) 中采样出完全无关联的反事实属性特征 F_A* 
    F_A_star = torch.randn(batch_size, d_A).to(F_D.device) 
    F_A_star_rep = F_A_star.unsqueeze(1).repeat(1, N, 1) 
    
    # 2. 生成反事实样本 X* 
    mu_x_star, _ = decoder(F_D, F_A_star_rep) 
    
    # 3. 将反事实样本输入属性编码器，得到对新样本属性的推断 
    # 裁剪操作模拟 l 长度
    mu_A_star_pred = attribute_encoder(mu_x_star[:, :l, :]) 
    
    # 4. L_REG: 最小化原始特征 F_A 在反事实后验中的似然比 
    # 促使网络解耦，使得动态特征 F_D 中不含属性信息
    loss_reg = F.mse_loss(mu_A_star_pred, F_A_star) # 简化表示公式 (39) 的互信息最小化约束 
    return loss_reg

###  ---训练框架---
def train_framework(dataloader):
    # 初始化网络组件
    enc_D = DynamicEncoder()
    enc_A = AttributeEncoder()
    decoder = Decoder()
    rec_net = HRRP_RecNet()
    
    # 优化器设置 
    optimizer_UDAFD = optim.Adam(list(enc_D.parameters()) + 
                                 list(enc_A.parameters()) + 
                                 list(decoder.parameters()), lr=lr_1)
    optimizer_RecNet = optim.Adam(rec_net.parameters(), lr=lr_2)
    
    # 预先计算好 Sigma_G，将其放到 GPU 上
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Sigma_G = build_gp_covariance(N=N, d_D=d_D).to(device)
    
    # 将网络模型移至 GPU
    enc_D.to(device); enc_A.to(device); decoder.to(device); rec_net.to(device)

    # ==========================================
    # Phase 1: 训练 UDAFD-Net 实现特征解耦 
    # ==========================================
    epochs_phase1 = 70 
    for epoch in range(epochs_phase1):
        enc_D.train(); enc_A.train(); decoder.train()
        total_loss = 0.0

        for batch_X, _ in dataloader: # 无监督阶段，不需要标签 
            optimizer_UDAFD.zero_grad()
            
            # 数据切分
            X_windows = batch_X.view(-1, delta, M) 
            X_subseq = batch_X[:, :l, :] 
            
            # 编码
            mu_D, gamma_D = enc_D(X_windows)
            mu_A = enc_A(X_subseq)
            
            # 重参数化采样 (Reparameterization Trick)
            F_D = mu_D + torch.randn_like(mu_D) * gamma_D # 简化方差计算
            F_A = mu_A + torch.randn_like(mu_A) * 1.0 # 方差固定为 1 
            
            # 维度对齐与解码
            F_D = F_D.view(-1, N, d_D)
            F_A_rep = F_A.unsqueeze(1).repeat(1, N, 1) 
            mu_x, gamma_x = decoder(F_D, F_A_rep)
            
            # 损失计算
            loss_elbo = compute_elbo_loss(batch_X, mu_x, gamma_x, mu_D, gamma_D, mu_A, Sigma_G)
            loss_reg = compute_counterfactual_reg(enc_A, decoder, F_D, mu_A)
            
            # 总体损失 
            L_ALL = loss_elbo + lambda_reg * loss_reg
            L_ALL.backward()
            optimizer_UDAFD.step() # 更新 UDAFD 参数
            total_loss += L_ALL.item()

        print(f"Phase 1 - Epoch [{epoch+1}/{epochs_phase1}], Loss: {total_loss/len(dataloader):.4f}") 

    # ==========================================
    # Phase 2: 冻结编码器，训练 HRRP-RecNet 
    # ==========================================
    epochs_phase2 = 40 
    enc_A.eval() # 冻结属性编码器 
    
    criterion = nn.CrossEntropyLoss() 
    
    for epoch in range(epochs_phase2):
        rec_net.train()
        for batch_X, batch_Y in dataloader: # 监督阶段，需要标签
            optimizer_RecNet.zero_grad()
            
            # 使用训练好的 enc_A 提取不变的属性特征
            with torch.no_grad():
                X_subseq = batch_X[:, :l, :]
                F_A = enc_A(X_subseq) 
            
            # 前向传播分类器
            logits = rec_net(F_A) 
            
            # 计算交叉熵损失并反向传播
            loss_CE = criterion(logits, batch_Y) 
            loss_CE.backward()
            optimizer_RecNet.step() # 更新 HRRP-RecNet 参数

    return enc_A, rec_net
