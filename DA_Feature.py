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
def compute_elbo_loss(X_true, mu_x, gamma_x, mu_D, gamma_D, mu_A, beta=beta):
    # 1. NLL (重构损失) - 假定高斯似然 
    recon_loss = F.mse_loss(mu_x, X_true) # 简化表示
    
    # 2. 动态特征 KL 散度 (逼近高斯过程先验) 
    # (注：此处需实现针对 GP 先验的多核协方差矩阵 Sigma_G 的计算，此处用占位计算表示)
    kl_D = torch.mean(torch.sum(mu_D**2, dim=1)) # 简化计算，实际需基于公式 (37)
    
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
    
    # ==========================================
    # Phase 1: 训练 UDAFD-Net 实现特征解耦 
    # ==========================================
    epochs_phase1 = 70 # [cite: 696]
    for epoch in range(epochs_phase1):
        enc_D.train(); enc_A.train(); decoder.train()
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
            loss_elbo = compute_elbo_loss(batch_X, mu_x, gamma_x, mu_D, gamma_D, mu_A)
            loss_reg = compute_counterfactual_reg(enc_A, decoder, F_D, mu_A)
            
            # 总体损失 
            L_ALL = loss_elbo + lambda_reg * loss_reg
            L_ALL.backward()
            optimizer_UDAFD.step() # 更新 UDAFD 参数 

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
