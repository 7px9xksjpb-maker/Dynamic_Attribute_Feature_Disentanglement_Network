import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm

# 1. 准备数据 (在实际应用中，你会加载雷达数据矩阵)
# 假设我们生成一个简单的示例矩阵来演示
n_range = 101     # X轴刻度数
n_sampling = 31   # Y轴刻度数

x = np.linspace(0, 100, n_range)
y = np.linspace(0, 30, n_sampling)

# 创建网格数据
X, Y = np.meshgrid(x, y)

# 实际应用中，这里是你处理好的 31x101 的幅度矩阵
# 为了演示，我们生成一些看起来像数据的噪声和峰值
Z = -30 + np.random.randn(n_sampling, n_range) * 2  # 基底噪声
for i in range(n_sampling):
    peak_pos = 50 + 5 * np.sin(i / 5) # 峰值位置随采样指数变化
    Z[i, :] += 20 * np.exp(-((x - peak_pos)**2) / (2 * 10**2))

# 2. 绘制 3D 表面图
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# 使用特定颜色映射（cmap），如 'viridis' 或 'jet'，来模拟原图效果
surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')

# 3. 美化
# 添加颜色条
cbar = fig.colorbar(surf, shrink=0.5, aspect=5)
cbar.set_label('Amplitude', rotation=270, labelpad=15)

# 设置轴标签
ax.set_xlabel('Range Cell')
ax.set_ylabel('Sampling Index')
ax.set_zlabel('Amplitude')

# 设置标题
ax.set_title('F15 HRRP sequence sample')

# 设置刻度范围（可选）
ax.set_xlim(0, 100)
ax.set_ylim(0, 30)
ax.set_zlim(-50, 20)

# 调整视角（仰角和方位角），使图表具有原图的透视感
ax.view_init(elev=30, azim=-120)

plt.show()