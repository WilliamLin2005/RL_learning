import numpy as np
import matplotlib.pyplot as plt

# --- 0. 固定随机种子方便调试 ---
np.random.seed(42) 

# --- 1. Initialization ---
true_expectation = 10.0 
sigma = 2.0 
sample_size = 2000

# 生成样本序列 {x_k}
x_samples = np.random.normal(loc=true_expectation, scale=sigma, size=sample_size)

# 初始化 w 序列
w = np.zeros(sample_size + 1)
w[0] = x_samples[0] 

# --- 2. Robbins-Monro Algorithm ---
for i in range(1, sample_size + 1):
    # 核心公式: w_k+1 = w_k - a_k * (w_k - x_k)
    # 这里的 i 对应 k，步长 a_k = 1/i
    w[i] = w[i-1] - (w[i-1] - x_samples[i-1]) / i

# --- 3. 绘图展示部分 ---
plt.figure(figsize=(10, 8))

# ==========================================
# 子图 1: 收敛轨迹 (Convergence Trajectory)
# ==========================================
plt.subplot(2, 1, 1)
# 样本点 (Noisy Observations)
plt.scatter(range(1, sample_size + 1), x_samples, color='lightgray', s=10, alpha=0.5, label='Noisy Samples (x_k)')
# 真实期望线 (Ground Truth)
plt.axhline(y=true_expectation, color='red', linestyle='--', linewidth=2, label='True Expectation E[X]')
# RM 算法轨迹 (RM Estimation)
plt.plot(range(sample_size + 1), w, color='blue', linewidth=2, label='RM Trajectory (w_k)')

plt.title('Robbins-Monro Algorithm Convergence', fontsize=14)
plt.xlabel('Iteration (k)', fontsize=12)
plt.ylabel('Value', fontsize=12)
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)

# ==========================================
# 子图 2: 样本分布 (Histogram)
# ==========================================
plt.subplot(2, 1, 2)
plt.hist(x_samples, bins=40, color='skyblue', edgecolor='black', alpha=0.7, density=True)
plt.axvline(x=true_expectation, color='red', linestyle='--', linewidth=2, label='True Mean')
plt.axvline(x=w[-1], color='blue', linestyle='-.', linewidth=2, label=f'Final Est: {w[-1]:.2f}')

plt.title('Distribution of Samples X', fontsize=14)
plt.xlabel('Value of X', fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.legend()

plt.tight_layout()
plt.show()

print(f"Final Estimation: {w[-1]:.4f}")
print(f"Error: {abs(w[-1] - true_expectation):.4f}")