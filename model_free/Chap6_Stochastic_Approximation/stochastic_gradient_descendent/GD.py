import numpy as np
import matplotlib.pyplot as plt

class GradientDescent:
    def __init__(self, initial_w, sample_size):
        self.w = np.array(initial_w, dtype=float)
        self.sample_size = sample_size  # 这里的 sample_size 代表 batch size (m)
        self.w_history = [self.w.copy()]

    def gradient_step(self, x_samples, total_samples, theta, mean_point, max_iters=1000):
        # ================== 1. BGD (m = total_samples) ==================
        if self.sample_size == total_samples:
            # BGD 每次迭代都用所有样本的均值
            x_sample_mean = np.mean(x_samples, axis=0)
            iterations = 1
            while iterations <= max_iters:
                alpha_k = 1.0 / iterations
                self.w = self.w - alpha_k * (self.w - x_sample_mean)
                self.w_history.append(self.w.copy())
                
                if np.linalg.norm(self.w - mean_point) < theta:
                    break
                iterations += 1

        # ================== 2. MBGD (1 < m < total_samples) ==================
        elif 1 < self.sample_size < total_samples:
            iterations = 1
            while iterations <= max_iters:
                alpha_k = 1.0 / iterations
                # 二维数组抽样方式：先抽索引
                indices = np.random.choice(len(x_samples), size=self.sample_size, replace=False)
                x_sample_arr = x_samples[indices]
                # 按列求均值，保持二维 [x, y]
                x_sample_mean = np.mean(x_sample_arr, axis=0) 
                
                self.w = self.w - alpha_k * (self.w - x_sample_mean)
                self.w_history.append(self.w.copy())
                
                if np.linalg.norm(self.w - mean_point) < theta:
                    break
                iterations += 1

        # ================== 3. SGD (m = 1) ==================
        else:
            iterations = 1 # 初始化必须在循环外面
            while iterations <= max_iters:
                alpha_k = 1.0 / iterations
                # 随机抽取 1 个样本
                idx = np.random.randint(len(x_samples))
                x = x_samples[idx]
                
                self.w = self.w - alpha_k * (self.w - x)
                self.w_history.append(self.w.copy())
                
                if np.linalg.norm(self.w - mean_point) < theta:
                    break
                iterations += 1

# ==========================================
# 主函数与绘图部分 
# ==========================================
if __name__ == "__main__":
    # --- 1. 参数初始化 ---
    np.random.seed(42)  
    mean_point = np.array([0.0, 0.0])  
    sigma = 4.0
    
    # A. 生成一个代表“总体”的、非常大的分布
    population_size = 100000 
    population_samples = np.random.normal(loc=mean_point, scale=sigma, size=(population_size, 2))
    
    # B. 从“总体”中，抽取一个我们真正拥有的、固定的“训练集”
    training_set_size = 2000 
    training_indices = np.random.choice(len(population_samples), size=training_set_size, replace=False)
    training_set = population_samples[training_indices]
    
    theta = 1e-4                       
    max_plot_iters = 30                
    initial_w = np.array([-20.0, 20.0]) 

    # --- 2. 运行四种算法 (全部基于固定的 training_set) ---
    # ① SGD (m=1)
    sgd_solver = GradientDescent(initial_w, sample_size=1)
    sgd_solver.gradient_step(training_set, training_set_size, theta, mean_point)
    sgd_history = np.array(sgd_solver.w_history)[:max_plot_iters+1]

    # ② MBGD (m=5)
    mbgd5_solver = GradientDescent(initial_w, sample_size=5)
    mbgd5_solver.gradient_step(training_set, training_set_size, theta, mean_point)
    mbgd5_history = np.array(mbgd5_solver.w_history)[:max_plot_iters+1]

    # ③ MBGD (m=50)
    mbgd50_solver = GradientDescent(initial_w, sample_size=50)
    mbgd50_solver.gradient_step(training_set, training_set_size, theta, mean_point)
    mbgd50_history = np.array(mbgd50_solver.w_history)[:max_plot_iters+1]

    # ④ BGD (m=training_set_size)
    bgd_solver = GradientDescent(initial_w, sample_size=training_set_size)
    bgd_solver.gradient_step(training_set, training_set_size, theta, mean_point)
    bgd_history = np.array(bgd_solver.w_history)[:max_plot_iters+1]

    # --- 3. 计算距离 ---
    def calc_distances(history, true_mean):
        return [np.linalg.norm(w - true_mean) for w in history]

    dist_sgd = calc_distances(sgd_history, mean_point)
    dist_mbgd5 = calc_distances(mbgd5_history, mean_point)
    dist_mbgd50 = calc_distances(mbgd50_history, mean_point)
    dist_bgd = calc_distances(bgd_history, mean_point)

    # --- 4. 开始绘图 (已修正变量名和缩进) ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # ================= 子图 1: 2D 轨迹平面图 =================
    # 使用 training_set 画散点
    ax1.scatter(training_set[:100, 0], training_set[:100, 1], facecolors='none', edgecolors='k', s=30, label='Samples')
    ax1.plot(mean_point[0], mean_point[1], 'ko', markerfacecolor='#0072BD', markersize=10, markeredgewidth=2.5, label='Mean')
    
    ax1.plot(sgd_history[:, 0], sgd_history[:, 1], '-d', color='#ED7D31', label='SGD (m=1)', linewidth=1.5, markersize=5)
    ax1.plot(mbgd5_history[:, 0], mbgd5_history[:, 1], '->', color='#77AC30', label='MBGD (m=5)', linewidth=1.5, markersize=6)
    ax1.plot(mbgd50_history[:, 0], mbgd50_history[:, 1], '-*', color='#0072BD', label='MBGD (m=50)', linewidth=1.5, markersize=5)
    # 使用 training_set_size 更新标签
    ax1.plot(bgd_history[:, 0], bgd_history[:, 1], '-s', color='#7E2F8E', label=f'BGD (m={training_set_size})', linewidth=2, markersize=5)

    ax1.set_xlabel('x', fontsize=12)
    ax1.set_ylabel('y', fontsize=12)
    ax1.legend(loc='upper right', framealpha=1, edgecolor='k')

    # ================= 子图 2: 距离衰减曲线 =================
    iters_sgd = range(len(dist_sgd))
    iters_mbgd5 = range(len(dist_mbgd5))
    iters_mbgd50 = range(len(dist_mbgd50))
    iters_bgd = range(len(dist_bgd))

    ax2.plot(iters_sgd, dist_sgd, '-d', color='#ED7D31', label='SGD (m=1)', linewidth=1.5, markersize=5)
    ax2.plot(iters_mbgd5, dist_mbgd5, '->', color='#77AC30', label='MBGD (m=5)', linewidth=1.5, markersize=6)
    ax2.plot(iters_mbgd50, dist_mbgd50, '-*', color='#0072BD', label='MBGD (m=50)', linewidth=1.5, markersize=5)
    ax2.plot(iters_bgd, dist_bgd, '-s', color='#7E2F8E', label=f'BGD (m={training_set_size})', linewidth=2, markersize=5)

    ax2.set_xlabel('Iteration step', fontsize=12)
    ax2.set_ylabel('Distance to mean', fontsize=12)
    ax2.set_xlim([0, 30])
    ax2.set_ylim(bottom=0)
    ax2.legend(loc='upper right', framealpha=1, edgecolor='k')

    plt.tight_layout()
    plt.show()