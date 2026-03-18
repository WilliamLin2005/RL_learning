import numpy as np

# 1. 状态转移矩阵 P (结合了策略 pi 之后的 MRP 转移概率)
p = [[0.0, 0.5, 0.5, 0.0],  # s1
     [0.0, 0.0, 0.0, 1.0],  # s2
     [0.0, 0.0, 0.0, 1.0],  # s3
     [0.0, 0.0, 0.0, 1.0]   # s4 (到达终点后不断循环)
] 
P = np.array(p)

# 2. 修正后的期望状态奖励 R(s)
# s1 的期望奖励: 0.5 * (-1) + 0.5 * 0 = -0.5
rewards =[-0.5, 1.0, 1.0, 1.0]
gamma = 0.9

# ---------------- 计算 State Value V(s) ----------------
def compute_v(P, rewards, gamma, states_num):
    rewards = np.array(rewards).reshape((-1, 1)) 
    value = np.dot(np.linalg.inv(np.eye(states_num) - gamma * P), rewards)
    return value

V = compute_v(P, rewards, gamma, 4)
print("State Value V(s) (as a column vector):", V)
print("\n")

print("========= State Value V(s) =========")
# V.flatten() 只是为了把 [[8.5], [10.]] 这种二维变成[8.5, 10.] 方便打印
V_flat = V.flatten() 
for i in range(4):
    print(f"V(s{i+1}) = {V_flat[i]:.2f}")


# ---------------- 计算 Action Value Q(s, a) ----------------
# 既然 V(s) 已经算出来了，Q(s,a) 只需要往前倒推一步即可
print("\n========= Action Value Q(s, a) =========")

# 状态 s1 的两个动作
# 选择向右: 奖励为 -1, 下一个状态必为 s2 (对应 V_flat[1])
Q_s1_right = -1 + gamma * V_flat[1]
print(f"Q(s1, right) = {Q_s1_right:.2f}")

# 选择向下: 奖励为 0, 下一个状态必为 s3 (对应 V_flat[2])
Q_s1_down = 0 + gamma * V_flat[2]
print(f"Q(s1, down)  = {Q_s1_down:.2f}")

# 状态 s2 的动作
# 选择向下: 奖励为 1, 下一个状态必为 s4 (对应 V_flat[3])
Q_s2_down = 1 + gamma * V_flat[3]
print(f"Q(s2, down)  = {Q_s2_down:.2f}")

# 状态 s3 的动作
# 选择向右: 奖励为 1, 下一个状态必为 s4 (对应 V_flat[3])
Q_s3_right = 1 + gamma * V_flat[3]
print(f"Q(s3, right) = {Q_s3_right:.2f}")

# 状态 s4 的动作 (终点保持)
Q_s4_stay = 1 + gamma * V_flat[3]
print(f"Q(s4, stay)  = {Q_s4_stay:.2f}")