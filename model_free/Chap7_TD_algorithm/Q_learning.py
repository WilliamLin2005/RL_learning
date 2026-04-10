import os
import sys

# 1. 使用 abspath(__file__) 获取当前文件的绝对路径
# 2. 向上返回 4 层，到达 python_rl_learn 这一级
current_path = os.path.abspath(__file__)
for _ in range(4):
    current_path = os.path.dirname(current_path)

root_path = current_path

# 3. 把这个根目录加入搜索路径的最前面
if root_path not in sys.path:
    sys.path.insert(0, root_path)
import random
import time
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from RL_learning.model_free.envs.grid_env import GridEnv
import matplotlib.pyplot as plt

class Q_learning:
    def __init__(self, env):
        self.gamma = 0.9
        self.env = env
        self.action_space_size = env.action_space_size
        self.state_space_size = env.size**2
        self.reward_list = env.reward_list
        
        # 初始化 Q 表
        self.qsa_value = np.zeros(shape=(self.state_space_size, self.action_space_size))
        
        # 平均策略初始化
        self.mean_policy = np.ones(shape=(self.state_space_size, self.action_space_size)) / self.action_space_size
        self.policy = self.mean_policy.copy()
        
        self.writer = SummaryWriter("logs")

        print("action_space_size: {} state_space_size：{}".format(self.action_space_size, self.state_space_size))
        print('----------------------------------------------------------------')

    def show_policy(self):
        for state in range(self.state_space_size):
            for action in range(self.action_space_size):
                prob = self.policy[state, action]
                if prob > 0: # 只画有概率的动作
                    self.env.render_.draw_action(pos=self.env.state2pos(state),
                                                 toward=prob * 0.25 * self.env.action_to_direction[action],
                                                 radius=prob * 0.05)

    def show_state_value(self, state_value, y_offset=0.2):
        for state in range(self.state_space_size):
            self.env.render_.write_word(pos=self.env.state2pos(state), word=str(round(state_value[state], 1)),
                                        y_offset=y_offset,
                                        size_discount=0.7)

    def obtain_experience(self, start_state, start_action):
            # 1. 接受 step 的 5 个返回值（用 _ 忽略不需要的 truncated 和 info）
            obs, reward, terminated, _, _ = self.env.step(start_action)
            
            # 2. obs 是个字典，必须提取 "agent" 坐标并转换为整数状态
            new_state = self.env.pos2state(obs["agent"])
            
            done = terminated
            
            best_next_action = np.argmax(self.qsa_value[new_state,:])
            next_action = np.random.choice(self.action_space_size, p=self.mean_policy[new_state, :]) 
            
            experience = {
                'state_t': start_state,
                'action_t': start_action,
                'reward': reward,
                'state_tp1': new_state,
                'action_tp1': next_action,
                'best_action_tp1': best_next_action,
                'done': done
            }
            return experience

    def Q_learning(self,max_iter=1000,alpha=0.1):
        reward_history = []
        length_history = []
        for episode in tqdm(range(max_iter)):
            # 1. 初始化环境，获取初始状态
            obs,_ = self.env.reset()
            state = self.env.pos2state(obs["agent"])
            # 2. 根据平均策略选择一个动作
            action = np.random.choice(self.action_space_size, p=self.mean_policy[state, :])
            done = False
            td_error = 0
            iteration = 0
            total_reward = 0
            step_count = 0
            while not done:
                experience = self.obtain_experience(state, action)
                state, action, reward, next_state, next_action,best_next_action, done = experience['state_t'], experience['action_t'], experience['reward'], experience['state_tp1'], experience['action_tp1'], experience['best_action_tp1'], experience['done']
                total_reward += reward
                step_count += 1

                # 2. 更新 Q 表
                td_target = reward + self.gamma * self.qsa_value[next_state, best_next_action] * (1 - done)
                td_error =  self.qsa_value[state, action]- td_target
                self.qsa_value[state, action] -= alpha * td_error
                    
                # 3. 更新策略
                best_action = np.argmax(self.qsa_value[state, :])
                self.policy[state, :] = 0
                self.policy[state, best_action] =1.0
                #4.更新状态和动作
                state = next_state
                action = next_action

                iteration += 1
                if iteration>1000:
                    break
            reward_history.append(total_reward)
            length_history.append(step_count)   
          
        # 训练结束，计算 state value 
        self.state_value = np.sum(self.policy * self.qsa_value, axis=1)
        return self.state_value, reward_history, length_history

def plot_training_stats(reward_history, length_history):
    """
    绘制训练统计图：总奖励随回合的变化 & 回合步数随回合的变化
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # 1. 绘制总奖励图 (Total Rewards)
    ax1.plot(reward_history, color='#1f77b4', linewidth=1) # 使用标准蓝色
    ax1.set_ylabel('Total rewards', fontsize=12)
    ax1.grid(True, linestyle=':', alpha=0.6)
    # 限制 y 轴范围，防止极小的惩罚值拉低整体观感
    # ax1.set_ylim([min(reward_history)-5, 5]) 

    # 2. 绘制回合步数图 (Episode Length)
    ax2.plot(length_history, color='#1f77b4', linewidth=1)
    ax2.set_xlabel('Episode index', fontsize=12)
    ax2.set_ylabel('Episode length', fontsize=12)
    ax2.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()

# --- 运行入口 ---
if __name__ == "__main__":
    # ==========================================
    # 15x15 复杂迷宫地形设计
    # ==========================================
    size = 15
    target = [13, 13]  # 终点在右下角
    
    forbidden = [
        # 1. 左侧竖墙 "|" (故意留了上下两个缺口)
        [3, 1], [3, 2], [3, 3], [3, 4], [3, 5], [3, 6], [3, 7], [3, 8], [3, 9], [3, 10], [3, 11],
        
        # 2. 中间横墙 "-" (迫使Agent往下走)
        [4, 6], [5, 6], [6, 6], [7, 6], [8, 6], [9, 6], [10, 6],
        
        # 3. 中心区域倒 "L" 型陷阱 (死胡同)
        [7, 2], [8, 2], [9, 2], [10, 2],
        [10, 3], [10, 4],
        
        # 4. 右侧竖墙 "|"
        [11, 2], [11, 3], [11, 4], [11, 5], [11, 6], [11, 7], [11, 8],
        
        # 5. 终点附近的 "U" 型包围圈 (只能从上方进入)
        [12, 12], [12, 13], [12, 14], # 左墙
        [13, 14],                     # 底墙
        [14, 12], [14, 13], [14, 14]  # 右墙
    ]

    env = GridEnv(size=size, 
                  target=target, 
                  forbidden=forbidden, 
                  render_mode='')
    
    # 2. 实例化算法
    solver = Q_learning(env)
    
    # 3. 运行 Q-learning 算法
    print("开始训练...")
    final_v, reward_hist, length_hist = solver.Q_learning(max_iter=1000, alpha=0.1)

    # 4. 可视化结果
    print("训练完成，正在渲染...")
    plot_training_stats(reward_hist, length_hist)

    solver.show_policy()
    solver.show_state_value(final_v)
    env.plot_title("Q-learning Results")

    env.render(block=True)
