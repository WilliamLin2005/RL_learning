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
from collections import deque

#MC_EXPLORING STARTS 调参的哲学
class n_step_SARSA:
    def __init__(self, env, n_step=1, alpha=0.1):
        self.gamma = 0.9
        self.env = env
        self.n_step = n_step
        self.alpha = alpha
        self.action_space_size = env.action_space_size
        self.state_space_size = env.size**2
        self.reward_list = env.reward_list
        
        # 初始探索率稍微设高一点，但后续会衰减
        self.epsilon = 0.5 
        self.min_epsilon = 0.01
        self.epsilon_decay = 0.99
        
        self.memory = deque(maxlen=n_step+1)
        
        self.qsa_value = np.zeros(shape=(self.state_space_size, self.action_space_size))
        self.mean_policy = np.ones(shape=(self.state_space_size, self.action_space_size)) / self.action_space_size
        self.policy = self.mean_policy.copy()
        
        self.writer = SummaryWriter("logs")

        print("action_space_size: {} state_space_size：{}".format(self.action_space_size, self.state_space_size))
        print('----------------------------------------------------------------')

    def show_policy(self):
        for state in range(self.state_space_size):
            for action in range(self.action_space_size):
                prob = self.policy[state, action]
                if prob > 0.1: # 过滤掉极小的概率，让画面更清晰
                    self.env.render_.draw_action(pos=self.env.state2pos(state),
                                                 toward=prob * 0.4 * self.env.action_to_direction[action],
                                                 radius=prob * 0.1)

    def show_state_value(self, state_value, y_offset=0.2):
        if isinstance(state_value, np.ndarray) and state_value.ndim == 2:
            state_value = np.max(state_value, axis=1)
        for state in range(self.state_space_size):
            self.env.render_.write_word(pos=self.env.state2pos(state), word=str(round(float(state_value[state]), 1)),
                                        y_offset=y_offset,
                                        size_discount=0.7)

    def obs_to_state(self, obs):
        if isinstance(obs, dict):
            return self.env.pos2state(np.array(obs["agent"]))
        return obs

    def get_action(self, state):
        action_probs = self.policy[state]
        action = np.random.choice(self.action_space_size, p=action_probs)
        return action
    
    def reset(self):
        self.memory.clear()

    def update_policy(self, state):
        best_action = np.argmax(self.qsa_value[state, :])
        # 均匀分配基础概率
        self.policy[state, :] = self.epsilon / self.action_space_size
        # 最优动作加上额外的贪婪概率
        self.policy[state, best_action] += (1.0 - self.epsilon)
    
    def update(self, state, action, reward, done):
        self.memory.append((state, action, reward, done))
        if len(self.memory) < self.n_step + 1:
            return
        
        update_state, update_action, _, _ = self.memory[0]
        td_target = 0.0
        
        # 累加前 n 步的 reward
        for i in range(self.n_step):
            _, _, reward, done = self.memory[i]
            td_target += (self.gamma ** i) * reward
            if done: # 如果中间遇到了 done，提前终止累加
                break
                
        # 加上第 n 步的 Q 值
        _, _, _, last_done = self.memory[self.n_step - 1]
        if not last_done:
            next_state, next_action, _, _ = self.memory[self.n_step]
            td_target += (self.gamma ** self.n_step) * self.qsa_value[next_state, next_action]

        # SARSA Q值更新
        td_error=self.qsa_value[update_state, update_action] - td_target
        self.qsa_value[update_state, update_action] -= self.alpha * td_error
        self.update_policy(update_state)

    def flush_memory(self):
        """回合结束时，把队列里剩下的状态全部更新完"""
        while len(self.memory) > 1:
            update_state, update_action, _, _ = self.memory[0]
            td_target = 0.0
            steps_left = len(self.memory) - 1
            for i in range(steps_left):
                _, _, reward, done = self.memory[i]
                td_target += (self.gamma ** i) * reward
                if done:
                    break
            #这里与上面 update 函数的区别在于，flush_memory 是在回合结束时调用的，此时队列里剩下的状态数可能不足 n_step，
            #所以我们要动态计算剩余步数 steps_left 来正确累加 reward 和 Q 值
            #并且此时不需要next_state 的 Q 值了，因为回合已经结束了（last_state的value为0），所以直接进行更新即可
            td_error = self.qsa_value[update_state, update_action] - td_target
            self.qsa_value[update_state, update_action] -= self.alpha * td_error
            self.update_policy(update_state)
            self.memory.popleft()

    def n_step_SARSA(self, max_iter=500):
        reward_history = []
        length_history = []
        
        for episode in tqdm(range(max_iter)):
            obs, _ = self.env.reset()
            state = self.obs_to_state(obs)
            self.reset() # 回合开始时清空记忆队列！
            
            total_reward = 0
            step_count = 0
            done = False
            
            while True:
                action = self.get_action(state)
                next_obs, reward, done, _, _ = self.env.step(action)
                next_state = self.obs_to_state(next_obs)
                
                total_reward += reward
                step_count += 1
                
                self.update(state, action, reward, done)
                
                if done:
                    self.flush_memory() # 处理回合末尾的 N 步更新
                    break
                    
                state = next_state
                
            reward_history.append(total_reward)
            length_history.append(step_count)
            
            # Epsilon 衰减，让模型从“探索”逐渐转为“利用”
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
        
        return self.qsa_value, reward_history, length_history
        

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
    # 1. 初始化环境 (网格世界)
    env = GridEnv(size=5, 
                  target=[2, 3],
                  forbidden=[[1, 1], [2, 1], [2, 2], [1, 3], [3, 3], [1, 4]],
                  render_mode='')
    
    # 2. 实例化算法
    solver = n_step_SARSA(env, n_step=4, alpha=0.1) 
    
    # 3. 运行 SARSA 算法
    print("开始训练...")
    final_v, reward_hist, length_hist = solver.n_step_SARSA(max_iter=1000)

    # 4. 可视化结果
    print("训练完成，正在渲染...")
    plot_training_stats(reward_hist, length_hist)

    solver.show_policy()
    solver.show_state_value(final_v)
    env.plot_title("N-Step SARSA Results")

    env.render(block=True)

