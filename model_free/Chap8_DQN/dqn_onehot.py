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
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import copy
from collections import deque

class ReplayBuffer:
    def __init__(self,buffer_size,batch_size):
        self.buffer=deque(maxlen=buffer_size)
        self.batch_size=batch_size

    def add(self,state,action,reward,next_state,done):
        data=(state,action,reward,next_state,done)
        self.buffer.append(data)

    def __len__(self):
        return len(self.buffer)
    
    def get_batch(self):
        data=random.sample(self.buffer,self.batch_size)

        state=torch.tensor([x[0] for x in data], dtype=torch.long)
        action=torch.tensor([x[1] for x in data], dtype=torch.long)
        reward=torch.tensor([x[2] for x in data], dtype=torch.float32)
        next_state=torch.tensor([x[3] for x in data], dtype=torch.long)
        done=torch.tensor([x[4] for x in data], dtype=torch.int32)
        return state,action,reward,next_state,done

class QNet(nn.Module):
    def __init__(self,input_size,action_size):
        super().__init__()
        self.function = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128,action_size)
        )

    def forward(self, x):
        return self.function(x)

class DQNAgent:
    def __init__(self,env):
        self.gamma=0.98
        self.learning_rate=0.05
        self.buffer_size=10000
        self.batch_size=64
        self.env = env
        self.action_space_size = env.action_space_size
        self.state_space_size = env.size**2
        self.reward_list = env.reward_list
        self.epsilon=0.9
        
        # DQN优化控制参数
        self.update_every_n_steps = 20
        self.step_counter = 0

        #DQN核心组件
        self.replay_buffer=ReplayBuffer(self.buffer_size,self.batch_size)
        self.qnet=QNet(self.state_space_size,self.action_space_size)
        self.qnet_target=QNet(self.state_space_size,self.action_space_size)
        self.qnet_target.load_state_dict(self.qnet.state_dict())
        self.optimizer=optim.Adam(self.qnet.parameters(),lr=self.learning_rate)

        #以下仅为了绘图，实际训练中不使用
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
                                                 toward=prob * 0.4 * self.env.action_to_direction[action],
                                                 radius=prob * 0.1)

    def show_state_value(self, state_value, y_offset=0.2):
        for state in range(self.state_space_size):
            self.env.render_.write_word(pos=self.env.state2pos(state), word=str(round(state_value[state], 1)),
                                        y_offset=y_offset,
                                        size_discount=0.7)
            
    
    #dqn 核心训练部分
    def get_action(self,state):
        if np.random.random()<self.epsilon:
            return np.random.choice(self.action_space_size,p=self.mean_policy[state, :])
        else:
            # 使用 PyTorch 原生 one_hot 转换，增加 unsqueeze(0) 以保证符合 [batch_size, input_size]
            state_tensor = torch.tensor(state, dtype=torch.long)
            state_one_hot = F.one_hot(state_tensor, num_classes=self.state_space_size).float().unsqueeze(0)
            
            with torch.no_grad():
                q_values=self.qnet(state_one_hot)
            action=q_values.argmax().item()
            return action
        
    def update(self,state,action,reward,next_state,done):
        self.replay_buffer.add(state,action,reward,next_state,done)

        # 使用计数器控制更新频率
        self.step_counter += 1
        if self.step_counter % self.update_every_n_steps != 0:
            return

        if len(self.replay_buffer)<self.batch_size:
            return
        
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.replay_buffer.get_batch()
        
        # 使用 PyTorch 原生 one_hot 批量转换
        state_one_hot = F.one_hot(state_batch, num_classes=self.state_space_size).float()
        next_state_one_hot = F.one_hot(next_state_batch, num_classes=self.state_space_size).float()
        
        qs=self.qnet(state_one_hot)
        qsa=qs[np.arange(len(action_batch)), action_batch]
        
        next_qs=self.qnet_target(next_state_one_hot)
        next_qsa_max=next_qs.max(1)[0]
        # 必须重新赋值或者直接使用 .detach() 的返回值来阻断梯度
        next_qsa_max = next_qsa_max.detach()

        target=reward_batch+(1-done_batch)*self.gamma*next_qsa_max

        loss=nn.MSELoss()(qsa,target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def sync_qnet(self):
        self.qnet_target.load_state_dict(self.qnet.state_dict())

    def train(self, max_iter=10000, sync_interval=20):
        reward_history = []
        length_history = []
        for episode in tqdm(range(max_iter)):
            self.epsilon = max(0.01, 1.0 - episode / (max_iter * 0.8))
            obs, _ = self.env.reset()
            state = self.env.pos2state(obs["agent"])
            done = False
            total_reward = 0
            step_count = 0
            
            while not done:
                action = self.get_action(state)
                obs, reward, terminated, _, _ = self.env.step(action)
                next_state = self.env.pos2state(obs["agent"])
                done = terminated
                
                # 调用 update，在 update 内部处理经验添加和按频率反向传播
                self.update(state, action, reward, next_state, done)

                state = next_state
                total_reward += reward
                step_count += 1
                
                if step_count > 200:
                    break
                    
            if episode % sync_interval == 0:
                self.sync_qnet()
                
            reward_history.append(total_reward)
            length_history.append(step_count)
            
            # TensorBoard logging
            self.writer.add_scalar('Train/Total_Reward', total_reward, episode)
            self.writer.add_scalar('Train/Episode_Length', step_count, episode)
            self.writer.add_scalar('Train/Epsilon', self.epsilon, episode)
            
        # 训练结束，计算 state value 和 策略用于可视化
        self.state_value = np.zeros(self.state_space_size)
        for s in range(self.state_space_size):
            s_tensor = torch.tensor(s, dtype=torch.long)
            s_one_hot = F.one_hot(s_tensor, num_classes=self.state_space_size).float().unsqueeze(0)
            
            with torch.no_grad():
                q_values = self.qnet(s_one_hot).numpy()[0]
            self.qsa_value[s] = q_values
            best_a = np.argmax(q_values)
            self.policy[s, :] = 0
            self.policy[s, best_a] = 1.0
            #强行让目标处策略为停在原地
            self.policy[self.env.pos2state(self.env.target_location), :] = 0
            self.policy[self.env.pos2state(self.env.target_location), 4] = 1.0
            self.state_value[s] = np.max(q_values)
            
        self.writer.close()
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

    # 2. 绘制回合步数图 (Episode Length)
    ax2.plot(length_history, color='#1f77b4', linewidth=1)
    ax2.set_xlabel('Episode index', fontsize=12)
    ax2.set_ylabel('Episode length', fontsize=12)
    ax2.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    plt.show()

# --- 运行入口 ---
if __name__ == "__main__":
    # 1. 初始化环境 (网格世界)
    env = GridEnv(size=5, 
                  target=[2, 3],
                  forbidden=[[1, 1], [2, 1], [2, 2], [1, 3], [3, 3], [1, 4]],
                  render_mode='')


    
    # 2. 实例化算法
    solver = DQNAgent(env)
    
    # 3. 运行 DQN 算法
    print("开始训练...")
    final_v, reward_hist, length_hist = solver.train(max_iter=20000, sync_interval=10)

    # 4. 可视化结果
    print("训练完成，正在渲染...")
    plot_training_stats(reward_hist, length_hist)

    solver.show_policy()
    solver.show_state_value(final_v)
    env.plot_title("DQN Results")

    env.render(block=True)
