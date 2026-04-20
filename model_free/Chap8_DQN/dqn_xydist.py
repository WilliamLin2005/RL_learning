import argparse
import os
import sys
import random
import unittest
from collections import deque

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# 1. 使用 abspath(__file__) 获取当前文件的绝对路径
# 2. 向上返回 4 层，到达 python_rl_learn 这一级
current_path = os.path.abspath(__file__)
for _ in range(4):
    current_path = os.path.dirname(current_path)

root_path = current_path

# 3. 把这个根目录加入搜索路径的最前面
if root_path not in sys.path:
    sys.path.insert(0, root_path)

from RL_learning.model_free.envs.grid_env import GridEnv

# 改动：网络输入 - 将 one-hot 改为 [x_norm, y_norm, dist]，位置：line 121，原因：降低输入维度并显式引入空间坐标与到目标的几何信息。


class ReplayBuffer:
    def __init__(self, buffer_size, batch_size):
        self.buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size

    def add(self, state, action, reward, next_state, done):
        data = (state, action, reward, next_state, done)
        self.buffer.append(data)

    def __len__(self):
        return len(self.buffer)

    def get_batch(self):
        data = random.sample(self.buffer, self.batch_size)
        state = torch.tensor([x[0] for x in data], dtype=torch.long)
        action = torch.tensor([x[1] for x in data], dtype=torch.long)
        reward = torch.tensor([x[2] for x in data], dtype=torch.float32)
        next_state = torch.tensor([x[3] for x in data], dtype=torch.long)
        done = torch.tensor([x[4] for x in data], dtype=torch.int32)
        return state, action, reward, next_state, done


class QNet(nn.Module):
    def __init__(self, input_size, action_size):
        super().__init__()
        # 输入维度改为 3，对应 [x_norm, y_norm, dist_to_target]
        self.function = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_size),
        )

    def forward(self, x):
        return self.function(x)


class DQNAgent:
    def __init__(self, env, log_dir="logs"):
        self.gamma = 0.98
        self.learning_rate = 0.05
        self.epsilon = 0.99
        self.buffer_size = 10000
        self.batch_size = 64
        self.env = env
        self.action_space_size = env.action_space_size
        self.state_space_size = env.size ** 2
        self.reward_list = env.reward_list

        # DQN优化控制参数
        self.update_every_n_steps = 20
        self.step_counter = 0

        # 动态读取网格宽高，用于 min-max 归一化
        self.grid_width = getattr(env, "grid_width", env.size)
        self.grid_height = getattr(env, "grid_height", env.size)

        # 显式给出 min-max 参数（按需求）
        self.x_min, self.x_max = 0, self.grid_width - 1
        self.y_min, self.y_max = 0, self.grid_height - 1

        # DQN核心组件
        self.replay_buffer = ReplayBuffer(self.buffer_size, self.batch_size)
        self.qnet = QNet(3, self.action_space_size)
        self.qnet_target = QNet(3, self.action_space_size)
        self.qnet_target.load_state_dict(self.qnet.state_dict())
        self.optimizer = optim.Adam(self.qnet.parameters(), lr=self.learning_rate)

        # 以下仅为了绘图，实际训练中不使用
        self.qsa_value = np.zeros(shape=(self.state_space_size, self.action_space_size))

        # 平均策略初始化
        self.mean_policy = np.ones(shape=(self.state_space_size, self.action_space_size)) / self.action_space_size
        self.policy = self.mean_policy.copy()

        self.writer = SummaryWriter(log_dir)

        print("action_space_size: {} state_space_size：{}".format(self.action_space_size, self.state_space_size))
        print("----------------------------------------------------------------")

    def _normalize_axis(self, value, v_min, v_max):
        if v_max == v_min:
            return 0.0
        return float((value - v_min) / (v_max - v_min))

    def _state_to_feature(self, state):
        pos = self.env.state2pos(int(state))
        x_idx, y_idx = int(pos[0]), int(pos[1])

        # 显式写出 min-max 归一化公式
        x_norm = self._normalize_axis(x_idx, self.x_min, self.x_max)
        y_norm = self._normalize_axis(y_idx, self.y_min, self.y_max)

        target = self.env.target_location
        dist = float(np.linalg.norm(np.array([x_idx, y_idx]) - target))
        return np.array([x_norm, y_norm, dist], dtype=np.float32)

    def _states_to_features(self, states):
        if isinstance(states, torch.Tensor):
            state_list = states.detach().cpu().numpy().tolist()
        else:
            state_list = list(states)
        features = [self._state_to_feature(s) for s in state_list]
        return torch.tensor(features, dtype=torch.float32)

    def show_policy(self):
        for state in range(self.state_space_size):
            for action in range(self.action_space_size):
                prob = self.policy[state, action]
                if prob > 0:
                    self.env.render_.draw_action(
                        pos=self.env.state2pos(state),
                        toward=prob * 0.4 * self.env.action_to_direction[action],
                        radius=prob * 0.1,
                    )

    def show_state_value(self, state_value, y_offset=0.2):
        for state in range(self.state_space_size):
            self.env.render_.write_word(
                pos=self.env.state2pos(state),
                word=str(round(state_value[state], 1)),
                y_offset=y_offset,
                size_discount=0.7,
            )

    # dqn 核心训练部分
    def get_action(self, state):
        # epsilon-greedy: 以 epsilon 概率随机探索，否则按当前 Q 网络贪心选动作
        if np.random.random() < self.epsilon:
            return int(np.random.choice(self.action_space_size))

        state_feature = torch.tensor(self._state_to_feature(state), dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            q_values = self.qnet(state_feature)
        return int(q_values.argmax().item())

    def update(self, state, action, reward, next_state, done):
        self.replay_buffer.add(state, action, reward, next_state, done)

        # 使用计数器控制更新频率
        self.step_counter += 1
        if self.step_counter % self.update_every_n_steps != 0:
            return

        if len(self.replay_buffer) < self.batch_size:
            return

        state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.replay_buffer.get_batch()
        state_features = self._states_to_features(state_batch)
        next_state_features = self._states_to_features(next_state_batch)

        qs = self.qnet(state_features)
        qsa = qs[np.arange(len(action_batch)), action_batch]

        next_qs = self.qnet_target(next_state_features)
        next_qsa_max = next_qs.max(1)[0].detach()

        target = reward_batch + (1 - done_batch) * self.gamma * next_qsa_max

        loss = nn.MSELoss()(qsa, target)
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
            self.writer.add_scalar("Train/Total_Reward", total_reward, episode)
            self.writer.add_scalar("Train/Episode_Length", step_count, episode)
            self.writer.add_scalar("Train/Epsilon", self.epsilon, episode)

        # 训练结束，计算 state value 和 策略用于可视化
        self.state_value = np.zeros(self.state_space_size)
        for s in range(self.state_space_size):
            s_feature = torch.tensor(self._state_to_feature(s), dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                q_values = self.qnet(s_feature).numpy()[0]
            self.qsa_value[s] = q_values
            best_a = np.argmax(q_values)
            self.policy[s, :] = 0
            self.policy[s, best_a] = 1.0
            self.policy[self.env.pos2state(self.env.target_location), :] = 0
            self.policy[self.env.pos2state(self.env.target_location), 4] = 1.0
            self.state_value[s] = np.max(q_values)

        self.writer.close()
        return self.state_value, reward_history, length_history


def plot_training_stats(reward_history, length_history):
    _, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    ax1.plot(reward_history, color="#1f77b4", linewidth=1)
    ax1.set_ylabel("Total rewards", fontsize=12)
    ax1.grid(True, linestyle=":", alpha=0.6)

    ax2.plot(length_history, color="#1f77b4", linewidth=1)
    ax2.set_xlabel("Episode index", fontsize=12)
    ax2.set_ylabel("Episode length", fontsize=12)
    ax2.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    plt.show()



def main():
    parser = argparse.ArgumentParser(description="DQN with mean-policy sampling and [x_norm, y_norm, dist] inputs.")
    parser.add_argument("--run-tests", action="store_true", help="Run unit tests.")
    parser.add_argument("--max-iter", type=int, default=10000, help="Training episodes.")
    parser.add_argument("--sync-interval", type=int, default=10, help="Target sync interval.")
    args = parser.parse_args()

    env = GridEnv(
        size=5,
        target=[2, 3],
        forbidden=[[1, 1], [2, 1], [2, 2], [1, 3], [3, 3], [1, 4]],
        render_mode="",
    )
    solver = DQNAgent(env)
    print("开始训练...")
    final_v, reward_hist, length_hist = solver.train(max_iter=args.max_iter, sync_interval=args.sync_interval)

    print("训练完成，正在渲染...")
    plot_training_stats(reward_hist, length_hist)
    solver.show_policy()
    solver.show_state_value(final_v)
    env.plot_title("DQN Results (MeanPolicy + XYDist)")
    env.render(block=True)


if __name__ == "__main__":
    main()
