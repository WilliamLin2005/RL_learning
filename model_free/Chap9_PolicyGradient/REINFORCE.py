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
import torch.nn.functional as F
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


class PolicyNet(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=128):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)
    
class REINFORCE():
    def __init__(self, env, log_dir="logs/REINFORCE"):
        self.gamma = 0.98
        self.learning_rate = 0.005
        self.entropy_coef = 0.001
        self.baseline_decay = 0.9
        self.return_baseline = 0.0
        self.baseline_initialized = False
        self.env = env
        self.action_space_size = env.action_space_size
        self.state_space_size = env.size ** 2
        self.reward_list = env.reward_list

        # 动态读取网格宽高，用于 min-max 归一化
        self.grid_width = getattr(env, "grid_width", env.size)
        self.grid_height = getattr(env, "grid_height", env.size)

        # 显式给出 min-max 参数（按需求）
        self.x_min, self.x_max = 0, self.grid_width - 1
        self.y_min, self.y_max = 0, self.grid_height - 1
        self.max_dist = float(np.linalg.norm(np.array([self.x_max - self.x_min, self.y_max - self.y_min])))

        self.policy_net = PolicyNet(3, self.action_space_size)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)

        # 以下仅为了绘图，实际训练中不使用
        self.qsa_value = np.zeros(shape=(self.state_space_size, self.action_space_size))
        self.qsa_count = np.zeros(shape=(self.state_space_size, self.action_space_size))
        self.state_value = np.zeros(shape=self.state_space_size)

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
        dist_norm = dist / self.max_dist if self.max_dist > 0 else 0.0
        return np.array([x_norm, y_norm, dist_norm], dtype=np.float32)

    def _states_to_features(self, states):
        if isinstance(states, torch.Tensor):
            state_list = states.detach().cpu().numpy().tolist()
        else:
            state_list = list(states)
        features = [self._state_to_feature(s) for s in state_list]
        return torch.tensor(np.array(features, dtype=np.float32), dtype=torch.float32)

    def show_policy(self, greedy=True, min_prob=0.05):
        for state in range(self.state_space_size):
            if greedy:
                action = int(np.argmax(self.policy[state]))
                prob = self.policy[state, action]
                self.env.render_.draw_action(
                    pos=self.env.state2pos(state),
                    toward=0.4 * self.env.action_to_direction[action],
                    radius=max(prob * 0.1, 0.04),
                )
                continue

            for action, prob in enumerate(self.policy[state]):
                if prob < min_prob:
                    continue
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

    def get_action(self, state):
        state_tensor = torch.tensor(self._state_to_feature(state), dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            logits = self.policy_net(state_tensor)
            action_dist = torch.distributions.Categorical(logits=logits)
            action = action_dist.sample().item()
        return action

    def reset_episode(self, random_start=True):
        obs, _ = self.env.reset()
        state = self.env.pos2state(obs["agent"])

        if random_start:
            target_state = self.env.pos2state(self.env.target_location)
            candidate_states = [s for s in range(self.state_space_size) if s != target_state]
            state = int(np.random.choice(candidate_states))
            self.env.agent_location = self.env.state2pos(state).copy()

        return state
    
    def update_policy(self, states, actions, rewards):
        if len(states) == 0:
            return 0.0

        # Value update: q_t = r_{t+1} + gamma * r_{t+2} + ...
        returns = []
        discounted_return = 0.0
        for reward in reversed(rewards):
            discounted_return = reward + self.gamma * discounted_return
            returns.insert(0, discounted_return)

        # 记录每个 (s, a) 的 Monte Carlo return 均值，仅用于后续可视化。
        for state, action, q_value in zip(states, actions, returns):
            self.qsa_count[state, action] += 1
            count = self.qsa_count[state, action]
            self.qsa_value[state, action] += (q_value - self.qsa_value[state, action]) / count

        state_features = self._states_to_features(states)
        action_tensor = torch.tensor(actions, dtype=torch.long)
        return_tensor = torch.tensor(returns, dtype=torch.float32)

        logits = self.policy_net(state_features)
        log_probs = F.log_softmax(logits, dim=1)
        selected_log_probs = log_probs.gather(1, action_tensor.unsqueeze(1)).squeeze(1)

        baseline = self.return_baseline
        if not self.baseline_initialized:
            baseline = return_tensor.mean().item()
            self.return_baseline = baseline
            self.baseline_initialized = True

        advantage = return_tensor - baseline
        if len(returns) > 1:
            advantage = advantage / (advantage.std(unbiased=False) + 1e-8)

        self.return_baseline = (
            self.baseline_decay * self.return_baseline
            + (1.0 - self.baseline_decay) * return_tensor.mean().item()
        )

        # Policy update: theta <- theta + alpha * grad(log pi(a|s, theta)) * q_t
        policy_loss = -(selected_log_probs * advantage.detach()).mean()
        entropy = torch.distributions.Categorical(logits=logits).entropy().mean()
        loss = policy_loss - self.entropy_coef * entropy

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        return loss.item()

    def train(self, max_iter=10000, max_steps=200, random_start=True):
        reward_history = []
        length_history = []

        for episode in tqdm(range(max_iter)):
            state = self.reset_episode(random_start=random_start)

            states = []
            actions = []
            rewards = []
            total_reward = 0
            step_count = 0
            done = False

            while not done and step_count < max_steps:
                action = self.get_action(state)
                next_obs, reward, terminated, truncated, _ = self.env.step(action)
                next_state = self.env.pos2state(next_obs["agent"])

                states.append(state)
                actions.append(action)
                rewards.append(reward)

                total_reward += reward
                step_count += 1
                state = next_state
                done = terminated or truncated

            loss = self.update_policy(states, actions, rewards)

            reward_history.append(total_reward)
            length_history.append(step_count)

            self.writer.add_scalar("Train/Total_Reward", total_reward, episode)
            self.writer.add_scalar("Train/Episode_Length", step_count, episode)
            self.writer.add_scalar("Train/Policy_Loss", loss, episode)

        # 训练结束后，把神经网络策略导出到 self.policy，方便 show_policy() 画图。
        for state in range(self.state_space_size):
            state_tensor = torch.tensor(self._state_to_feature(state), dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                action_probs = torch.softmax(self.policy_net(state_tensor), dim=1).squeeze(0).numpy()
            self.policy[state, :] = action_probs

        target_state = self.env.pos2state(self.env.target_location)
        self.policy[target_state, :] = 0
        self.policy[target_state, 4] = 1.0
        self.state_value = self.evaluate_current_policy()

        self.writer.close()
        return self.state_value, reward_history, length_history

    def evaluate_current_policy(self):
        reward_sa = np.tensordot(self.env.Rsa, np.array(self.reward_list), axes=([2], [0]))
        p_policy = np.einsum("sa,san->sn", self.policy, self.env.Psa)
        r_policy = np.sum(self.policy * reward_sa, axis=1)

        target_state = self.env.pos2state(self.env.target_location)
        p_policy[target_state, :] = 0
        r_policy[target_state] = 0

        identity = np.eye(self.state_space_size)
        return np.linalg.solve(identity - self.gamma * p_policy, r_policy)

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
    parser = argparse.ArgumentParser(description="REINFORCE Policy Gradient Example")
    parser.add_argument("--run-tests", action="store_true", help="Run unit tests.")
    parser.add_argument("--max-iter", type=int, default=5000, help="Training episodes.")
    parser.add_argument("--fixed-start", action="store_true", help="Train only from the environment reset state.")
    args = parser.parse_args()

    env = GridEnv(
        size=5,
        target=[2, 3],
        forbidden=[[1, 1], [2, 1], [2, 2], [1, 3], [3, 3], [1, 4]],
        render_mode="",
    )
    solver = REINFORCE(env)
    print("开始训练...")
    final_v, reward_hist, length_hist = solver.train(max_iter=args.max_iter, random_start=not args.fixed_start)

    print("训练完成，正在渲染...")
    plot_training_stats(reward_hist, length_hist)
    solver.show_policy()
    solver.show_state_value(final_v)
    env.plot_title("REINFORCE Results (XYDist)")
    env.render(block=True)


if __name__ == "__main__":
    main()
