import os
import sys
import numpy as np

# 1. 动态获取根路径并加入 sys.path
current_path = os.path.abspath(__file__)
for _ in range(3):
    current_path = os.path.dirname(current_path)
root_path = current_path
if root_path not in sys.path:
    sys.path.insert(0, root_path)

from RL_learning.model_free.envs.grid_env import GridEnv
from RL_learning.model_free.envs.grid_env_stochastic import GridEnvStochastic

class MC_ES_LargeScale:
    def __init__(self, env, init_outside=True):
        self.gamma = 0.99  # 增大 gamma，因为路径更长了
        self.env = env
        self.action_space_size = env.action_space_size
        self.state_space_size = env.size**2
        self.init_outside = init_outside
        
        self.qsa_value = np.zeros(shape=(self.state_space_size, self.action_space_size))
        self.policy = np.ones(shape=(self.state_space_size, self.action_space_size)) / self.action_space_size

    def obtain_episode(self, policy, start_state, start_action, max_step):
        self.env.agent_location = self.env.state2pos(start_state)
        episode = []
        state = start_state
        action = start_action
        
        for _ in range(max_step):
            _, reward, done, _, _ = self.env.step(action)
            next_state = self.env.pos2state(self.env.agent_location)
            # 策略：完全贪婪
            next_action = np.argmax(policy[next_state])
            episode.append({"state": state, "action": action, "reward": reward})
            state = next_state
            action = next_action
            if done:
                break
        return episode

    def train(self, max_step=150, iteration=10000):
        if self.init_outside:
            sa_pair_count = np.zeros(shape=(self.state_space_size, self.action_space_size), dtype=int)
            # 采用乐观初始值 10.0
            return_of_sa_pair = np.ones(shape=(self.state_space_size, self.action_space_size), dtype=float) * 10.0
        
        for _ in range(iteration):
            if not self.init_outside:
                sa_pair_count = np.zeros(shape=(self.state_space_size, self.action_space_size), dtype=int)
                return_of_sa_pair = np.ones(shape=(self.state_space_size, self.action_space_size), dtype=float) * 10.0
            
            # Exploring Starts
            s0 = np.random.randint(0, self.state_space_size)
            a0 = np.random.randint(0, self.action_space_size)
            episode = self.obtain_episode(self.policy, s0, a0, max_step)
            
            G = 0
            visited = set()
            for t in reversed(range(len(episode))):
                state = episode[t]['state']
                action = episode[t]['action']
                reward = episode[t]['reward']
                G = self.gamma * G + reward
                
                # First-visit MC
                if (state, action) not in visited:
                    sa_pair_count[state, action] += 1
                    return_of_sa_pair[state, action] += G
                    self.qsa_value[state, action] = return_of_sa_pair[state, action] / sa_pair_count[state, action]
                    
                    # 立即更新策略 (Greedy)
                    best_action = np.argmax(self.qsa_value[state, :])
                    self.policy[state, :] = 0
                    self.policy[state, best_action] = 1
                    visited.add((state, action))
        
        return self.qsa_value

    def evaluate(self, num_episodes=20, max_step=150):
        success_count = 0
        for _ in range(num_episodes):
            self.env.reset()
            state = self.env.pos2state(self.env.agent_location)
            for _ in range(max_step):
                action = np.argmax(self.policy[state, :])
                _, _, done, _, _ = self.env.step(action)
                state = self.env.pos2state(self.env.agent_location)
                if done:
                    success_count += 1
                    break
        return success_count / num_episodes

def run_large_experiment():
    size = 8
    target = [6, 6]
    forbidden = [[3, 3], [3, 4], [4, 3], [4, 4]]
    
    iterations = 100000 
    max_step = 100

    print(f"--- 25x25 Large Scale Experiment (625 states) ---")
    
    # 1. 确定性环境
    print("\n[Environment: Deterministic]")
    for init_pos in ["Outside", "Inside"]:
        env = GridEnv(size=size, target=target, forbidden=forbidden, render_mode='')
        tester = MC_ES_LargeScale(env, init_outside=(init_pos == "Outside"))
        tester.train(max_step=max_step, iteration=iterations)
        success_rate = tester.evaluate(num_episodes=50, max_step=max_step)
        print(f"Init {init_pos} -> Success Rate: {success_rate*100}%")

    # 2. 随机环境 (success_prob=0.8)
    print("\n[Environment: Stochastic (0.8)]")
    for init_pos in ["Outside", "Inside"]:
        env = GridEnvStochastic(size=size, target=target, forbidden=forbidden, render_mode='', success_prob=0.8)
        tester = MC_ES_LargeScale(env, init_outside=(init_pos == "Outside"))
        tester.train(max_step=max_step, iteration=iterations)
        success_rate = tester.evaluate(num_episodes=50, max_step=max_step)
        print(f"Init {init_pos} -> Success Rate: {success_rate*100}%")

if __name__ == "__main__":
    run_large_experiment()
