import os
import sys
import numpy as np

# 2. 向上返回 3 层，到达 python_rl_learn 这一级
current_path = os.path.abspath(__file__)
for _ in range(3):
    current_path = os.path.dirname(current_path)
root_path = current_path
if root_path not in sys.path:
    sys.path.insert(0, root_path)

from RL_learning.model_free.envs.grid_env import GridEnv
from RL_learning.model_free.envs.grid_env_stochastic import GridEnvStochastic

class MC_EG_Tester:
    def __init__(self, env, init_outside=True):
        self.gamma = 0.9
        self.env = env
        self.action_space_size = env.action_space_size
        self.state_space_size = env.size**2
        self.reward_list = env.reward_list
        self.epsilon = 0.8
        self.init_outside = init_outside
        
        # 初始化状态价值和 Q 表
        self.state_value = np.zeros(shape=self.state_space_size)
        self.qsa_value = np.zeros(shape=(self.state_space_size, self.action_space_size))
        
        # 平均策略初始化
        self.policy = np.ones(shape=(self.state_space_size, self.action_space_size)) / self.action_space_size

    def obtain_episode(self, policy, start_state, start_action, max_step):
        self.env.agent_location = self.env.state2pos(start_state)
        episode = []
        state = start_state
        action = start_action
        
        while max_step > 0:
            max_step -= 1
            _, reward, done, _, _ = self.env.step(action)
            next_state = self.env.pos2state(self.env.agent_location)
            next_action = np.random.choice(np.arange(self.action_space_size), p=policy[next_state])
            episode.append({"state": state, "action": action, "reward": reward})
            state = next_state
            action = next_action
            if done:
                break
        return episode

    def train(self, max_step=30, iteration=500):
        if self.init_outside:
            sa_pair_count = np.zeros(shape=(self.state_space_size, self.action_space_size), dtype=int)
            return_of_sa_pair = np.zeros(shape=(self.state_space_size, self.action_space_size), dtype=np.double)
        
        for _ in range(iteration):
            if not self.init_outside:
                sa_pair_count = np.zeros(shape=(self.state_space_size, self.action_space_size), dtype=int)
                return_of_sa_pair = np.zeros(shape=(self.state_space_size, self.action_space_size), dtype=np.double)
            
            self.epsilon = max(0.1, self.epsilon * 0.999) # Slow decay
            s0 = np.random.randint(0, self.state_space_size)
            a0 = np.random.choice(np.arange(self.action_space_size))
            episode = self.obtain_episode(self.policy, s0, a0, max_step)
            G = 0
            for t in reversed(range(len(episode))):
                state = episode[t]['state']
                action = episode[t]['action']
                reward = episode[t]['reward']
                G = self.gamma * G + reward
                
                sa_pair_count[state, action] += 1
                return_of_sa_pair[state, action] += G
                self.qsa_value[state, action] = return_of_sa_pair[state, action] / sa_pair_count[state, action]
                
                best_action = np.argmax(self.qsa_value[state, :])
                self.policy[state, :] = np.double(self.epsilon / self.action_space_size)
                self.policy[state, best_action] = np.double(1 - ( (self.action_space_size-1) * self.epsilon) / self.action_space_size)
        
        self.state_value = np.max(self.qsa_value, axis=1)
        return self.state_value

    def evaluate(self, num_episodes=100, max_step=30):
        total_rewards = []
        for _ in range(num_episodes):
            s = self.env.reset()[0]
            # Use greedy policy for evaluation
            state = self.env.pos2state(self.env.agent_location)
            episode_reward = 0
            for _ in range(max_step):
                action = np.argmax(self.policy[state, :])
                _, reward, done, _, _ = self.env.step(action)
                episode_reward += reward
                state = self.env.pos2state(self.env.agent_location)
                if done:
                    break
            total_rewards.append(episode_reward)
        return np.mean(total_rewards)

def run_experiment():
    size = 5
    target = [2, 3]
    forbidden = [[1, 1], [2, 1], [2, 2], [1, 3], [3, 3], [1, 4]]
    
    # 增加迭代次数以更好地观察差异
    iterations = 10000
    eval_episodes = 50
    max_step = 100

    results = {}

    for env_name, env_class in [("Deterministic", GridEnv), ("Stochastic", GridEnvStochastic)]:
        for init_pos in ["Outside", "Inside"]:
            print(f"Running: {env_name} Env, Init {init_pos}")
            env = env_class(size=size, target=target, forbidden=forbidden, render_mode='')
            tester = MC_EG_Tester(env, init_outside=(init_pos == "Outside"))
            tester.train(max_step=max_step, iteration=iterations)
            score = tester.evaluate(num_episodes=eval_episodes, max_step=max_step)
            results[f"{env_name}_{init_pos}"] = score
            print(f"Result: {score}")

    print("\n--- Summary ---")
    for key, val in results.items():
        print(f"{key}: {val}")

if __name__ == "__main__":
    run_experiment()
