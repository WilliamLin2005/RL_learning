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

#MC_EXPLORING STARTS
class MC_ES:
    def __init__(self, env):
        self.gamma = 0.9
        self.env = env
        self.action_space_size = env.action_space_size
        self.state_space_size = env.size**2
        self.reward_list = env.reward_list
        
        # 初始化状态价值和 Q 表
        self.state_value = np.zeros(shape=self.state_space_size)
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

    def obtain_episode(self, policy, start_state, start_action, max_step):
        # 强行空投 Agent 到指定状态
        self.env.agent_location = self.env.state2pos(start_state)
        episode = []
        
        # 第一步执行指定的 action (Exploring Starts)
        state = start_state
        action = start_action
        
        while max_step > 0:
            max_step -= 1
            _, reward, done, _, _ = self.env.step(action)
            
            # 获取移动后的新状态
            next_state = self.env.pos2state(self.env.agent_location)
            
            # 根据当前策略采样下一个动作
            next_action = np.random.choice(np.arange(self.action_space_size), p=policy[next_state])
            
            episode.append({"state": state, "action": action, "reward": reward})
            
            if done:
                break
            
            # 接力更新状态和动作
            state = next_state
            action = next_action
            
        return episode

    def MC_ES(self, max_step=30, iterations=1000):
            # 计数器放在循环外面，保持跨 episode 积累
            sa_pair_count = np.zeros(shape=(self.state_space_size, self.action_space_size), dtype=int)
            
            for i in tqdm(range(iterations), desc="MC-ES Training"):
                # 1. Exploring Starts: 随机空投起点和第一个动作
                s0 = np.random.randint(0, self.state_space_size)
                a0 = np.random.randint(0, self.action_space_size)
                
                episode = self.obtain_episode(self.policy, s0, a0, max_step)
                
                G = 0
                # 2. 逆向计算 Return (Every-visit 策略)
                for t in reversed(range(len(episode))):
                    s = episode[t]['state']
                    a = episode[t]['action']
                    r = episode[t]['reward']
                    
                    G = self.gamma * G + r
                    
                    # 3. 增量式更新 Q 值
                    sa_pair_count[s, a] += 1
                    # 均值更新：New_Avg = Old_Avg + (New_Return - Old_Avg) / Count
                    self.qsa_value[s, a] += (G - self.qsa_value[s, a]) / sa_pair_count[s, a]
                    
                    # 4. 即时策略提升 (GPI)
                    best_action = np.argmax(self.qsa_value[s, :])
                    self.policy[s, :] = 0
                    self.policy[s, best_action] = 1
                    
            # 训练完更新 V 表用于可视化展示
            self.state_value = np.max(self.qsa_value, axis=1)
            return self.state_value

        

# --- 运行入口 ---
if __name__ == "__main__":
    # 1. 初始化环境 (网格世界)
    env = GridEnv(size=5, 
                  target=[2, 3],
                  forbidden=[[1, 1], [2, 1], [2, 2], [1, 3], [3, 3], [1, 4]],
                  render_mode='')
    
    # 2. 实例化算法
    solver = MC_ES(env)
    
    # 3. 运行 MC ES 算法
    print("开始训练...")
    final_v = solver.MC_ES(max_step=30, iterations=100)
    
    # 4. 可视化结果
    print("训练完成，正在渲染...")
    solver.show_policy()
    solver.show_state_value(final_v)
    env.plot_title("MC ES Results")
    env.render(block=True)