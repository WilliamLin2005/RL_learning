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

#MC_EXPLORING STARTS 调参的哲学
class SARSA:
    def __init__(self, env):
        self.gamma = 0.9
        self.env = env
        self.action_space_size = env.action_space_size
        self.state_space_size = env.size**2
        self.reward_list = env.reward_list
        self.epsilon=0.9
        
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
                                                 toward=prob * 0.4 * self.env.action_to_direction[action],
                                                 radius=prob * 0.1)

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
            
            # 3. 按当前 policy 的概率去“采样”下一个动作，绝对不能用 np.max!
            next_action = np.random.choice(self.action_space_size, p=self.policy[new_state, :]) 
            
            experience = {
                'state_t': start_state,
                'action_t': start_action,
                'reward': reward,
                'state_tp1': new_state,
                'action_tp1': next_action,
                'done': done
            }
            return experience

    def SARSA(self,max_iter=1000,alpha=0.5):
        for episode in tqdm(range(max_iter)):
            self.epsilon = max(0.01, self.epsilon * 0.95)  # 每个 episode 后衰减 epsilon，最低到 0.01
            # 1. 初始化环境，获取初始状态
            obs,_ = self.env.reset()
            state = self.env.pos2state(obs["agent"])
            # 2. 根据当前策略选择一个动作
            action = np.random.choice(self.action_space_size, p=self.policy[state, :])
            done = False
            td_error = 0
            iteration = 0
            while not done:
                experience = self.obtain_experience(state, action)
                state, action, reward, next_state, next_action, done = experience['state_t'], experience['action_t'], experience['reward'], experience['state_tp1'], experience['action_tp1'], experience['done']
                    
                # 2. 更新 Q 表
                td_target = reward + self.gamma * self.qsa_value[next_state, next_action] * (1 - done)
                td_error =  self.qsa_value[state, action]- td_target
                self.qsa_value[state, action] -= alpha * td_error
                    
                # 3. 更新策略(episilon-greedy)
                best_action = np.argmax(self.qsa_value[state, :])
                self.policy[state, :] = np.double(self.epsilon/5)
                self.policy[state, best_action] = np.double(1-(4*self.epsilon)/5)

                #4.更新状态和动作
                state = next_state
                action = next_action

                iteration += 1
                if iteration>1500:
                    print("迭代过多，可能进入死循环了，强制结束")
                    break
            
        #根据qsa计算state value
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
    solver = SARSA(env)
    
    # 3. 运行 SARSA 算法
    print("开始训练...")
    final_v = solver.SARSA(max_iter=20000, alpha=0.1)

    # 4. 可视化结果
    print("训练完成，正在渲染...")
    solver.show_policy()
    solver.show_state_value(final_v)
    env.plot_title("SARSA Results")
    env.render(block=True)

