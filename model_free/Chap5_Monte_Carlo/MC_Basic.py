import random
import time
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from RL_learning.model_free.envs.grid_env import GridEnv

class MC_Basic:
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

    def mc_basic(self, max_step=20, epoch=10):
        num_episode = 5
        for ep in range(epoch):
            # 使用 NumPy 数组存储临时的 Q 值估计
            new_qsa = np.zeros((self.state_space_size, self.action_space_size))
            
            # 遍历每一个状态和每一个动作 (MC Basic 的核心：全部遍历)
            for state in tqdm(range(self.state_space_size), desc=f"Epoch {ep+1}/{epoch}"):
                for action in range(self.action_space_size):
                    total_return = 0
                    for _ in range(num_episode):
                        episode = self.obtain_episode(self.policy, state, action, max_step)
                        
                        # 计算这条轨迹的 Return (从后往前或直接正向算均可，因为我们只关心起点的 G)
                        g = 0
                        for i in range(len(episode)):
                            g += (self.gamma**i) * episode[i]['reward']
                        total_return += g
                    
                    # 取平均作为 Q(s,a) 的估计
                    new_qsa[state, action] = total_return / num_episode
            
            # 更新全局 Q 表
            self.qsa_value = new_qsa.copy()

            # Policy Improvement (策略提升：变成贪婪策略)
            for state in range(self.state_space_size):
                best_action = np.argmax(self.qsa_value[state])
                self.policy[state, :] = 0
                self.policy[state, best_action] = 1

            # 更新状态价值（用于可视化）
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
    solver = MC_Basic(env)
    
    # 3. 运行 MC Basic 算法
    print("开始训练...")
    final_v = solver.mc_basic(max_step=30, epoch=5)
    
    # 4. 可视化结果
    print("训练完成，正在渲染...")
    solver.show_policy()
    solver.show_state_value(final_v)
    env.plot_title("MC Basic Results")
    env.render(block=True)