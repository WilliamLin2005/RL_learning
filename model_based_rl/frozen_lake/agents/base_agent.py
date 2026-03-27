from abc import ABC, abstractmethod
import numpy as np

# 继承 ABC，说明这是一个抽象基类
class BaseAgent(ABC):
    def __init__(self, env, gamma=0.9, theta=1e-6, truncated_step=None):
        """
        构造函数：初始化所有强化学习算法通用的“成员变量”
        """
        # 1. 引用环境的动力学模型 P
        # C++ 类比：相当于存了一个指向环境数据的指针/引用
        self.P = env.P            
        
        # 2. 地图维度信息
        self.nrow = env.nrow      
        self.ncol = env.ncol      
        self.n_states = self.nrow * self.ncol
        self.n_actions = 4        
        
        # 3. 超参数：折扣因子
        self.gamma = gamma
        self.theta = theta
        self.truncated_step = truncated_step
        
        # 4. 核心数据结构：状态价值函数 V
        # C++ 类比：std::vector<double> v(n_states, 0.0);
        self.v = np.zeros(self.n_states)
        
        # 5. 核心数据结构：策略 pi
        # C++ 类比：std::vector<int> pi(n_states, 0);
        # 使用 dtype=int 是为了后面直接作为索引访问 action
        self.pi = np.zeros(self.n_states, dtype=int)

    @abstractmethod
    def train(self, theta=1e-4):
        """
        这是一个纯虚函数。
        为什么？因为 Value Iteration 和 Policy Iteration 的训练逻辑完全不同。
        具体的逻辑留给子类去 override（重写）。
        """
        pass

    def get_action(self, state):
        """
        这是一个普通成员函数（非虚函数）。
        因为无论什么算法，一旦训练完成，根据当前状态 s 查表找动作 a 的逻辑是一样的。
        这就是“代码复用”。
        """
        return self.pi[state]