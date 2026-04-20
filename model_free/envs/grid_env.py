import time
from typing import Optional, Union, List, Tuple

import gymnasium as gym
# import gym
import numpy as np
from gymnasium import spaces
# from gym import spaces
from gymnasium.core import RenderFrame, ActType, ObsType
# from gym.core import RenderFrame, ActType, ObsType

#随机数生成器将产生相同的随机数序列, 这在需要可重复结果的情况下非常有用
np.random.seed(1)
from . import render


def arr_in_list(array, _list):
    for element in _list:
        if np.array_equal(element, array):
            return True
    return False


'''
grid_world坐标系：
行 * 列
-------> +x
|
|
v  +y
'''
class GridEnv(gym.Env):

    def __init__(self, size: int, target: Union[list, tuple, np.ndarray], forbidden: Union[list, tuple, np.ndarray],
                 render_mode: str, reward_list: Optional[List[float]] = None):
        """
        GridEnv 的构造函数
        :param size: grid_world 的边长
        :param target: 目标点的pos
        :param forbidden: 不可通行区域 二维数组 或者嵌套列表 如 [[1,2],[2,2]]
        :param render_mode: 渲染模式 video表示保存视频
        :param reward_list: 奖励列表，索引顺序固定为 [other, target, forbidden, overflow]
        """
        # 初始化可视化
        self.agent_location = np.array([0, 0])
        self.time_steps = 0
        self.size = size
        self.render_mode = render_mode
        self.render_ = render.Render(target=target, forbidden=forbidden, size=size)
        # 初始化起点 障碍物 目标点
        self.forbidden_location = []
        for fob in forbidden:
            self.forbidden_location.append(np.array(fob))
        self.target_location = np.array(target)
        # 初始化 动作空间 观测空间
        self.action_space, self.action_space_size = spaces.Discrete(5,seed = 1), spaces.Discrete(5).n  #动作空间：0-up 1-right 2-down 3-left 4-stay
        print("self.action_space:{}, self.action_space_size:{}".format(self.action_space, self.action_space_size))  #从0开始索引

        # reward_list index convention (used by Rsa):
        # reward_list[other, target, forbidden, overflow]
        #
        # Examples:
        # reward_list[other, target, forbidden, overflow]
        # - [0, 1, -10, -10]  # strong penalty for forbidden/overflow (default) , reward list chapter8
        # - [0, 1, -1, -10]   # small forbidden penalty, strong overflow penalty
        # - [-1, 0, -10, -10] # chapter7: step cost -1, target 0
        # - [0, 1, -1, -1]    # reward list for TD linear: small penalty for forbidden/overflow,
        default_reward_list = [-1, 10, -10, -10] # reward list chapter8
        reward_list = default_reward_list if reward_list is None else reward_list
        if len(reward_list) != 4:
            raise ValueError("reward_list must have length 4: [other, target, forbidden, overflow]")
        self.reward_list = list(reward_list)
        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(low = 0, high = size - 1, shape=(2,), dtype=int),
                "target": spaces.Box(low = 0, high = size - 1, shape=(2,), dtype=int),
                "barrier": spaces.Box(low = 0, high = size - 1, shape=(2,), dtype=int),
            }
        )
        # action to pos偏移量 的一个map
        #坐标系  ------>    x > 0
        #       |
        #       | y >0
        #       v

        self.action_to_direction = {
            0: np.array([0, -1]), #up
            1: np.array([1, 0]),  #right
            2: np.array([0, 1]),  #down
            3: np.array([-1, 0]), #left
            4: np.array([0, 0]),  #stay
        }
        # Rsa表示 在 指定 state 选取指定 action 得到Immediate reward的概率
        self.Rsa = None
        # Psa表示 在 指定 state 选取指定 action 跳到下一个state的概率
        self.Psa = None
        self.psa_rsa_init()

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None, ) -> Tuple[ObsType, dict]:
        super().reset(seed=seed)
        self.agent_location = np.array([0, 0])
        observation = self.get_obs()
        info = self.get_info()
        return observation, info

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, bool, dict]:  #  -> 是函数的返回类型注解，

        reward = self.reward_list[  self.Rsa[self.pos2state(self.agent_location), action].tolist().index(1)  ]
        direction = self.action_to_direction[action]
        self.render_.upgrade_agent(self.agent_location, direction, self.agent_location + direction)
        self.agent_location = np.clip(self.agent_location + direction, 0, self.size - 1)
        terminated = np.array_equal(self.agent_location, self.target_location)
        observation = self.get_obs()
        info = self.get_info()
        return observation, reward, terminated, False, info

    def render(self, t: float = 0.3, block: bool = False) -> Optional[Union[RenderFrame, List[RenderFrame]]]:
        if self.render_mode == "video":
            self.render_.save_video('image/' + str(time.time()))

        self.render_.show_frame(t=t, block=block)
        return None
    def render_clear(self):
        self.render_.close_frame()
        return None

    def plot_title(self,title = "title"):
        self.render_.plot_title(title)

    def get_obs(self) -> ObsType:
        return {"agent": self.agent_location, "target": self.target_location, "barrier": self.forbidden_location}

    def get_info(self) -> dict:
        return {"time_steps": self.time_steps}


    def state2pos(self, state: int) -> np.ndarray:
        """
        用于将状态（state）转换为位置（pos）。这在一些环境中是很常见的.
        比如在一个二维的格子世界中，我们可能会将每个格子看作一个状态，然后用一个整数来表示这个状态。而这个函数就是用来将这个整数转换回对应的格子位置。
        :param state: state number
        :return: 二维列表，表示agent的位置：x列 y行
        """
        return np.array((state % self.size,state // self.size))

    def pos2state(self, pos: np.ndarray) -> int:
        """
        假设 self.size 是 5，那么位置 (1, 2) 对应的状态就是 1 + 2*5 = 11
        :param pos:位置数组[1, 2]  pos[0]： 列， pos[1] 行。
        :return: state number
        """
        return pos[1] * self.size + pos[0]

    def psa_rsa_init(self):
        
        state_size = self.size ** 2
        self.Psa = np.zeros(shape=(state_size, self.action_space_size, state_size), dtype=float)
        self.Rsa = np.zeros(shape=(state_size, self.action_space_size, len(self.reward_list)), dtype=float)
        # 填充Psa、Rsa矩阵
        for state_index in range(state_size):
            for action_index in range(self.action_space_size):
                pos = self.state2pos(state_index) # 二维列表
                next_pos = pos + self.action_to_direction[action_index] # action_index：0~4，将动作映射给pos，得到next_pos

                if next_pos[0] < 0 or next_pos[1] < 0 or next_pos[0] > self.size - 1 or next_pos[1] > self.size - 1:  #如果“撞墙”了，超出地图边界
                    self.Psa[state_index, action_index, state_index] = 1
                    self.Rsa[state_index, action_index, 3] = 1
                else:
                    self.Psa[state_index, action_index, self.pos2state(next_pos)] = 1
                    if np.array_equal(next_pos, self.target_location): #如果到达target area
                        self.Rsa[state_index, action_index, 1] = 1
                    elif arr_in_list(next_pos, self.forbidden_location): #如果进入forbidden_area
                        self.Rsa[state_index, action_index, 2] = 1
                    else: #other
                        self.Rsa[state_index, action_index, 0] = 1
                    #reward_list[other, target, forbidden, overflow]
        #print("self.Psa:{}\n self.Rsa:{}".format(self.Psa,self.Rsa))
    def close(self):
        pass


if __name__ == "__main__":
    # grid = GridEnv(size=5, target=[1, 3], forbidden=[[2, 2],[2,0],[4,2],[3,4]], render_mode='')
    # grid = GridEnv(size=5, target=[0, 1], forbidden=[[2, 2],[2,0],[4,2],[3,4]], render_mode='')
    grid = GridEnv(size=5, target=[2, 3],
                                  forbidden=[[1, 1], [2, 1], [2, 2], [1, 3], [3, 3], [1, 4]],
                                  render_mode='')
    grid.render(block=True)