#这个文件负责环境的初始化。我们在这里处理 unwrapped（解包）逻辑，并提取地图的行列信息。

import gymnasium as gym

class LakeEnv:
    """
    FrozenLake 环境的包装类。
    职责：
    1. 初始化指定模式的环境(human 或 None)。
    2. 提供底层动力学 P 矩阵。
    3. 提供地图的尺寸 (nrow, ncol)。
    """
    def __init__(self, is_slippery=True, render_mode=None, map_name="4x4"):
        # 1. 创建原始环境
        # map_name 可以是 "4x4" 或 "8x8"
        self.gym_env = gym.make(
            "FrozenLake-v1", 
            map_name=map_name, 
            is_slippery=is_slippery, 
            render_mode=render_mode
        )
        
        # 2. 核心操作：Unwrap
        # 类似于 C++ 中的底层指针转换，获取没有被 Wrapper 包装的原始环境对象
        # 只有这样才能访问到 env.P (状态转移矩阵)
        self.env = self.gym_env.unwrapped
        
        # 3. 提取维度信息 (4x4 环境下 nrow=4, ncol=4)
        self.nrow = self.env.nrow
        self.ncol = self.env.ncol
        
        # 4. 暴露状态转移概率 P 给算法使用
        # 结构：P[state][action] = [(prob, next_state, reward, done), ...]
        self.P = self.env.P
        self.desc = getattr(self.env, "desc", None)

    def reset(self):
        """重置环境，返回初始状态"""
        return self.gym_env.reset()

    def step(self, action):
        """执行动作，返回 (next_state, reward, terminated, truncated, info)"""
        return self.gym_env.step(action)

    def render(self):
        """渲染画面"""
        return self.gym_env.render()

    def close(self):
        """释放资源，关闭窗口"""
        self.gym_env.close()
