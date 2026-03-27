import numpy as np

class Visualizer:
    """
    可视化工具类
    职责：
    1. 格式化打印状态价值函数 V (Matrix 形式)
    2. 将策略 pi 转换为箭头符号打印
    """
    @staticmethod
    def print_value_function(v, nrow, ncol):
        """打印状态价值函数 V"""
        print("--- 状态价值函数 V(s) ---")
        for i in range(nrow):
            for j in range(ncol):
                # 打印每个格子，保留两位小数，宽度为 8
                print(f"{v[i * ncol + j]:8.2f}", end=" ")
            print() # 换行

    @staticmethod
    def print_policy(pi, nrow, ncol, holes=None, ends=None):
        """
        打印策略箭头图
        pi: 策略列表 (包含动作索引)
        holes: 坑的索引集合 (可选)
        ends: 终点的索引集合 (可选)
        """
        # 动作映射表
        action_mapping = {0: "←", 1: "↓", 2: "→", 3: "↑"}
        
        print("\n--- 最优策略 箭头图 (pi) ---")
        for i in range(nrow):
            for j in range(ncol):
                curr_s = i * ncol + j
                
                # 1. 逻辑处理：如果是终点或坑，打印特殊字符，不打印动作
                if ends and curr_s in ends:
                    print(f"{'终':^3}", end=" ")
                elif holes and curr_s in holes:
                    print(f"{'坑':^3}", end=" ")
                else:
                    # 2. 正常打印动作箭头
                    a = pi[curr_s]
                    print(f"{action_mapping[a]:^4}", end=" ")
            print()