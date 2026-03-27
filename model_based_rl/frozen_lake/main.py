import time
import numpy as np
# 从创建的包中导入模块
from envs.frozen_lake_env import LakeEnv
from agents.policy_iteration import PolicyIterationAgent
from agents.value_iteration import ValueIterationAgent
from utils.visualizer import Visualizer

def get_holes_ends(env):
    holes = set()
    ends = set()
    desc = getattr(env, "desc", None)
    if desc is None:
        return holes, ends
    for i in range(env.nrow):
        for j in range(env.ncol):
            c = desc[i][j]
            if isinstance(c, bytes):
                c = c.decode("utf-8")
            idx = i * env.ncol + j
            if c == "H":
                holes.add(idx)
            elif c == "G":
                ends.add(idx)
    return holes, ends

def run_task(agent_name, agent, theta, holes, ends):
    """
    通用执行函数：负责训练和结果打印
    """
    print(f"\n" + "="*20)
    print(f"正在启动算法: {agent_name}")
    print("="*20)
    
    # 1. 算法训练 (纯数学计算，在内存中完成)
    start_time = time.perf_counter()
    agent.train(theta=theta)
    end_time = time.perf_counter()
    print(f"训练耗时: {end_time - start_time:.4f} 秒")

    # 2. 结果可视化 (调用 utils 中的工具)
    Visualizer.print_value_function(agent.v, agent.nrow, agent.ncol)
    Visualizer.print_policy(agent.pi, agent.nrow, agent.ncol, holes=holes, ends=ends)

def evaluate_agent(agent, is_slippery, map_name, episodes, max_steps):
    env = LakeEnv(is_slippery=is_slippery, render_mode=None, map_name=map_name)
    success = 0
    for _ in range(episodes):
        state, _ = env.reset()
        done = False
        step_count = 0
        while not done and step_count < max_steps:
            action = agent.get_action(state)
            state, reward, term, trunc, _ = env.step(action)
            done = term or trunc
            step_count += 1
            if reward == 1:
                success += 1
                break
    env.close()
    return success / episodes if episodes > 0 else 0.0

def main():
    # --- 全局配置 ---
    GAMMA = 0.9
    THETA = 1e-6
    MAP_NAME = "8x8"
    IS_SLIPPERY = False
    EVAL_EPISODES = 200
    EVAL_MAX_STEPS = 200
    DEMO_MAX_STEPS = 200
    DEMO_EPISODES = 20

    # ======================================================
    # 阶段 1: 训练阶段 (Training Phase)
    # 使用 render_mode=None
    # ======================================================
    print("初始化训练环境...")
    train_env = LakeEnv(is_slippery=IS_SLIPPERY, render_mode=None, map_name=MAP_NAME)
    holes, ends = get_holes_ends(train_env)

    # 实例化三个不同的 Agent 
    # 1. 价值迭代
    vi_agent = ValueIterationAgent(train_env, gamma=GAMMA, theta=THETA)
    # 2. 标准策略迭代 (truncated_step=None)
    pi_agent = PolicyIterationAgent(train_env, gamma=GAMMA, truncated_step=None, theta=THETA)
    # 3. 截断策略迭代 (这里设定截断步数为 10)
    tpi_agent = PolicyIterationAgent(train_env, gamma=GAMMA, truncated_step=10, theta=THETA)

    # 分别执行训练
    run_task("Value Iteration", vi_agent, THETA, holes, ends)
    run_task("Standard Policy Iteration", pi_agent, THETA, holes, ends)
    run_task("Truncated Policy Iteration (k=10)", tpi_agent, THETA, holes, ends)

    train_env.close()

    # ======================================================
    # 阶段 2: 演示阶段 (Demonstration Phase)
    # 使用 render_mode="human"，开启 GUI 窗口观察训练成果
    # ======================================================
    success_rate = evaluate_agent(
        vi_agent,
        is_slippery=IS_SLIPPERY,
        map_name=MAP_NAME,
        episodes=EVAL_EPISODES,
        max_steps=EVAL_MAX_STEPS,
    )
    print(f"\n评估结果：episodes={EVAL_EPISODES}, 成功率={success_rate:.2%}")

    print("\n准备开始可视化演示...")
    demo_env = LakeEnv(is_slippery=IS_SLIPPERY, render_mode="human", map_name=MAP_NAME)
    demo_success = False
    total_reward = 0
    step_count = 0

    for _ in range(DEMO_EPISODES):
        state, _ = demo_env.reset()
        done = False
        total_reward = 0
        step_count = 0
        visited = set()
        while not done and step_count < DEMO_MAX_STEPS:
            demo_env.render()
            if state in visited:
                break
            visited.add(state)
            action = vi_agent.get_action(state)
            state, reward, term, trunc, _ = demo_env.step(action)
            total_reward += reward
            step_count += 1
            time.sleep(0.5)
            done = term or trunc
            if reward == 1:
                demo_success = True
                break
        if demo_success:
            break

    print(f"\n演示结束！总步数: {step_count}, 累计奖励: {total_reward}, 成功: {demo_success}")
    demo_env.close()

# 类似于 C++ 的 int main() 入口保护
if __name__ == "__main__":
    main()
