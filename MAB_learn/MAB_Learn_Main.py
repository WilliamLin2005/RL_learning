import numpy as np
import matplotlib.pyplot as plt
from bandit import Bandit
from solver import Epsilon_Greedy_Solver


K=10
bandit_10_arm=Bandit(K)
print("随机生成了一个%d臂老虎机"%(bandit_10_arm.K))
print("获奖概率最大的的拉杆%d号,其获奖概率为:%.5f"%(bandit_10_arm.best_idx,bandit_10_arm.best_prob))
epsilons = [1e-4, 0.01, 0.1, 0.25, 0.5]
epsilon_greedy_solver_list = [
    Epsilon_Greedy_Solver(bandit_10_arm, epsilon=e) for e in epsilons
]
epsilon_greedy_solver_names = ["epsilon={}".format(e) for e in epsilons]
for solver in epsilon_greedy_solver_list:
    solver.run(50000)

plt.figure(figsize=(12, 8))
for solver, name in zip(epsilon_greedy_solver_list, epsilon_greedy_solver_names):
    plt.plot(solver.regrets, label=name)
plt.xlabel("Steps")
plt.ylabel("Cumulative Regret")
plt.title("Epsilon-Greedy Solver Regret Curves")
plt.legend()
plt.show()

        

        
