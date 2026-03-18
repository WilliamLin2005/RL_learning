import numpy as np
import matplotlib.pyplot as plt
from bandit import Bandit 

class Solver:
    def __init__(self,Bandit:Bandit):
        self.Bandit=Bandit
        self.counts=np.zeros(self.Bandit.K) #每根拉杆尝试次数
        self.regret=0 #当前步的累计遗憾
        self.actions=[] #维护一个列表,记录每一步的动作
        self.regrets=[] #维护一个列表,记录每一步的累积懊悔
    
    def upgrade_regret(self,k):
        self.regret+=self.Bandit.best_prob-self.Bandit.probs[k]
        self.regrets.append(self.regret)

    def run_one_step(self):
        raise NotImplementedError
    
    def run(self,num_steps):
        for _ in range(num_steps):
            k=self.run_one_step()
            self.counts[k]+=1
            self.actions.append(k)
            self.upgrade_regret(k)

class Epsilon_Greedy_Solver(Solver):
    def __init__(self,Bandit:Bandit,epsilon=0.01,init_prob=1.0):
        super().__init__(Bandit)
        self.epsilon=epsilon
        self.estimates=np.array([init_prob]*self.Bandit.K)#初始化拉动所有K根拉杆的期望奖励估值,初始值为init_prob=1.0

    def run_one_step(self):
        if np.random.random()<self.epsilon:
            k=np.random.randint(0,self.Bandit.K)#随机选择一根拉杆
        else:
            k=self.Bandit.best_idx#选择当前期望奖励估值最大的拉杆
        r=self.Bandit.step(k)
        self.estimates[k]+=1./(self.counts[k]+1)*(r-self.estimates[k])
        #更新第k根拉杆的期望奖励估值,其中r是第k根拉杆的奖励,counts[k]是第k根拉杆被尝试的次数
        #更新公式为:新的估值=旧的估值+步长*(实际奖励-旧的估值)
        return k
    
    def plot_regret(self):
        plt.plot(self.regrets)
        plt.xlabel("Steps")
        plt.ylabel("Cumulative Regret")
        plt.title("Epsilon-Greedy Solver Regret Curve")
        plt.show()
    


