import copy

class CliffWalkingEnv:
    #配置悬崖漫步基础环境
    def __init__(self,number_col=12,number_row=4):
        self.column=number_col
        self.row=number_row
        self.P=self.createP()
    #初始化
    def createP(self):
        #初始化一个大小为 [总状态数][总动作数][动态长度] 的三维数组，用来存储环境的动态转移规则。
        P=[[[]for j in range(5)]for i in range(self.row*self.column)]
        #定义5种动作:上,下,左,右,不动,需要注意的是,这里是坐标系是(0,1,2....,11)x(0,1,2,3)
        action=[[0,-1],[0,1],[-1,0],[1,0],[0,0]]
        for i in range(self.row):
            for j in range(self.column):
                for a in range(5):
                    #若处于悬崖处,任何动作奖励为0,采取任何动作都保持在原地,并且结束
                    if i == self.row-1  and 0<j<self.column-1:
                        #左边:当 Agent 站在第 i 行、第 j 列的格子上，并且它打算执行动作 a 时，环境的反馈
                        #右边:环境会返回一个四元组 (p, next_state, reward, done)，
                        #其中 p 是转移概率，next_state 是下一个状态的索引，reward 是奖励值，done 是一个布尔值，表示是否达到了终止状态。
                        P[i*self.column+j][a]=[(1,i*self.column+j,0,True)]
                        continue
                    elif i == self.row-1 and j==self.column-1:
                        P[i*self.column+j][a]=[(1,i*self.column+j,0,True)]
                        continue
                    
                    #其他位置
                    #计算下一步的位置，如果撞墙了就留在原地
                    next_x=min(self.column-1,max(0,j+action[a][0]))
                    next_y=min(self.row-1,max(0,i+action[a][1]))
                    #把计算出来的二维物理坐标 (next_y, next_x) 重新映射回状态索引
                    next_state=next_y*self.column+next_x
                    reward=-1
                    done=False
                    #如果下一步是悬崖,奖励为-100,并且结束,如果下一步是目标,奖励为1
                    if next_y==self.row-1 and next_x>0:
                        if next_x<self.column-1:
                            reward=-100
                            done=True
                        else:
                            reward=1
                            done=True
                    #将转移规则添加到环境的动态转移规则 P 中
                    P[i*self.column+j][a]=[(1,next_state,reward,done)]
        return P
    
class ValueIteration:
    def __init__(self,CliffWalkingEnv:CliffWalkingEnv,theta,gamma):
        self.env=CliffWalkingEnv
        self.value=[0]*self.env.column*self.env.row
        #theta 是一个小的正数，表示迭代的收敛阈值。当状态值函数的更新变化小于 theta 时，我们认为算法已经收敛，可以停止迭代。
        self.theta=theta
        #gamma 是折扣因子，取值范围在 0 和 1 之间。它决定了未来奖励在当前决策中的重要程度。
        #较高的 gamma 值表示未来奖励更重要，而较低的 gamma 值表示当前奖励更重要。
        self.gamma=gamma
        #创建一个包含 48 个元素的列表，表示初始策略，全部设为 None
        self.pi=[None for i in range(self.env.column*self.env.row)]
    
    def value_iteration(self):
        count=0
        #在每次迭代中，我们计算每个状态 s 的新值 new_value[s]，并跟踪最大差异 max_diff
        while 1:
            max_diff=0
            new_value=[0]*self.env.column*self.env.row
            for s in range(self.env.column*self.env.row):
                #计算该状态s下的q(s|a)
                qsa_list=[]
                for a in range(5):
                    qsa=0
                    for p,next_state,reward,done in self.env.P[s][a]:
                        #根据环境的转移规则，计算动作 a 在状态 s 下的动作价值 q(s, a)。
                        #具体来说，对于每个可能的转移 (p, next_state, reward, done)，
                        #我们将其贡献加到 qsa 中。贡献的计算方式是
                        #转移概率 p 乘以 (奖励 reward 加上折扣因子 gamma 乘以下一个状态的值 value[next_state]，如果 done 是 False 的话)。
                        #如果 done 是 True，那么下一个状态的值不应该被考虑，因为 episode 已经结束了。
                        qsa+=p*(reward+self.gamma*self.value[next_state]*(not done))
                    qsa_list.append(qsa)
                #更新状态 s 的值，并计算与之前值的差异，更新 max_diff
                new_value[s]=max(qsa_list)
                max_diff=max(max_diff,abs(new_value[s]-self.value[s]))
            self.value=new_value
            if max_diff<self.theta:
                break
            count+=1
        print("迭代次数:",count)
        self.get_policy()

    def get_policy(self):
        for s in range(self.env.column*self.env.row):
            qsa_list=[]
            for a in range(5):
                qsa=0
                for p,next_state,reward,done in self.env.P[s][a]:
                    qsa+=p*(reward+self.gamma*self.value[next_state]*(not done))
                qsa_list.append(qsa)
            #找到动作价值最大的动作索引
            best_action=qsa_list.index(max(qsa_list))
            self.pi[s]=best_action

env=CliffWalkingEnv()
value_iteration=ValueIteration(env,theta=1e-4,gamma=0.9)
value_iteration.value_iteration()
print("\n最优策略 pi(s) (修正显示逻辑):")
action_mapping = {0: "↑", 1: "↓", 2: "←", 3: "→", 4: "·"} 

for i in range(env.row):
    for j in range(env.column):
        curr_s = i * env.column + j
        # 检查是否是悬崖 (1~10列的最后一行)
        if i == env.row - 1 and 0 < j < env.column - 1:
            print(f"{'崖':^3}", end=" ")
        # 检查是否是目标点 (最后一行最后一列)
        elif i == env.row - 1 and j == env.column - 1:
            print(f"{'终':^3}", end=" ")
        else:
            action_index = value_iteration.pi[curr_s]
            print(f"{action_mapping[action_index]:^4}", end=" ")
    print()