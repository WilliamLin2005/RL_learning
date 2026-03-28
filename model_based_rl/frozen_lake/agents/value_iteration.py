from .base_agent import BaseAgent
import numpy as np

class ValueIterationAgent(BaseAgent):

    def policy_update(self):
        for s in range(self.n_states):
            qsa_list = []
            for a in range(self.n_actions):
                qsa = 0.0
                for p, next_state, reward, done in self.P[s][a]:
                    qsa += p * (reward + self.gamma * self.v[next_state] * (not done))
                qsa_list.append(qsa)
            self.pi[s] = int(np.argmax(qsa_list))
    def value_update(self):
        new_v = np.zeros(self.n_states, dtype=float)
        for s in range(self.n_states):
            a = int(self.pi[s])
            v_s = 0.0
            for p, next_state, reward, done in self.P[s][a]:
                v_s += p * (reward + self.gamma * self.v[next_state] * (not done))
            new_v[s] = v_s
        self.v = new_v
    
    def train(self,theta=1e-6):
        self.theta = theta
        while True:
            self.policy_update()
            old_v = self.v.copy()
            self.value_update()
            max_diff = np.max(np.abs(self.v - old_v))
            if max_diff < self.theta:
                break

    def get_action(self, state):
        return int(self.pi[state])
