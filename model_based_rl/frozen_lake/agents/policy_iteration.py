from .base_agent import BaseAgent
import copy
import numpy as np

class PolicyIterationAgent(BaseAgent):
    def __init__(self, env, truncated_step=None, gamma=0.9, theta=1e-6):
        super().__init__(env, gamma=gamma, theta=theta, truncated_step=truncated_step)

    def policy_evaluation(self):
        if self.truncated_step is None:
            while True:
                new_v = np.zeros(self.n_states, dtype=float)
                for s in range(self.n_states):
                    a = int(self.pi[s])
                    v_s = 0.0
                    for p, next_state, reward, done in self.P[s][a]:
                        v_s += p * (reward + self.gamma * self.v[next_state] * (not done))
                    new_v[s] = v_s
                max_diff = float(np.max(np.abs(new_v - self.v)))
                self.v = new_v
                if max_diff < self.theta:
                    break
        else:
            for _ in range(self.truncated_step):
                new_v = np.zeros(self.n_states, dtype=float)
                for s in range(self.n_states):
                    a = int(self.pi[s])
                    v_s = 0.0
                    for p, next_state, reward, done in self.P[s][a]:
                        v_s += p * (reward + self.gamma * self.v[next_state] * (not done))
                    new_v[s] = v_s
                self.v = new_v

    def policy_improvement(self):
        policy_stable = True
        for s in range(self.n_states):
            old_a = int(self.pi[s])
            qsa_list = []
            for a in range(self.n_actions):
                qsa = 0.0
                for p, next_state, reward, done in self.P[s][a]:
                    qsa += p * (reward + self.gamma * self.v[next_state] * (not done))
                qsa_list.append(qsa)
            best_a = int(np.argmax(qsa_list))
            self.pi[s] = best_a
            if best_a != old_a:
                policy_stable = False
        return policy_stable

    def train(self, theta=None):
        if theta is not None:
            self.theta = theta
        while True:
            self.policy_evaluation()
            old_pi = copy.deepcopy(self.pi)
            policy_stable = self.policy_improvement()
            if policy_stable or np.array_equal(old_pi, self.pi):
                break
