from .base_agent import BaseAgent

class ValueIterationAgent(BaseAgent):
    def train(self, theta=None):
        if theta is None:
            theta = self.theta
        while True:
            max_diff = 0.0
            for s in range(self.n_states):
                qsa_list = []
                for a in range(self.n_actions):
                    qsa = 0.0
                    for p, next_state, reward, done in self.P[s][a]:
                        qsa += p * (reward + self.gamma * self.v[next_state] * (not done))
                    qsa_list.append(qsa)
                new_value = max(qsa_list)
                max_diff = max(max_diff, abs(new_value - self.v[s]))
                self.v[s] = new_value
            if max_diff < theta:
                break
        for s in range(self.n_states):
            qsa_list = []
            for a in range(self.n_actions):
                qsa = 0.0
                for p, next_state, reward, done in self.P[s][a]:
                    qsa += p * (reward + self.gamma * self.v[next_state] * (not done))
                qsa_list.append(qsa)
            self.pi[s] = int(qsa_list.index(max(qsa_list)))

    def get_action(self, state):
        return int(self.pi[state])
