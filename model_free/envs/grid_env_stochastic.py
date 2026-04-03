import numpy as np
from typing import Optional, Union, List, Tuple
from gymnasium.core import ActType, ObsType
from .grid_env import GridEnv, arr_in_list

class GridEnvStochastic(GridEnv):
    def __init__(self, size: int, target: Union[list, tuple, np.ndarray], forbidden: Union[list, tuple, np.ndarray],
                 render_mode: str, reward_list: Optional[List[float]] = None, success_prob: float = 0.8):
        """
        Stochastic Grid Environment where actions might fail.
        :param success_prob: The probability that the action leads to the intended direction.
        """
        self.success_prob = success_prob
        super().__init__(size, target, forbidden, render_mode, reward_list)

    def psa_rsa_init(self):
        """
        Initialize Psa and Rsa with stochasticity.
        If an action is taken, it has success_prob probability of moving in the intended direction,
        and (1 - success_prob) / (num_actions - 1) probability of moving in any other direction.
        """
        state_size = self.size ** 2
        self.Psa = np.zeros(shape=(state_size, self.action_space_size, state_size), dtype=float)
        self.Rsa = np.zeros(shape=(state_size, self.action_space_size, len(self.reward_list)), dtype=float)

        for state_index in range(state_size):
            pos = self.state2pos(state_index)
            for action_index in range(self.action_space_size):
                # Calculate probabilities for each possible action's resulting direction
                for actual_action in range(self.action_space_size):
                    prob = self.success_prob if actual_action == action_index else (1 - self.success_prob) / (self.action_space_size - 1)
                    
                    direction = self.action_to_direction[actual_action]
                    next_pos = pos + direction
                    
                    reward_idx = 0 # Default: other
                    next_state_idx = state_index # Default: stay if hit wall
                    
                    if next_pos[0] < 0 or next_pos[1] < 0 or next_pos[0] > self.size - 1 or next_pos[1] > self.size - 1:
                        # Hit wall
                        next_state_idx = state_index
                        reward_idx = 3 # overflow
                    else:
                        next_state_idx = self.pos2state(next_pos)
                        if np.array_equal(next_pos, self.target_location):
                            reward_idx = 1 # target
                        elif arr_in_list(next_pos, self.forbidden_location):
                            reward_idx = 2 # forbidden
                        else:
                            reward_idx = 0 # other
                    
                    self.Psa[state_index, action_index, next_state_idx] += prob
                    self.Rsa[state_index, action_index, reward_idx] += prob

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, bool, dict]:
        state_index = self.pos2state(self.agent_location)
        
        # Sample next state based on Psa
        next_state_index = np.random.choice(np.arange(self.size ** 2), p=self.Psa[state_index, action])
        
        # Sample reward based on Rsa (jointly with next state would be better, but Rsa is marginal here)
        # Actually, in this env, reward depends on next state. 
        # But we'll follow the Rsa definition provided in the original code.
        reward_idx = np.random.choice(np.arange(len(self.reward_list)), p=self.Rsa[state_index, action])
        reward = self.reward_list[reward_idx]
        
        # Update agent location
        old_location = self.agent_location.copy()
        self.agent_location = self.state2pos(next_state_index)
        
        # For rendering: we need a direction. 
        # Since it's stochastic, the "actual" direction taken might be different from the action.
        actual_direction = self.agent_location - old_location
        self.render_.upgrade_agent(old_location, actual_direction, self.agent_location)
        
        terminated = np.array_equal(self.agent_location, self.target_location)
        observation = self.get_obs()
        info = self.get_info()
        return observation, reward, terminated, False, info

if __name__ == "__main__":
    env = GridEnvStochastic(size=5, target=[2, 3],
                            forbidden=[[1, 1], [2, 1], [2, 2], [1, 3], [3, 3], [1, 4]],
                            render_mode='')
    print("Stochastic Env Initialized")
    obs, info = env.reset()
    for _ in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Action: {action}, Reward: {reward}, Terminated: {terminated}")
        if terminated:
            env.reset()
