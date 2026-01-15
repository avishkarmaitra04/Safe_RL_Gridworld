import numpy as np
import gym
from gym import spaces

class GridworldEnv(gym.Env):

    def __init__(self, grid_size=6):
        self.grid_size = grid_size
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Discrete(grid_size * grid_size)

        self.start_pos = (0, 0)
        self.goal_pos = (grid_size - 1, grid_size - 1)
        self.hazards = [
    (1,1), (1,2), (1,3),
    (2,1), (3,1),
    (3,3), (4,3),
    (2,4), (3,4)
]

                # Create grid representation
        self.grid = np.zeros((grid_size, grid_size), dtype=int)

        for h in self.hazards:
            self.grid[h] = 1   # Hazard cells

        self.grid[self.goal_pos] = 2  # Goal cell


        self.max_steps = 30
        self.reset()

    def reset(self):
        self.agent_pos = self.start_pos
        self.steps = 0
        self.safety_violations = 0
        return self._get_state()

    def _get_state(self):
        row, col = self.agent_pos
        return row * self.grid_size + col

    def step(self, action):
        row, col = self.agent_pos
        # Add action noise
        if np.random.rand() < 0.15:
            action = self.action_space.sample()
        if action == 0:
            row -= 1
        elif action == 1:
            row += 1
        elif action == 2:
            col -= 1
        elif action == 3:
            col += 1

        row = np.clip(row, 0, self.grid_size - 1)
        col = np.clip(col, 0, self.grid_size - 1)
        self.agent_pos = (row, col)

        reward = -1
        done = False
        self.steps += 1

        if self.agent_pos in self.hazards:
            reward = -10
            self.safety_violations += 1

        if self.agent_pos == self.goal_pos:
            reward = 20
            done = True

        if self.steps >= self.max_steps:
            done = True

        info = {"safety_violations": self.safety_violations}

        return self._get_state(), reward, done, info
