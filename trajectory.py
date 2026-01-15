import matplotlib.pyplot as plt
import numpy as np
import torch
from gridworld_env import GridworldEnv
from dqn_agent import DQNAgent

env = GridworldEnv()
state_size = env.observation_space.n
action_size = env.action_space.n

agent = DQNAgent(state_size, action_size)
agent.model.load_state_dict(torch.load("dqn_model.pth"))
agent.model.eval()

state = env.reset()
state_vec = np.eye(state_size)[state]

trajectory = [env.agent_pos]

done = False
while not done:
    state_tensor = torch.tensor(state_vec, dtype=torch.float32)
    with torch.no_grad():
        action = torch.argmax(agent.model(state_tensor)).item()

    next_state, _, done, _ = env.step(action)
    trajectory.append(env.agent_pos)
    state_vec = np.eye(state_size)[next_state]

# Plot trajectory
grid = np.zeros((env.grid_size, env.grid_size))
for pos in trajectory:
    grid[pos] = 1

plt.figure(figsize=(6,6))
plt.imshow(grid, cmap="Blues")
plt.title("Agent Trajectory Visualization")
plt.xticks([])
plt.yticks([])
plt.savefig("trajectory.png", dpi=300)
plt.show()
