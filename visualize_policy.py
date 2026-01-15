import numpy as np
import matplotlib.pyplot as plt
from gridworld_env import GridworldEnv

env = GridworldEnv()
policy = np.zeros((env.grid_size, env.grid_size))

# Simple heuristic policy (for visualization)
for r in range(env.grid_size):
    for c in range(env.grid_size):
        if (r, c) in env.hazards:
            policy[r, c] = -1   # Hazard
        elif (r, c) == env.goal_pos:
            policy[r, c] = 1    # Goal
        else:
            policy[r, c] = 0    # Safe

plt.figure(figsize=(6, 6))
plt.imshow(policy, cmap="coolwarm")
plt.title("Policy Visualization (Safe Regions vs Hazards)")
plt.xticks([])
plt.yticks([])
plt.savefig("policy_visual.png", dpi=300)
plt.show()
