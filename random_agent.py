import numpy as np
from gridworld_env import GridworldEnv

env = GridworldEnv()
episodes = 50

success = 0
violations = []

for _ in range(episodes):
    env.reset()
    done = False

    while not done:
        action = env.action_space.sample()
        _, _, done, info = env.step(action)

    if env.agent_pos == env.goal_pos:
        success += 1

    violations.append(info["safety_violations"])

print("Random Policy Results")
print("Success Rate:", success / episodes * 100)
print("Avg Safety Violations:", np.mean(violations))
