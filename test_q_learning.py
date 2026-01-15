import numpy as np
from gridworld_env import GridworldEnv

Q = np.load("q_table.npy")

env = GridworldEnv()
episodes = 50

success = 0
violations = []

for _ in range(episodes):
    state = env.reset()
    done = False

    while not done:
        action = np.argmax(Q[state])
        state, _, done, info = env.step(action)

    if env.agent_pos == env.goal_pos:
        success += 1

    violations.append(info["safety_violations"])

print("Q-learning Results")
print("Success Rate:", success / episodes * 100)
print("Average Safety Violations:", np.mean(violations))
