import numpy as np
from gridworld_env import GridworldEnv

env = GridworldEnv()

state_size = env.observation_space.n
action_size = env.action_space.n

Q = np.zeros((state_size, action_size))

alpha = 0.1
gamma = 0.95
epsilon = 1.0
epsilon_decay = 0.995
episodes = 500

for ep in range(episodes):
    state = env.reset()
    done = False

    while not done:
        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[state])

        next_state, reward, done, info = env.step(action)

        Q[state, action] += alpha * (
            reward + gamma * np.max(Q[next_state]) - Q[state, action]
        )

        state = next_state

    epsilon *= epsilon_decay

print("Q-learning training done")
np.save("q_table.npy", Q)
 