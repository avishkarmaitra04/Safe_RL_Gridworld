from gridworld_env import GridworldEnv
from dqn_agent import DQNAgent
import numpy as np
import matplotlib.pyplot as plt

# Create environment
env = GridworldEnv()

state_size = env.observation_space.n
action_size = env.action_space.n

# Create DQN agent
agent = DQNAgent(state_size=state_size, action_size=action_size)

episodes = 300
batch_size = 32

# Tracking for analysis
episode_rewards = []
total_violations = []

visit_counts = np.zeros((env.grid_size, env.grid_size))

for episode in range(episodes):
    state = env.reset()
    state = np.eye(state_size)[state]   # one-hot encoding

    total_reward = 0
    done = False

    while not done:
        action = agent.act(state)
        next_state, reward, done, info = env.step(action)
        row, col = env.agent_pos
        visit_counts[row, col] += 1

        next_state = np.eye(state_size)[next_state]

        agent.remember(state, action, reward, next_state, done)
        agent.replay(batch_size)

        state = next_state
        total_reward += reward

    # Store episode statistics
    episode_rewards.append(total_reward)
    total_violations.append(info["safety_violations"])

    print(
        f"Episode {episode + 1}/{episodes} | "
        f"Reward: {total_reward:4d} | "
        f"Safety Violations: {info['safety_violations']} | "
        f"Epsilon: {agent.epsilon:.2f}"
    )

print("Training finished successfully!")

# Save data
np.save("rewards.npy", episode_rewards)
np.save("violations.npy", total_violations)
np.save("visit_counts.npy", visit_counts)


# ---------------- PLOTTING ---------------- #

# Reward vs Episodes
plt.figure()
plt.plot(episode_rewards)
plt.xlabel("Episodes")
plt.ylabel("Cumulative Reward")
plt.title("Reward vs Episodes")
plt.grid(True)
plt.savefig("reward_plot.png", dpi=300)
plt.close()

# Safety Violations vs Episodes
plt.figure()
plt.plot(total_violations)
plt.xlabel("Episodes")
plt.ylabel("Safety Violations")
plt.title("Safety Violations vs Episodes")
plt.grid(True)
plt.savefig("violations_plot.png", dpi=300)
plt.close()
import torch

torch.save(agent.model.state_dict(), "dqn_model.pth")
print("Model saved successfully as dqn_model.pth")
