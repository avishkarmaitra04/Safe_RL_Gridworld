import torch
import numpy as np
from gridworld_env import GridworldEnv
from dqn_agent import DQNAgent

env = GridworldEnv()

state_size = env.observation_space.n
action_size = env.action_space.n

agent = DQNAgent(state_size, action_size)

# Load trained model
agent.model.load_state_dict(torch.load("dqn_model.pth"))
agent.epsilon = 0.0   # disable exploration for testing

episodes = 50
success = 0
rewards = []
violations = []

for _ in range(episodes):
    state = env.reset()
    state = np.eye(state_size)[state]
    done = False
    total_reward = 0

    while not done:
        action = agent.act(state)
        next_state, reward, done, info = env.step(action)
        next_state = np.eye(state_size)[next_state]

        state = next_state
        total_reward += reward

    # ✅ SUCCESS CHECK (VERY IMPORTANT)
    if env.agent_pos == env.goal_pos:
        success += 1

    rewards.append(total_reward)
    violations.append(info["safety_violations"])

print("===== TEST RESULTS =====")
print(f"Success Rate: {success / episodes * 100:.2f}%")
print(f"Average Reward: {np.mean(rewards):.2f}")
print(f"Average Safety Violations: {np.mean(violations):.2f}")
