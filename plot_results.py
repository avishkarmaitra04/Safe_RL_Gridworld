import numpy as np
import matplotlib.pyplot as plt

rewards = np.load("rewards.npy")
violations = np.load("violations.npy")

plt.figure()
plt.plot(rewards)
plt.xlabel("Episodes")
plt.ylabel("Total Reward")
plt.title("Reward vs Episodes")
plt.show()

plt.figure()
plt.plot(violations)
plt.xlabel("Episodes")
plt.ylabel("Safety Violations")
plt.title("Safety Violations vs Episodes")
plt.show()
