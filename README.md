# Explainable Safe Reinforcement Learning for Hazard-Aware Navigation

This repository implements an **Explainable Safe Reinforcement Learning (Safe RL)** framework for hazard-aware navigation in a Gridworld environment using **Deep Q-Networks (DQN)**.  
The goal is to enable an agent to reach a target location while **minimizing safety violations** and maintaining **policy interpretability**.

---

## 🚀 Project Overview

Traditional Reinforcement Learning agents focus on reward maximization and often ignore safety constraints during exploration. This project addresses that limitation by integrating **safety-aware reward shaping** and **explainability techniques** into the learning process.

Key contributions:
- Safe navigation using penalty-based constraints
- Deep Q-Network (DQN) agent
- Policy visualization and state visitation heatmaps
- Trajectory analysis for explainability

---

## 🧠 Methodology

- **Environment**: Discrete Gridworld with safe cells, hazardous regions, start state, and goal state  
- **Agent**: Deep Q-Network (DQN)
- **Actions**: Up, Down, Left, Right
- **Safety Mechanism**: Strong negative penalties for hazardous states
- **Explainability**:
  - Agent trajectory visualization
  - Policy visualization
  - State visitation heatmaps

---

## 🗺️ Environment Details

- Grid size: `6 × 6`
- Start state: Top-left corner
- Goal state: Bottom-right corner
- Hazardous cells: Penalized heavily and counted as safety violations

### Reward Structure
| Event | Reward |
|------|--------|
| Normal step | -1 |
| Hazardous state | -30 |
| Goal reached | +10 |

---

## ⚙️ Training Configuration

- Episodes: 500
- Max steps per episode: 50
- Discount factor (γ): 0.99
- Learning rate: 0.001
- Exploration strategy: ε-greedy
- Replay buffer size: 10,000
- Batch size: 64

---

## 📊 Results

- **Success Rate**: 100%
- **Average Safety Violations**: 0.14
- **Improved cumulative reward over episodes**
- **Clear reduction in unsafe exploration**

Visual outputs include:
- Reward curve
- Safety violation plot
- Agent trajectory
- Policy visualization
- State visitation heatmap

---

## 📁 Repository Structure

```text
Safe_RL_Gridworld/
│
├── environment.py        # Gridworld environment definition
├── dqn_agent.py          # Deep Q-Network implementation
├── train.py              # Training script
├── trajectory.py         # Agent trajectory visualization
├── heatmap.py            # State visitation heatmap
├── plots/                # Generated plots and figures
├── models/               # Saved trained models
├── README.md             # Project documentation
└── report.pdf            # IEEE-style project report
