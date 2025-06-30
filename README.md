

# 🚕 Q-Learning Taxi Agent (OpenAI Gym)

This project implements a **Q-Learning** agent to solve the classic `Taxi-v3` environment from OpenAI Gym. The agent learns to efficiently pick up and drop off passengers using Reinforcement Learning.

---

## 📁 Project Structure

```
qlearning-cartpole/
├── README.md                # Project documentation
├── requirements.txt         # Required Python packages
├── main.py                  # Main script for training and evaluation
├── notebooks/
│   └── 01_qlearning_cartpole.ipynb  # Jupyter notebook walkthrough
├── qlearning/
│   ├── agent.py             # QLearningAgent class
│   ├── train.py             # Training logic
│   ├── evaluate.py          # Evaluation logic
│   ├── utils.py             # Utility functions: plot, save/load Q-table
├── models/
│   └── qtable.npy           # Trained Q-table
├── results/
│   └── rewards.png          # Training reward plot
```

---

## ⚙️ Installation

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Launch Jupyter Notebook (optional)

```bash
jupyter notebook
```

---

## 🚀 How to Use

### ✔️ Run the training and evaluation via terminal:

```bash
python main.py
```

This will:
- Train the Q-Learning agent in the `Taxi-v3` environment
- Save the Q-table to `models/qtable.npy`
- Generate a reward plot in `results/rewards.png`
- Evaluate the agent and display the map (ASCII-based) if `render=True`

---

### 📓 Run in Jupyter Notebook

Open `notebooks/01_qlearning_cartpole.ipynb` and execute the cells to:

1. Train the agent
2. Plot the reward curve
3. Evaluate the learned policy (with interactive map display)

---

## 🧠 Agent Overview

- **Algorithm**: Q-Learning with epsilon-greedy exploration
- **Environment**: `Taxi-v3` (discrete state and action space)
- **Render mode**: `"ansi"` to visualize agent movement in ASCII terminal
- **Performance**: Reward increases progressively; agent performs consistently after ~3000 episodes

---

## 📊 Analysis Suggestions

- Compare training performance across different values of:
  - `alpha` (learning rate)
  - `gamma` (discount factor)
  - `epsilon_decay`
- Visualize and log average reward over time
- Evaluate stability and convergence of the Q-table

---

## 🧱 Possible Extensions

- ✅ Adapt to `CartPole-v1` (requires discretizing the continuous state space)
- ✅ Implement Deep Q-Network (DQN) with PyTorch
- ✅ Grid Search for hyperparameter optimization
- ✅ Create a custom environment under `envs/`

---

## 📌 Author

Created by Viet Phung with architectural and technical guidance from ChatGPT.
Released for educational and research purposes.