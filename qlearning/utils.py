import numpy as np
import matplotlib.pyplot as plt
import os

def save_qtable(q_table, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, q_table)
    print(f"✅ Q-table saved to: {path}")

def load_qtable(path):
    q_table = np.load(path)
    print(f"📥 Q-table loaded from: {path}")
    return q_table

def plot_rewards(rewards, path=None):
    plt.figure(figsize=(10, 5))
    plt.plot(rewards, label='Total reward per episode')
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Q-learning Performance Over Time")
    plt.grid(True)
    plt.legend()

    if path:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        plt.savefig(path)
        print(f"📈 Reward plot saved to: {path}")
    else:
        plt.show()