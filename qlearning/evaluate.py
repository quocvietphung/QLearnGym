import gym
import numpy as np
from qlearning.utils import load_qtable

def evaluate_agent(
    q_table_path="models/qtable.npy",
    env_name='Taxi-v3',
    episodes=100,
    max_steps=100,
    render=False
):
    env = gym.make(env_name)
    q_table = load_qtable(q_table_path)

    total_rewards = []

    for episode in range(episodes):
        state = env.reset()[0] if isinstance(env.reset(), tuple) else env.reset()
        total_reward = 0
        for step in range(max_steps):
            action = np.argmax(q_table[state])  # Always exploit
            next_state, reward, done, *_ = env.step(action)
            total_reward += reward
            state = next_state

            if render:
                env.render()

            if done:
                break
        total_rewards.append(total_reward)

    avg_reward = np.mean(total_rewards)
    print(f"🎯 Evaluation over {episodes} episodes — Avg reward: {avg_reward:.2f}")
    return avg_reward