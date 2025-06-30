import gym
import numpy as np
from qlearning.agent import QLearningAgent
from qlearning.utils import plot_rewards, save_qtable

def train_agent(
    env_name='Taxi-v3',
    num_episodes=5000,
    max_steps=100,
    alpha=0.7,
    gamma=0.95,
    epsilon=1.0,
    epsilon_min=0.1,
    epsilon_decay=0.995,
    model_path="models/qtable.npy",
    plot_path="results/rewards.png"
):
    env = gym.make(env_name)
    state_size = env.observation_space.n
    action_size = env.action_space.n

    agent = QLearningAgent(state_size, action_size, alpha, gamma, epsilon, epsilon_min, epsilon_decay)
    rewards = []

    for episode in range(num_episodes):
        state = env.reset()[0] if isinstance(env.reset(), tuple) else env.reset()
        total_reward = 0

        for _ in range(max_steps):
            action = agent.choose_action(state)
            next_state, reward, done, *_ = env.step(action)
            agent.update(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            if done:
                break

        rewards.append(total_reward)
        if (episode + 1) % 100 == 0:
            print(f"Episode {episode+1}/{num_episodes} - Reward: {total_reward:.2f} - Epsilon: {agent.epsilon:.4f}")

    # Lưu kết quả
    save_qtable(agent.q_table, model_path)
    plot_rewards(rewards, path=plot_path)

    return agent.q_table, rewards