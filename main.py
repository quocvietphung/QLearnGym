from qlearning.train import train_agent
from qlearning.evaluate import evaluate_agent

q_table, rewards = train_agent()
evaluate_agent(
    q_table_path="models/qtable.npy",
    env_name='Taxi-v3',
    episodes=5,
    render=True
)