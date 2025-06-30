import numpy as np

class QLearningAgent:
    def __init__(self, state_size, action_size, alpha, gamma, epsilon, epsilon_min, epsilon_decay):
        self.q_table = np.zeros((state_size, action_size))  # Khởi tạo Q-table
        self.alpha = alpha                                  # Tốc độ học
        self.gamma = gamma                                  # Hệ số giảm giá
        self.epsilon = epsilon                              # Xác suất explore
        self.epsilon_min = epsilon_min                      # Giá trị nhỏ nhất của epsilon
        self.epsilon_decay = epsilon_decay                  # Tốc độ giảm epsilon

    def choose_action(self, state):
        # Exploration vs Exploitation
        if np.random.uniform(0, 1) < self.epsilon:
            return np.random.randint(self.q_table.shape[1])  # Chọn hành động ngẫu nhiên
        return np.argmax(self.q_table[state])                # Chọn hành động tốt nhất

    def update(self, state, action, reward, next_state, done):
        best_next_action = np.argmax(self.q_table[next_state])
        td_target = reward + self.gamma * self.q_table[next_state, best_next_action] * (not done)
        td_error = td_target - self.q_table[state, action]
        self.q_table[state, action] += self.alpha * td_error

        # Giảm epsilon nếu cần
        if done and self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay