import random
import numpy as np


class QLearningAgent:

    def __init__(
        self,
        state_size,
        action_size,
        learning_rate=0.1,
        gamma=0.9,
        epsilon=0.1
    ):

        self.state_size = state_size

        self.action_size = action_size

        self.learning_rate = learning_rate

        self.gamma = gamma

        self.epsilon = epsilon

        self.q_table = {}

    def get_q_value(self, state, action):

        return self.q_table.get((state, action), 0.0)

    def choose_action(self, state):

        if random.uniform(0, 1) < self.epsilon:

            return random.randint(0, self.action_size - 1)

        q_values = [
            self.get_q_value(state, a)
            for a in range(self.action_size)
        ]

        return int(np.argmax(q_values))

    def learn(self, state, action, reward, next_state):

        old_q = self.get_q_value(state, action)

        future_q = max([
            self.get_q_value(next_state, a)
            for a in range(self.action_size)
        ])

        new_q = old_q + self.learning_rate * (
            reward + self.gamma * future_q - old_q
        )

        self.q_table[(state, action)] = new_q