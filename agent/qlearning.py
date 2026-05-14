import random


class QLearningAgent:

    def __init__(
        self,
        actions,
        alpha=0.1,
        gamma=0.95,
        epsilon=0.3
    ):

        self.q_table = {}

        self.actions = actions

        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def get_q_values(self, state):

        if state not in self.q_table:

            self.q_table[state] = [0] * len(self.actions)

        return self.q_table[state]

    def choose_action(self, state):

        if random.random() < self.epsilon:

            return random.randint(0, len(self.actions)-1)

        q_values = self.get_q_values(state)

        return q_values.index(max(q_values))

    def update(
        self,
        state,
        action,
        reward,
        next_state
    ):

        q_values = self.get_q_values(state)

        next_q_values = self.get_q_values(next_state)

        target = reward + self.gamma * max(next_q_values)

        q_values[action] += self.alpha * (
            target - q_values[action]
        )