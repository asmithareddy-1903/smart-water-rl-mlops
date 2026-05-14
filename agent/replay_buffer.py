from collections import deque
import random


class ReplayBuffer:

    def __init__(self, capacity=1000):

        self.buffer = deque(maxlen=capacity)

    def add(self, experience):

        self.buffer.append(experience)

    def sample(self, batch_size=32):

        return random.sample(self.buffer, batch_size)