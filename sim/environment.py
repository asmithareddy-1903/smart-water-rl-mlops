import random

from sim.demand_generator import generate_demand

ACTIONS = [
    [15, 10, 5],
    [5, 15, 10],
    [20, 5, 5],
    [10, 20, 5],
    [5, 5, 20],
    [15, 15, 0],
    [0, 15, 15]
]

class WaterEnv:

    def __init__(self):
        self.ACTIONS = ACTIONS
        self.reset()

    def reset(self):

        self.tank_level = 100

        self.regions = [50, 50, 50]

        self.leakage = random.randint(0, 5)

        self.emergency_flag = random.choice([0, 1])

        self.current_demand = generate_demand()

        return self.get_state()

    def get_state(self):

        return tuple([
            self.tank_level,
            self.regions[0],
            self.regions[1],
            self.regions[2],
            self.current_demand[0],
            self.current_demand[1],
            self.current_demand[2],
            self.leakage,
            self.emergency_flag
        ])

    def step(self, action):

        shortage = 0
        wastage = 0

        for i in range(3):

            self.regions[i] += action[i]

            self.regions[i] -= self.current_demand[i]

            if self.regions[i] < 0:

                shortage += abs(self.regions[i])

                self.regions[i] = 0

            if self.regions[i] > 100:

                wastage += self.regions[i] - 100

                self.regions[i] = 100

        leakage_penalty = self.leakage

        balanced_bonus = 10 - abs(
            self.regions[0] - self.regions[1]
        )

        reward = (
            - shortage * 2
            - wastage
            - leakage_penalty
            + balanced_bonus
        )

        self.current_demand = generate_demand()

        next_state = self.get_state()

        return next_state, reward