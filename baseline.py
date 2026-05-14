from sim.environment import WaterEnv


env = WaterEnv()


def baseline_policy():

    return [10, 10, 10]


episodes = 50

total_reward = 0

for ep in range(episodes):

    state = env.reset()

    ep_reward = 0

    for step in range(50):

        action = baseline_policy()

        next_state, reward = env.step(action)

        ep_reward += reward

    total_reward += ep_reward

print(
    "Baseline Average Reward:",
    total_reward / episodes
)