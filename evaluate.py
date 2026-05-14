from sim.environment import WaterEnv, ACTIONS
from agent.policy_utils import load_policy


env = WaterEnv()

q_table = load_policy("policies/policy_v2_optimized.pkl")

state = env.reset()

for step in range(10):

    if state in q_table:

        action_idx = q_table[state].index(
            max(q_table[state])
        )

    else:

        action_idx = 0

    action = ACTIONS[action_idx]

    next_state, reward = env.step(action)

    print(
        f"Step {step+1} | Action: {action} | Reward: {reward}"
    )

    state = next_state