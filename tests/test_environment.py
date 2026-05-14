import sys
import os

sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            ".."
        )
    )
)

from sim.environment import WaterEnv


def test_environment_reset():

    env = WaterEnv()

    state = env.reset()

    assert len(state) == 3


def test_environment_step():

    env = WaterEnv()

    state = env.reset()

    action = [20, 20, 20]

    next_state, reward = env.step(action)

    assert len(next_state) == 3

    assert isinstance(reward, (int, float))


def test_reward_value():

    env = WaterEnv()

    env.reset()

    action = [30, 30, 30]

    _, reward = env.step(action)

    assert reward <= 0


print("All tests passed successfully!")