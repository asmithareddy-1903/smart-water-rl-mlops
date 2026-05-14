import os
import csv
import yaml
import mlflow

from sim.environment import WaterEnv, ACTIONS
from agent.qlearning import QLearningAgent
from agent.policy_utils import save_policy

os.makedirs("results", exist_ok=True)
os.makedirs("policies", exist_ok=True)

with open("configs/qlearning_v1.yaml") as f:

    config = yaml.safe_load(f)

mlflow.set_experiment("Smart-Water-RL")

env = WaterEnv()

agent = QLearningAgent(
    ACTIONS,
    alpha=config["learning_rate"],
    gamma=config["gamma"],
    epsilon=config["epsilon"]
)

rewards_per_episode = []

with mlflow.start_run():

    mlflow.log_params(config)

    for ep in range(config["episodes"]):

        state = env.reset()

        total_reward = 0

        for step in range(config["steps_per_episode"]):

            action_idx = agent.choose_action(state)

            action = ACTIONS[action_idx]

            next_state, reward = env.step(action)

            agent.update(
                state,
                action_idx,
                reward,
                next_state
            )

            state = next_state

            total_reward += reward

        rewards_per_episode.append(total_reward)

        mlflow.log_metric(
            "reward",
            total_reward,
            step=ep
        )

        agent.epsilon = max(
            0.05,
            agent.epsilon * 0.995
        )

        print(
            f"Episode {ep+1}: Reward = {total_reward}"
        )

    avg_reward = sum(rewards_per_episode) / len(rewards_per_episode)

    best_reward = max(rewards_per_episode)

    save_policy(
        agent.q_table,
        "policies/policy_v1.pkl"
    )

    save_policy(
        agent.q_table,
        "policies/policy_v2_optimized.pkl"
    )

    with open(
        "results/results.csv",
        "w",
        newline=""
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "run_id",
            "episodes",
            "average_reward",
            "best_reward",
            "epsilon"
        ])

        writer.writerow([
            "run_01",
            config["episodes"],
            avg_reward,
            best_reward,
            agent.epsilon
        ])

    with open(
        "results/episode_rewards.csv",
        "w",
        newline=""
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "episode",
            "reward"
        ])

        for i, r in enumerate(rewards_per_episode):

            writer.writerow([i+1, r])

    mlflow.log_metric(
        "average_reward",
        avg_reward
    )

    mlflow.log_metric(
        "best_reward",
        best_reward
    )

    mlflow.log_artifact("results/results.csv")
    mlflow.log_artifact("results/episode_rewards.csv")

print("\nTraining Complete!")