import os
import pickle
import pandas as pd
import mlflow

from sim.environment import WaterEnv
from agent.qlearning import QLearningAgent

# =========================
# CREATE REQUIRED FOLDERS
# =========================

os.makedirs("results", exist_ok=True)
os.makedirs("policies", exist_ok=True)

# =========================
# SET MLFLOW EXPERIMENT
# =========================

mlflow.set_tracking_uri("file:./mlruns")

experiment_name = "SmartWaterRL"

mlflow.set_experiment(experiment_name)

# =========================
# INITIALIZE ENVIRONMENT
# =========================

env = WaterEnv()

state = env.reset()

state_size = len(state)

action_size = len(env.ACTIONS)

# =========================
# INITIALIZE AGENT
# =========================

agent = QLearningAgent(
    state_size,
    action_size
)

# =========================
# TRAINING SETTINGS
# =========================

episodes = 100

episode_rewards = []

# =========================
# START MLFLOW RUN
# =========================

with mlflow.start_run(run_name="Q_Learning_Training_Run"):

    # ---------------------
    # LOG PARAMETERS
    # ---------------------

    mlflow.log_param("Algorithm", "Q-Learning")
    mlflow.log_param("Episodes", episodes)
    mlflow.log_param("State_Size", state_size)
    mlflow.log_param("Action_Size", action_size)

    # =====================
    # TRAINING LOOP
    # =====================

    for episode in range(episodes):

        state = env.reset()

        done = False

        total_reward = 0

        step_count = 0

        while not done:

            # Choose action
            action_index = agent.choose_action(state)

            action = env.ACTIONS[action_index]

            # Environment step
            next_state, reward = env.step(action)

            # Learn
            agent.learn(
                state,
                action_index,
                reward,
                next_state
            )

            # Update state
            state = next_state

            total_reward += reward

            step_count += 1

            # Stop condition
            if step_count >= 50:
                done = True

        # Store reward
        episode_rewards.append(total_reward)

        # ---------------------
        # LOG METRICS
        # ---------------------

        mlflow.log_metric(
            "Episode_Reward",
            total_reward,
            step=episode
        )

        mlflow.log_metric(
            "Steps",
            step_count,
            step=episode
        )

        print(
            f"Episode {episode + 1} | "
            f"Reward = {total_reward}"
        )

    # =====================
    # SAVE RESULTS CSV
    # =====================

    rewards_df = pd.DataFrame({
        "episode": list(range(1, episodes + 1)),
        "reward": episode_rewards
    })

    rewards_csv_path = "results/episode_rewards.csv"

    rewards_df.to_csv(
        rewards_csv_path,
        index=False
    )

    # =====================
    # SAVE POLICY FILES
    # =====================

    policy_v1_path = "policies/policy_v1.pkl"

    with open(policy_v1_path, "wb") as f:
        pickle.dump(agent.q_table, f)

    policy_v2_path = "policies/policy_v2_optimized.pkl"

    with open(policy_v2_path, "wb") as f:
        pickle.dump(agent.q_table, f)

    # =====================
    # LOG ARTIFACTS
    # =====================

    mlflow.log_artifact(rewards_csv_path)

    mlflow.log_artifact(policy_v1_path)

    mlflow.log_artifact(policy_v2_path)

    # Log complete folders
    mlflow.log_artifacts("results")

    mlflow.log_artifacts("policies")

# =========================
# FINAL OUTPUT
# =========================

print("\n===================================")
print("TRAINING COMPLETED SUCCESSFULLY")
print("===================================")

print("Rewards logged to MLflow")
print("Policies saved")
print("Artifacts uploaded")
print("Experiment Name:", experiment_name)