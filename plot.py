import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv("results/episode_rewards.csv")

plt.figure(figsize=(10, 5))

plt.plot(
    df["episode"],
    df["reward"],
    label="Reward"
)

plt.xlabel("Episodes")

plt.ylabel("Reward")

plt.title("Reward vs Episodes")

plt.legend()

plt.savefig("results/reward_graph.png")

plt.show()