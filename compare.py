import matplotlib.pyplot as plt


baseline_reward = -4200
rl_reward = -3200

models = [
    "Baseline System",
    "RL Optimized System"
]

values = [
    baseline_reward,
    rl_reward
]

plt.figure(figsize=(8, 5))

bars = plt.bar(
    models,
    values
)

plt.ylabel("Average Reward")

plt.xlabel("Model Type")

plt.title(
    "Baseline vs RL Performance"
)

for bar in bars:

    yval = bar.get_height()

    plt.text(
        bar.get_x() + bar.get_width()/2,
        yval,
        round(yval, 2),
        ha='center',
        va='bottom'
    )

plt.grid(axis='y')

plt.savefig(
    "results/comparison_graph.png"
)

plt.show()