import matplotlib.pyplot as plt
import random

time_steps = list(range(50))

queue_lengths = [
    random.randint(20, 80)
    for _ in time_steps
]

plt.figure(figsize=(10, 5))

plt.plot(
    time_steps,
    queue_lengths
)

plt.xlabel("Time")

plt.ylabel("Water Demand")

plt.title("Water Demand Over Time")

plt.savefig("results/queue_graph.png")

plt.show()