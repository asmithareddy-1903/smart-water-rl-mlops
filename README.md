# Smart Water Distribution RL + MLOps Project

## Project Overview

This project implements a Smart Water Distribution System using Reinforcement Learning (RL) and MLOps practices.

The system simulates intelligent water distribution across multiple regions by learning optimal water allocation policies using the Q-Learning algorithm. The project also integrates MLOps components such as experiment tracking, versioning, Docker deployment, monitoring, API serving, testing, and reproducibility.

The main objective is to reduce:
- Water shortages
- Supply imbalance
- Distribution inefficiency

while improving:
- Fair water allocation
- Pressure balance
- Supply efficiency

---

# Problem Statement

Traditional water distribution systems use fixed allocation strategies that cannot dynamically adapt to changing demand conditions.

This project solves the problem by using Reinforcement Learning to:
- Observe changing water demand
- Learn optimal allocation strategies
- Improve supply efficiency over time

---

# SDG Impact

## SDG 6 – Clean Water and Sanitation

The RL-based system helps improve:
- Water availability
- Efficient resource management
- Sustainable water distribution

by minimizing wastage and improving allocation fairness.

---

# Technologies Used

- Python
- Reinforcement Learning (Q-Learning)
- MLflow
- Docker
- FastAPI
- Matplotlib
- Git & GitHub

---

# Project Structure

smart-water-rl-mlops/

│

├── agent/

│   ├── qlearning.py

│   ├── replay_buffer.py

│   └── policy_utils.py

│

├── sim/

│   ├── environment.py

│   ├── demand_generator.py

│   └── metrics.py

│

├── configs/

│   └── qlearning_v1.yaml

│

├── experiments/

│   ├── exp_01/

│   ├── exp_02/

│   └── exp_03/

│

├── results/

│   ├── results.csv

│   ├── episode_rewards.csv

│   ├── reward_graph.png

│   └── comparison_graph.png

│

├── policies/

│   ├── policy_v1.pkl

│   └── policy_v2_optimized.pkl

│

├── monitoring/

│   └── monitor_metrics.json

│

├── tests/

│   └── test_environment.py

│

├── api.py

├── train.py

├── baseline.py

├── evaluate.py

├── compare.py

├── plot.py

├── queue_plot.py

├── Dockerfile

├── docker-compose.yml

├── requirements.txt

├── README.md

└── .gitignore

---

# Reinforcement Learning Methodology

## Algorithm Used

Q-Learning Algorithm

### Reason for Choosing Q-Learning

Q-Learning was selected because:
- The environment is discrete
- State and action spaces are small
- It is simple and effective for learning optimal policies

---

# Environment Design

## State

The state represents:
- Water demand levels in different regions
- Current supply distribution status

Example:

```python
[region_1_demand, region_2_demand, region_3_demand]
```

---

## Action

The action represents:
- Amount of water allocated to each region

Example:

```python
[20, 30, 25]
```

---

## Reward Function

The reward is designed to:
- Penalize shortages
- Penalize imbalance
- Encourage efficient distribution

Higher rewards indicate better allocation performance.

---

# Exploration Strategy

## Epsilon-Greedy Exploration

- Random exploration during initial episodes
- Gradual reduction of exploration over time
- More exploitation after learning stabilizes

---

# Training Details

## Hyperparameters

| Parameter | Value |
|---|---|
| Episodes | 300 |
| Learning Rate | 0.1 |
| Discount Factor | 0.9 |
| Initial Epsilon | 1.0 |
| Minimum Epsilon | 0.05 |

---

# MLOps Implementation

## Experiment Tracking

MLflow is used for:
- Logging rewards
- Tracking experiments
- Storing metrics
- Managing runs

Tracked metrics:
- Average reward
- Best reward
- Episode statistics

---

## Versioning

Git and GitHub are used for:
- Code versioning
- Experiment tracking
- Collaboration
- Policy version storage

Example tags:
- exp-qlearning-v1
- exp-qlearning-v2

---

# Monitoring

The project includes monitoring metrics such as:
- Supply efficiency
- Pressure balance
- Water shortage tracking
- Distribution fairness

Monitoring metrics are stored in:

```text
monitoring/monitor_metrics.json
```

---

# API Deployment

FastAPI is used to serve the RL system.

## API Endpoints

### Root Endpoint

```text
/
```

Returns:

```json
{
  "message": "Smart Water RL API Running"
}
```

---

### Status Endpoint

```text
/status
```

Returns:

```json
{
  "status": "active",
  "model": "Q-Learning"
}
```

---

# Docker Deployment

The entire project is containerized using Docker.

## Build Docker Image

```bash
docker build -t water-rl .
```

## Run Docker Container

```bash
docker run -p 8000:8000 water-rl
```

---

# Running the Project

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Train RL Model

```bash
python train.py
```

---

## Generate Reward Graph

```bash
python plot.py
```

---

## Generate Comparison Graph

```bash
python compare.py
```

---

## Run API

```bash
uvicorn api:app --reload
```

---

## Run Tests

```bash
python tests/test_environment.py
```

---

# Results

The RL-based system performs better than the baseline fixed allocation strategy by:
- Reducing shortages
- Improving supply efficiency
- Maintaining better balance between regions

Generated outputs:
- Reward graph
- Comparison graph
- Policy files
- Monitoring metrics

---

# Graphs

## Reward Graph

Shows:
- Reward improvement over episodes
- Learning stabilization

---

## Comparison Graph

Compares:
- Baseline distribution
- RL-based optimized distribution

---

# Policies

The trained policies are stored in:

```text
policies/
```

Files:
- policy_v1.pkl
- policy_v2_optimized.pkl

---

# Testing

Environment validation tests are included to verify:
- Environment reset
- Step function
- Reward calculation

---

# Future Improvements

Future enhancements may include:
- Deep Reinforcement Learning
- Real-time sensor integration
- Multi-agent coordination
- Live dashboard monitoring
- Cloud deployment

---

# Conclusion

This project demonstrates how Reinforcement Learning combined with MLOps practices can improve smart water distribution systems.

The system successfully:
- Learns optimized allocation strategies
- Tracks experiments using MLflow
- Supports Docker deployment
- Provides monitoring and API serving
- Maintains reproducibility and version control

---

# Authors

BMSCE AI & ML Department
