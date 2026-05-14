# Smart Water Distribution RL + MLOps Project

## Problem Statement

Efficient water distribution is critical in smart cities. Traditional fixed water allocation systems may lead to shortage or wastage.

This project uses Reinforcement Learning (Q-Learning) to optimize water distribution dynamically based on demand.

---

## SDG Goal

SDG 6: Clean Water and Sanitation

The project improves:
- efficient water usage
- reduced wastage
- balanced water distribution

---

## Technologies Used

- Python
- Q-Learning
- MLflow
- FastAPI
- Docker
- Matplotlib
- GitHub

---

## Project Structure

- agent/ → RL agent
- sim/ → environment simulation
- results/ → generated outputs
- policies/ → saved trained policies
- monitoring/ → monitoring metrics
- tests/ → testing files

---

## How to Run

### Install Requirements

pip install -r requirements.txt

### Train RL Model

python train.py

### Generate Graphs

python plot.py

python compare.py

python queue_plot.py

### Run MLflow

mlflow ui

### Run API

uvicorn api:app --reload

### Run Docker

docker build -t water-rl .

docker run -p 8000:8000 water-rl

---

## Outputs

- reward_graph.png
- comparison_graph.png
- queue_graph.png
- policy_v1.pkl
- policy_v2_optimized.pkl

---

## Results

The RL-based system performs better than the baseline system by reducing shortages and improving distribution balance.

---

## Authors

BMSCE AI & ML Department