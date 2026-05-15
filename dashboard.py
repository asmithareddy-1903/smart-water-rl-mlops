import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.title("Smart Water Distribution Monitoring Dashboard")

st.subheader("Reward Monitoring")

df = pd.read_csv("results/episode_rewards.csv")

st.line_chart(df["reward"])

st.subheader("Reward Statistics")

st.write(df.describe())

st.subheader("System Monitoring Metrics")

monitoring_data = {
    "Water Shortage": 12,
    "Supply Efficiency": 91,
    "Pressure Balance": 88,
    "Distribution Fairness": 90
}

st.bar_chart(monitoring_data)

st.success("System Running Successfully")