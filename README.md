# NNDL Project — Deep Lagrangian Networks for Robot Inverse Dynamics

Project for the Neural Networks and Deep Learning course at the University of Padova.

The goal is to learn the inverse dynamics of the PANDA robot arm using physics-informed neural networks, comparing standard deep networks against **DeLaN** (Deep Lagrangian Network), which embeds the Euler-Lagrange structure into the architecture.

## Structure

```
data generation code/   — scripts to generate sum-of-sines trajectories via pinocchio
training and validation code/   — training notebooks for DeLaN and standard networks
test and compare code/  — evaluation: out-of-distribution tests, MSE/nMSE comparison
report/                 — project report (PDF)
```

## Key experiments

- Comparison between shallow and deep networks
- MSE vs. normalized MSE metrics
- Normal vs. sin/cos joint angle encoding
- Out-of-distribution generalization of DeLaN

## Stack

Python · PyTorch · Pinocchio · Jupyter Notebook
