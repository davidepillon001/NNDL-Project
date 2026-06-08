# NNDL Project — Learning Robot Inverse Dynamics with Deep Networks

Project for the Neural Networks and Deep Learning course at the University of Padova.

**A.Y. 2024-2025 — Prof. Jacopo Pegoraro**

## Objective

Learn the **inverse dynamics** of the 7-DOF **PANDA robot arm** — predicting joint torques τ from joint positions q, velocities q̇, and accelerations q̈ — using two approaches:
1. **Standard deep neural networks** (black-box, data-driven)
2. **DeLAN** (Deep Lagrangian Network) — a physics-informed architecture that embeds the Euler-Lagrange structure directly into the network

## Structure

```
data generation code/       — trajectory generation and inverse dynamics simulation
training and validation code/  — model training pipelines (standard NN and DeLAN)
test and compare code/      — evaluation and comparison experiments
report/                     — project report (PDF)
```

## Data Generation (`data generation code/`)

- **`Get_PANDA_data_inv_dyn_sum_of_sin.py`** — generates training data by simulating **sum-of-sines** reference trajectories on the PANDA model. Computes noiseless joint torques via inverse dynamics. Data saved as `.pkl` files.
- **`Get_PANDA_equations.py`** — derives symbolic expressions for the dynamics matrices M(q), C(q,q̇), g(q).
- **`gen_data_sum_of_sins.sh`** — batch script to generate datasets across multiple random seeds.

## Training (`training and validation code/`)

- **`Train_network.ipynb`** — PyTorch pipeline for training standard feed-forward networks. Input: (q, q̇, q̈) → Output: τ. Implements a custom `nMSE` loss (normalized MSE), trains with Adam, and plots per-joint and total training/validation loss curves.
- **`Train_DeLAN.ipynb`** — Training pipeline for the **Deep Lagrangian Network**. DeLAN learns separate sub-networks for the kinetic energy H(q) (inertia matrix) and potential energy V(q), enforcing the structure: `τ = H(q)q̈ + C(q,q̇)q̇ + g(q)`. Architecture: hidden layers [128, 256, 128], Adam optimizer.

## Experiments (`test and compare code/`)

| Notebook | Experiment |
|----------|-----------|
| `Compare_Shallow_Deep_Net.ipynb` | Shallow vs. deep architecture comparison |
| `Compare_nMSE_MSE.ipynb` | nMSE vs. MSE loss function: convergence and accuracy |
| `Compare_normal_sin_cos.ipynb` | Raw joint angles q vs. sin/cos(q) input encoding |
| `Compare_normalized_NMSE.ipynb` | Normalized nMSE metric analysis |
| `Test_data_out_of_distribution.ipynb` | OOD generalization of standard networks |
| `Test_DeLAN_out_of_distribution.ipynb` | OOD generalization of DeLAN vs. standard networks |

## Key Ideas

- **DeLAN** guarantees physical consistency (energy conservation) by construction, leading to better generalization especially out-of-distribution
- **nMSE loss** normalizes per-joint, preventing high-torque joints from dominating training
- **sin/cos encoding** avoids angle wrapping discontinuities in joint space

## Stack

Python · PyTorch · Pinocchio · NumPy · Pandas · Jupyter Notebook
