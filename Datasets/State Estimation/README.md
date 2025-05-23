# State Estimation Dataset

This folder contains code for generating datasets for evaluating state estimation methods in unbalanced power distribution networks. These datasets include time-series voltage measurements under varying operating conditions, generated using OpenDSS simulations of standard IEEE feeders.

The goal is to support the development of learning-based and hybrid approaches for estimating system states (e.g., nodal voltage magnitudes) using limited and noisy sensor measurements.

---

## 📂 Folder Structure

```text
State Estimation/
│
├── 34-bus/               # State estimation data for IEEE 34-bus network
├── 123-bus/              # State estimation data for IEEE 123-bus network
├── 8500-node/            # State estimation data for IEEE 8500-node network
└── README.md             # You are here
