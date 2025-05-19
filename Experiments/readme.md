# PowerBench: Experiments for Power Grid Outage, Attack Detection, localization, and State estimate

This directory contains experimental code for training and evaluating machine learning models for power grid outage classification, cyberattack detection and localization, and State estimate tasks. These experiments use topological features (e.g., Betti numbers) along with graph-based and deep learning models.

---

## 📁 Directory Structure
```
PowerBench/
└── Experiments/
├── betti_extraction_outage.py # Extracts topological features
├── data_loader.py # Loads datasets
├── logger.py # Logging utility
├── models.py # Model definitions (MLP, GNN, etc.)
├── train_StateE.py # State estimation
├── train_attack_detection.py # Cyberattack detection
├── train_location_acc.py # Attack localization
├── train_outage.py # Outage detection
└── train_outage_MP_MLP.py # Multi-persitent MLP
```

---

## 📦 Datasets

Please make sure the datasets are organized under the following structure before running the experiments:
```
datasets/
├── EVCSAttacks/
│ ├── 123bus/
│ ├── 34bus/
│ └── 8500bus/
├── PVAttacks/
│ ├── 123bus/
│ ├── 34bus/
│ └── 8500bus/
└── SensorAttacks/
|  ├── 123bus/
|  ├── 34bus/
|  └── 8500bus/
└── StateEstimate/
  ├── 123bus/
  ├── 34bus/
  └── 8500bus/
```
## 🚀 Running an Experiment

Example:

```bash
python train_attack_detection.py --dataset datasets/EVCSAttacks/123bus/
