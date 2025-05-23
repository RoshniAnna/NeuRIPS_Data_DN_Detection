# PowerBench

**PowerBench** is a comprehensive benchmark suite for evaluating machine learning methods on resilience oriented grid-monitoring and security tasks in power distribution networks. It includes structured datasets generated synthetically using distribution network models in OpenDSS and standardized evaluation scenarios for:

- Line Failure Detection
- Cyberattack Detection
- State Estimation

Each dataset is organized by task type and IEEE test feeders (34-bus, 123-bus, 8500-node).

---

## 📁 Repository Structure

```text
PowerBench/
│
├── Datasets/              # All datasets organized by task
│   ├── Cyber Attack Detection/
│   ├── Line Failure Detection/
│   └── State Estimation/
│
├── Experiments/           # Scripts or model files for evaluating datasets
├── README.md              # You are here!
```

---

## 📊 Dataset Tasks


| Task                       | Description                                                                                 |
|----------------------------|---------------------------------------------------------------------------------------------|
| **Cyber Attack Detection** | Detect attacks on EVCS, PV, and sensors and locate compromised devices                      |
| **Line Failure Detection** | Identify if lines have failed using partial obervability in unbalanced distribution networks|
| **State Estimation**       | Estimate voltage magnitudes using partial measurements                                      |

---

## 🚀 Usage

To evaluate or benchmark models, refer to the `Experiments/` directory.
Each dataset folder includes a dedicated `README.md` file containing details about the dataset generation.



