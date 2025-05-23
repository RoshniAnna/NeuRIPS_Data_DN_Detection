## 📂 Contents

This folder includes synthetic dataset generation code using OpenDSSdirect.py to support the development and evaluation of intelligent resilience oriented monitoring in power distribution networks. Specifically, it contains code for generating:

- **Line Failure Datasets**: Simulating scenarios of physical line outages under various fault conditions
- **Cyber-Physical Attack Datasets**: Includes attacks targeting:
  - Photovoltaic (PV) systems
  - Electric Vehicle Charging Stations (EVCS)
  - Voltage/current sensors
- Accompanying metadata and structured folder-wise descriptions

---

## 🧠 Motivation

With the growing deployment of distributed energy resources and digitization of grid infrastructure, distribution networks face increasing exposure to both:
- **Natural disruptions** such as line failures caused by storms, wildfires, or equipment degradation
- **Cyber-physical threats** targeting DERs, sensors, and control devices

This dataset collection is designed to:
- Enable the development and benchmarking of robust detection and diagnostic algorithms
- Support research in graph-based learning, outage localization, and anomaly detection
- Promote improved situational awareness and resilience strategies for smart grids

---

## 🧪 Dataset Generation

To generate the datasets, navigate to the desired task and network folder.
Each network-specific folder (e.g., `34-bus`, `123-bus`, `8500-node`) includes all necessary files to simulate and generate datasets. These include:

- **DSS Files (`.dss`)** – Describe the distribution network topology, components, and operational parameters
- **Load Shape and Input Data Files** – Provide varying conditions (e.g., `LoadShape1.dss`, `5mins Irradiance Data (NSRDB).csv`)
- **`GraphBuild.py`** – Contains functions to construct a graph representation of the network using base circuit information
- **`DSSCircuit_Interface.py`** – Enables circuit-level modifications and handles I/O operations with OpenDSS
- **`DataGeneration.py`** – Main script to simulate scenarios and generate labeled datasets
- **`DataGeneration_Multi.py`** – Parallelized or batch scenario generation (if available)

These components work together to produce realistic and reproducible datasets for machine learning tasks related to grid monitoring and security in power distribution systems.

---

### 🔧 How to Run


```bash
python DataGeneration.py
```

or

```text
DataGeneration_multi.py should be executed via a shell script (not provided here due to system-specific configurations).
PostProcess_Multi.py can be used for merging output files. This should also be run via a shell script or batch job.
```

These scripts will automatically simulate the desired scenarios and store the resulting data files in the appropriate directories.


