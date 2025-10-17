# 🧠 Time-Series Continual Learning (TSCIL) — Experiments and Analysis

This repository contains our analysis and experimental notebooks for **Time-Series Class-Incremental Learning (TSCIL)**.  
It builds upon the official [TSCIL framework](https://arxiv.org/abs/2402.12035), providing additional visualizations, experiment management, and interpretability tools for studying continual learning behavior in time-series data.

Our goal is to explore how factors such as replay memory size, normalization strategy, and model architecture affect **catastrophic forgetting**, **stability**, and **efficiency** in continual learning.

---

## 👥 Authors

- **Tyler Marino** — Université Paris-Saclay  
- **João Figueiredo** — Universidade do Porto  

This repository was developed as part of a short research project on *frugal AI and continual learning for time-series data*.

---

## 📁 Repository Structure

```
.
├── experiments.ipynb       # Notebook for running new experiments
├── results.ipynb           # Notebook for analyzing and visualizing results
├── environment.yml         # Full Conda environment specification
├── their_README.md         # Original README from the TSCIL authors
└── data/UCI HAR Dataset/           # Preloaded dataset folder (UCI-HAR)
```

---

## ⚙️ Environment Setup

All dependencies are managed through Conda for reproducibility.  
The environment file provided reproduces the exact setup used for all experiments.

### 1. Clone the Repository

```bash
git clone https://github.com/jmcfig/TSCIL-FRUGAL-AI.git
cd results-experiments
```

### 2. Create and Activate the Conda Environment

```bash
conda env create -f environment.yml
conda activate tscl
```

### 3. Launch Jupyter Notebook or JupyterLab

```bash
pip install notebook jupyterlab
jupyter lab
```

> **Note:** All experiments were executed on CPU due to compatibility issues, but the provided environment supports CUDA (11.6) for compatible NVIDIA GPUs.

---

## 📊 Dataset — UCI Human Activity Recognition (HAR)

This project uses the **UCI-HAR** dataset, a well-known benchmark for human activity recognition from smartphone sensor data.  
It contains accelerometer and gyroscope readings from 30 individuals performing six daily activities (e.g., walking, sitting, standing).  

### Dataset Availability

The dataset is **already included** in this repository under the folder:

```
data/uci_har/
```

If you wish to manually download or update it, it can be obtained from the official UCI repository:

🔗 [UCI-HAR Dataset Page](https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones)

After downloading, ensure the files are placed in the same structure used by the TSCIL framework:

```
data/
└── uci_har/
    ├── train/
    ├── test/
    ├── features.txt
    ├── activity_labels.txt
    └── README.txt
```

No further preprocessing is required — TSCIL automatically handles normalization and splitting during runtime.

---

## 🧩 Key Dependencies

Below is a summarized list of major dependencies included in `environment.yml`:

| Category | Libraries |
|-----------|------------|
| **Core ML Frameworks** | `pytorch`, `torchvision`, `torchaudio`, `scikit-learn` |
| **Data Processing** | `numpy`, `pandas`, `scipy`, `pyts`, `imbalanced-learn` |
| **Visualization** | `matplotlib`, `seaborn`, `matplotlib-venn` |
| **Experiment Management** | `ray`, `tensorboardx`, `tqdm`, `wandb` |
| **Time-Series & Deep Learning** | `fastai`, `tsai`, `numba`, `pyts` |
| **System and Utils** | `PyYAML`, `pathlib`, `joblib`, `pickle` |

The full dependency list with CUDA support can be found in [`environment.yml`](./environment.yml).

---

## 🚀 Usage

### Running Experiments

Use `experiments.ipynb` to configure and execute experiments using the TSCIL framework.  
You can modify parameters such as:

- **Agent**: `SFT`, `ER`, `EWC`, etc.  
- **Encoder**: `CNN`, `Transformer`, etc.  
- **Normalization**: `LN`, `BN`, or none  
- **Memory budget**: percentage of past samples retained  

Example (from notebook cell):

```python
!python main.py --agent ER --encoder CNN --norm LN --mem_budget 0.1 --data har
```

---

### Analyzing Results

After running experiments, open `results.ipynb` to visualize and interpret metrics such as:

- **Average End Accuracy**
- **Average Forgetting**
- **Average Current Accuracy**
- **Training Efficiency and Memory Trends**

Plots and CSV summaries are automatically generated under the `exp_logs/` directory.

---

## 🧾 Credits

This repository builds on the official **TSCIL Framework**:

- **Paper**: *Class-Incremental Learning for Time Series: Benchmark and Evaluation*  
  Authors: Zhongzheng Qiao et al. (SIGKDD 2024)  
  [📄 Read the paper](https://arxiv.org/abs/2402.12035)
- **Original Repository**: [TSCIL Framework GitHub](https://github.com/your-tscil-repo)

The original README by the TSCIL authors is included in [`their_README.md`](./their_README.md) for reference.

---

## 📬 Contact

For any questions or issues, please reach out to:  
- **Tyler Marino** — tyler.marino@universite-paris-saclay.fr  
- **João Figueiredo** — up202108829@up.pt
