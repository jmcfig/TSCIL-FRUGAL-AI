# Results and Experiments Analysis

This repository contains Jupyter notebooks for analyzing results and conducting experiments related to time series class-incremental learning (TSCIL). The analysis is based on the TSCIL framework, which provides a unified experimental setup for benchmarking and evaluating continual learning algorithms for time series data.

## Authors
- Tyler Marino, Université Paris-Saclay
- João Figueiredo

---

## Dependencies

To run the notebooks, you need the following dependencies installed:

- **Python 3.10**
- **Jupyter Notebook** or **JupyterLab**
- **TSCIL Framework Dependencies**:
  - `pytorch==1.13.1`
  - `ray==2.3.1`
  - `PyYAML==6.0`
  - `scikit-learn==1.0.2`
  - `matplotlib==3.7.1`
  - `pandas==1.5.3`
  - `seaborn==0.12.2`
- **Additional Libraries**:
  - `numpy`
  - `pickle`
  - `tqdm`

### Installation

1. Clone the repository:
   ```sh
   git clone https://github.com/your-repo/results-experiments.git
   cd results-experiments
   ```

2. Create a Conda environment:
   ```sh
   conda env create -f environment.yml
   ```

3. Activate the environment:
   ```sh
   conda activate tscl
   ```

4. Install Jupyter Notebook:
   ```sh
   pip install notebook
   ```

---

## How to Run

### 1. Results Analysis (`results.ipynb`)
This notebook is used to analyze the results of experiments conducted using the TSCIL framework. It visualizes metrics such as accuracy, forgetting, and memory usage.

To run:
```sh
jupyter notebook results.ipynb
```

### 2. Experiment Setup (`experiments.ipynb`)
This notebook allows you to set up and run new experiments using the TSCIL framework. You can modify hyperparameters, datasets, and algorithms directly in the notebook.

To run:
```sh
jupyter notebook experiments.ipynb
```

---

## Credits

This work is based on the **TSCIL Framework**:
- **Title**: Class-incremental Learning for Time Series: Benchmark and Evaluation
- **Authors**: Zhongzheng Qiao et al.
- **Paper**: [SIGKDD 2024](https://arxiv.org/abs/2402.12035)
- **GitHub Repository**: [TSCIL Framework](https://github.com/your-tscil-repo)

We acknowledge the authors of the TSCIL framework for providing the experimental setup and datasets used in this work.

---

## Contact

For any questions or issues, please contact:
- Tyler Marino: tyler.marino@universite-paris-saclay.fr
- João Figueiredo: joao.figueiredo@example.com