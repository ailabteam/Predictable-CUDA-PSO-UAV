# C-DPSO-CUDA: Predictable Multi-UAV Path Planning

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![CUDA](https://img.shields.io/badge/CUDA-12.x-green.svg)](https://developer.nvidia.com/cuda-toolkit)

This repository contains the official implementation of the paper:  
**"Optimality vs. Predictability: A Statistical and CUDA-Accelerated Analysis of Dynamic PSO for Real-Time Multi-UAV Path Planning"**

## 🚀 Overview
Path planning for large-scale UAV swarms in 3D environments is computationally expensive ($O(N^2 M)$ complexity). Traditional methods often suffer from **temporal variance**, making them unreliable for hard real-time, safety-critical systems.

This project introduces **Chaos-Enhanced Dynamic Parameter PSO (C-DPSO)**, a high-performance framework that leverages **massive CUDA acceleration** to achieve:
1. **Superior Optimality**: Escapes local minima using non-linear chaotic dynamics.
2. **Temporal Predictability**: Provides a guaranteed, low-variance execution time suitable for real-time systems.

---

## ✨ Key Features
*   **C-DPSO Algorithm**: Implements a Logistic Chaos Map for adaptive parameter tuning ($W, C_1, C_2$), significantly improving global search consistency.
*   **Massive CUDA Acceleration**: Utilizes **CuPy** and **Numba JIT** to parallelize fitness evaluations across thousands of GPU threads.
*   **Statistical Analysis Toolset**: Built-in scripts to analyze the trade-off between solution quality (Fitness) and computational reliability (Standard deviation of time).
*   **Multi-Objective Constraints**: Handles dynamic inter-agent collision avoidance, static obstacles, and swarm boundary constraints.

---

## 📊 Performance at a Glance
| Metric | SPSO (Fixed) | L-DPSO (Linear) | **C-DPSO (Chaos)** |
| :--- | :---: | :---: | :---: |
| **Mean Fitness (Lower is better)** | 20681.33 | 15920.62 | **12279.99** |
| **Fitness Std. Dev ($\sigma$)** | 1637.85 | 1053.67 | **628.89** |
| **Time Std. Dev ($\sigma_{Time}$)** | 0.0709s | **0.0114s** | 0.0175s |

> **Key Finding**: C-DPSO offers the best path quality, while L-DPSO provides the highest temporal predictability for mission-critical deadlines.

---

## 🛠️ Installation

### Prerequisites
*   NVIDIA GPU (RTX 30-series or 40-series recommended)
*   CUDA Toolkit 11.x / 12.x
*   Python 3.8+

### Setup
```bash
# Clone the repository
git clone https://github.com/your-username/Predictable-CUDA-PSO-UAV.git
cd Predictable-CUDA-PSO-UAV

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install numpy cupy-cuda12x numba matplotlib pandas
```
*(Note: Replace `cupy-cuda12x` with your specific CUDA version, e.g., `cupy-cuda11x`)*

---

## 💻 Usage

### 1. Run a Single Simulation
To visualize the 3D path planning for an 8-UAV swarm:
```bash
python main_visualize.py --mode chaos --agents 8
```

### 2. Run Statistical Benchmark
To reproduce the "Optimality vs. Predictability" analysis ($N=10$ runs):
```bash
python benchmark_stats.py --iterations 700 --runs 10
```

---

## 📄 Citation
If you use this code or the C-DPSO algorithm in your research, please cite:

```bibtex
@article{do2025optimality,
  title={Optimality vs. Predictability: A Statistical and CUDA-Accelerated Analysis of Dynamic PSO for Real-Time Multi-UAV Path Planning},
  author={Do, Phuc Hao and [Co-authors]},
  journal={Submitted to [IEEE Systems Journa]},
  year={2025}
}
```

---

## 📝 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
