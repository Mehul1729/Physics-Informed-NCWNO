# Physics-Informed Neural Combinatorial Wavelet Neural Operator (PI-NCWNO)

[![WandB Report](https://img.shields.io/badge/Weights_&_Biases-Report-orange)](https://wandb.ai/mehul22-iiser-thiruvananthapuram/New%20Seq.%20Learning%20-%20Parallel/reports/Physics-Informed-NCWNO-Report--VmlldzoxNTIyOTEwMw?accessToken=r1wxzrhuyzlyp7qojpkj1ipaptazqr5ypk5k29h6112mxxk4ih2ag9xpwgtu0dol)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Work_in_Progress-yellow)]()

## 📌 Overview
This repository contains the PyTorch implementation of the **PI-NCWNO**, a foundational model for solving Partial Differential Equations (PDEs). 

By hybridizing the **Neural Combinatorial Wavelet Neural Operator (NCWNO)** with **Stochastic Projection (SP-PINN)** and **Crank-Nicolson** time-stepping, this framework achieves:
- **Gradient-Free Physics Loss:** Eliminates expensive Automatic Differentiation (AD).
- **6x Training Speedup:** Compared to standard operator learning baselines.
- **Data-Free Learning:** Learns solution operators purely from PDE residuals.

##  Key Features
- **Foundational Backbone:** Mixture-of-Experts (MoE) architecture using Wavelet Neural Operators.
- **Stable Time-Stepping:** Implicit Crank-Nicolson method for robust long-horizon predictions.
- **Efficient Derivatives:** Stochastic projection-based spatial gradient estimation.

---

##  How to run the files:

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/Mehul1729/Physics-Informed-NCWNO.git](https://github.com/Mehul1729/Physics-Informed-NCWNO.git)
   cd Physics-Informed-NCWNO

2. **Install dependency Python libraries:**
    ```bash
    pip install -r requirements.txt

3. **Run/Test the models: **
   - for Foundational training:
    ```bash
    python context_phy_foundational.py

- for testing the model:
  ```bash
  python testing foundational.py
     
