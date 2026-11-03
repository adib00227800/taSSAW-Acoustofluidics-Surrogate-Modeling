# 3D Multiphysics & Physics-Informed ML Surrogate Modeling for taSSAW Acoustofluidic Separation

[![COMSOL Multiphysics](https://img.shields.io/badge/COMSOL_Multiphysics-3D_FEM-blue.svg)](#)
[![Python](https://img.shields.io/badge/Python-3.9+-yellow.svg)](#)
[![CatBoost](https://img.shields.io/badge/Surrogate_Model-CatBoost_%7C_Scikit--Learn-green.svg)](#)
[![License: MIT](https://img.shields.io/badge/License-MIT-red.svg)](https://opensource.org/licenses/MIT)

## 📌 Overview
This repository contains the **physics-informed machine learning pipeline (`particle_ml_pipeline.py`)** and numerical modeling results from my B.Sc. thesis at the **School of Mechanical Engineering, University of Tehran** (Supervisor: **Dr. Mehdi Moghimi Zand**).

We developed a 3D multiphysics finite element model in **COMSOL Multiphysics** to simulate continuous, label-free separation of microparticles and biological cells (including **WBCs, RBCs, Platelets, HeLa, and MCF-7 breast cancer cells**) using **tilted-angle Standing Surface Acoustic Waves (taSSAW)**. Using data from 45 parametric 3D simulations (360 particle-run observations), we built a **two-stage data-driven surrogate framework** combining reachability classification and physics-informed regression to optimize pairwise separation without expensive 3D FEM re-evaluations.

---

## 🔬 1. 3D Multiphysics Modeling (COMSOL)
* **Piezoelectric Substrate:** Anisotropic $128^\circ\text{ YX-cut } LiNbO_3$ substrate ($20.3 \times 15 \times 1.2\text{ mm}$) excited via Interdigitated Transducers (IDTs) at an operating frequency of **$10.06\text{ MHz}$**.
* **Acoustic & Fluid Coupling:** First-order Helmholtz pressure acoustics coupled with second-order acoustic streaming modeled via **Nyborg's Limiting Velocity method** (slip boundary condition in Creeping Flow, avoiding direct viscous boundary layer meshing $\delta_v = \sqrt{2\nu/\omega}$).
* **Particle Tracing:** Coupled Stokes drag ($F_{\text{drag}}$) and **Settnes & Bruus viscous Acoustic Radiation Force ($F_{\text{rad}}$)** accounting for monopole ($f_1$) and dipole ($f_2$) scattering coefficients.
* **Model Validation:** Validated against *Namnabat et al. (Scientific Reports, 2021)* for $r=2\text{ }\mu\text{m}$ and $r=5\text{ }\mu\text{m}$ polystyrene particles ($\theta = 10^\circ$, $V = 40\text{ V}$), achieving a final separation distance of **$406.20\text{ }\mu\text{m}$ vs. $402.50\text{ }\mu\text{m}$ ($0.90\%$ relative error)**.

### Target Particles & Biological Cells Simulated
| Particle / Cell | Radius $r$ ($\mu\text{m}$) | Diameter $d$ ($\mu\text{m}$) | Density $\rho_p$ ($\text{kg/m}^3$) | Compressibility $\kappa_p$ ($\text{Pa}^{-1}$) |
| :--- | :---: | :---: | :---: | :---: |
| **PS_r2** (Polystyrene) | 2.0 | 4.0 | 1050.0 | $1.72 \times 10^{-10}$ |
| **PS_r5** (Polystyrene) | 5.0 | 10.0 | 1050.0 | $1.72 \times 10^{-10}$ |
| **PS_d15** (Polystyrene) | 7.5 | 15.0 | 1050.0 | $1.72 \times 10^{-10}$ |
| **Platelet** | 1.5 | 3.0 | 1058.0 | $3.48 \times 10^{-10}$ |
| **RBC** (Red Blood Cell) | 3.5 | 7.0 | 1096.0 | $3.48 \times 10^{-10}$ |
| **WBC** (White Blood Cell) | 6.0 | 12.0 | 1019.0 | $3.99 \times 10^{-10}$ |
| **HeLa** (Cervical Cancer) | 8.0 | 16.0 | 1035.7 | $3.55 \times 10^{-10}$ |
| **MCF-7** (Breast Cancer) | 10.0 | 20.0 | 1068.0 | $4.22 \times 10^{-10}$ |

---

## 🤖 2. Physics-Informed Machine Learning Pipeline
Instead of relying on raw excitation voltage ($V_{\text{in}}$) or categorical particle names, the ML pipeline (`particle_ml_pipeline.py`) constructs generalizable physical features to predict the wall-relative lateral deflection $\Delta X = X_{\text{out}} - X_{\text{wall}}(\theta)$, where $X_{\text{wall}}(\theta) = 9.9 - 9\sin(\theta)\text{ [mm]}$:

* **Engineered Physics Features:**
  * Viscous acoustic contrast factor: $\Phi = \frac{1}{3}f_1(\tilde{\kappa}) + \frac{1}{2}\text{Re}[f_2(\tilde{\rho}, \tilde{\delta}_v)]$
  * Acoustic radiation scaling proxy: $\text{Acoustic Proxy} = P_{\max}^2 \cdot r_p^3 \cdot |\Phi|$
  * Trigonometric channel orientation ($\sin\theta, \cos\theta$), $P_{\max}$, $P_{\max}^2$, density ratio ($\rho_p/\rho_f$), and compressibility ratio ($\kappa_p/\kappa_f$).

### Two-Stage Surrogate Architecture
1. **Stage 1 — Reachability Classifier (`HistGradientBoostingClassifier`):** Predicts whether a particle successfully reaches the channel outlet ($346/360$ valid instances). Evaluated via 5-fold Run-Grouped CV:
   * **ROC-AUC:** `0.984` | **Balanced Accuracy:** `0.920`
2. **Stage 2 — Lateral Deflection Regressor:** Benchmarked **10 regression algorithms** across **three strict cross-validation protocols** to prevent data leakage and test out-of-distribution generalization:
   * **Run-CV:** 5-Fold `GroupKFold` grouped by `run_id`.
   * **Leave-One-Angle-Out (LOAO):** Testing interpolation/extrapolation to unseen tilt angles ($\theta \in \{5^\circ, 10^\circ, 15^\circ, 20^\circ, 25^\circ\}$).
   * **Leave-One-Particle-Out (LOPO):** Predicting deflection for completely unseen particle/cell types based solely on $(\rho_p, \kappa_p, d_p)$.

### Benchmark Results Across Validation Protocols
| Model | Run-CV $R^2$ | Run-CV MAE ($\mu\text{m}$) | Run-CV RMSE ($\mu\text{m}$) | LOAO $R^2$ | LOAO RMSE ($\mu\text{m}$) | LOPO $R^2$ | LOPO RMSE ($\mu\text{m}$) | **Composite RMSE ($\mu\text{m}$)** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **CatBoost (Selected)** | **0.942** | 43.53 | 63.79 | 0.861 | 98.66 | **0.742** | **134.28** | **98.91** |
| **Random Forest** | 0.934 | 43.10 | 67.76 | 0.859 | 99.22 | 0.649 | 156.51 | 107.83 |
| **Gradient Boosting** | 0.906 | 54.85 | 80.89 | 0.822 | 111.39 | 0.660 | 154.03 | 115.43 |
| **XGBoost** | 0.934 | 45.08 | 67.94 | 0.809 | 115.54 | 0.619 | 163.09 | 115.52 |
| **HistGradientBoosting** | 0.921 | 51.97 | 74.13 | 0.791 | 120.88 | 0.670 | 151.75 | 115.59 |
| **Extra Trees** | **0.946** | **40.36** | **61.52** | **0.875** | **93.60** | 0.432 | 199.27 | 118.13 |
| **SVR (RBF)** | 0.763 | 81.69 | 128.79 | 0.707 | 142.96 | 0.464 | 193.48 | 155.08 |
| **MLP Neural Net (64,32,16)**| 0.890 | 57.24 | 87.63 | 0.534 | 180.47 | -0.228 | 292.85 | 186.99 |

---

## 🎯 3. Constrained Pairwise Optimization & COMSOL Verification
Using the trained **CatBoost** regressor and **Reachability** classifier ($P(\text{reach}) \ge 0.80$), we searched 905 operating states within the angle-dependent pressure envelope and verified the recommended optima via direct 3D COMSOL simulations:

| Target Pair | Optimal Angle ($\theta$) | Optimal $P_{\max}$ (MPa) | Equiv. Drive ($V_{\text{in}}$) | Surrogate Predicted Separation ($\mu\text{m}$) | Direct 3D COMSOL Separation ($\mu\text{m}$) | Relative Error (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **PS_r2 – PS_r5** | $20^\circ$ | $1.886\text{ MPa}$ | $111.47\text{ V}$ | $868.15\text{ }\mu\text{m}$ | **$852.30\text{ }\mu\text{m}$** | **1.86%** |
| **WBC – MCF-7** | $15^\circ$ | $1.484\text{ MPa}$ | $62.92\text{ V}$ | $88.85\text{ }\mu\text{m}$ | **$169.20\text{ }\mu\text{m}$** | 47.49%* |

> *\*Note on WBC–MCF-7:* Although sharp local gradients in acoustic response led to an under-prediction of absolute separation magnitude, the surrogate model successfully pinpointed an operating regime yielding **$169.20\text{ }\mu\text{m}$ separation** in direct COMSOL verification—within **$0.44\text{ }\mu\text{m}$** of the global maximum observed across all 45 baseline runs ($169.64\text{ }\mu\text{m}$).

---

## 🚀 Usage & Reproducibility
Run the complete data preparation, multi-protocol cross-validation, pair optimization, and automated HTML dashboard generation:

```bash
pip install numpy pandas scikit-learn matplotlib joblib catboost xgboost
python particle_ml_pipeline.py --data-glob "run_results_master*.csv" --output-dir "particle_ml_outputs"
