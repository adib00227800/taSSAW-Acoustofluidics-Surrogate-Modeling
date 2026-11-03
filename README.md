# 3D Multiphysics & Data-Driven Modeling of taSSAW Acoustofluidic Separation

[![COMSOL Multiphysics](https://img.shields.io/badge/COMSOL-Multiphysics_3D-blue.svg)](#)
[![Python](https://img.shields.io/badge/Python-3.9+-yellow.svg)](#)
[![CatBoost](https://img.shields.io/badge/Machine_Learning-CatBoost-green.svg)](#)
[![License: MIT](https://img.shields.io/badge/License-MIT-red.svg)](https://opensource.org/licenses/MIT)

## 📌 Overview
This repository contains the data-driven surrogate modeling framework and simulation outputs from my B.Sc. thesis at the **University of Tehran** (Supervisor: **Dr. Mehdi Moghimi Zand**). 

We developed a fully coupled **3D Finite Element (COMSOL Multiphysics)** model of a **tilted-angle Standing Surface Acoustic Wave (taSSAW)** microfluidic device for continuous, label-free particle and circulating tumor cell (CTC) separation. To overcome the high computational cost of 3D multiphysics evaluations, a **two-stage CatBoost machine learning surrogate model** was trained across parametric simulations to predict particle reachability and final lateral deflection.

---

## 🔬 Methodology & Workflow

### 1. 3D Multiphysics Numerical Modeling (COMSOL)
* **Coupled Physics:** Piezoelectric substrate ($LiNbO_3$) $\rightarrow$ First-order Acoustic Pressure Field $\rightarrow$ Second-order Acoustic Streaming & Laminar Flow $\rightarrow$ Particle Tracing for Fluid Flow (Acoustophoretic + Drag forces).
* **Model Validation:** Benchmark validation against published experimental/numerical literature achieved a **0.9% relative error** in final separation distance.
* **Parametric Dataset Generation:** Conducted **45 high-fidelity 3D simulations** spanning **8 distinct particle and biological cell types** (including polystyrene microparticles, White Blood Cells [WBCs], and MCF-7 breast cancer cells).

### 2. Two-Stage Data-Driven Surrogate Model (CatBoost)
* **Stage 1 (Classification - Reachability Prediction):** Predicts whether a target particle/cell successfully traverses the acoustic active region without trapping or boundary collision.
* **Stage 2 (Regression - Lateral Position Estimation):** Estimates the exact outlet lateral coordinate ($y_{out}$) to determine separation efficiency and purity.

---

## 📊 Key Results & Performance
* **Benchmark Validation Error (COMSOL):** `0.90%` discrepancy in final separation distance.
* **Surrogate Model Accuracy (Polystyrene 2–5 µm separation):** Identified optimal operating parameters with only **`1.86%` prediction error** compared to direct 3D FEM simulations.
* **Biological Cell Separation (WBC vs. MCF-7):** Successfully identified effective separation regimes and systematically analyzed the domain-shift discrepancy in deformable biological cells as a model boundary condition.

---

## 📂 Repository Structure
```text
├── data/                  # Parametric simulation results (CSV) from 3D COMSOL runs
├── figures/               # 3D acoustic pressure contours, streaming fields, and ML parity plots
├── notebooks/             # Jupyter Notebooks for EDA, CatBoost training, and evaluation
├── src/                   # Modular Python scripts for the two-stage surrogate pipeline
└── README.md              # Project documentation
