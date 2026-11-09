# 3D Multiphysics & Physics-Guided Machine Learning Surrogate Modeling for taSSAW Acoustofluidic Separation

[![COMSOL Multiphysics](https://img.shields.io/badge/COMSOL_Multiphysics-3D_FEM-blue.svg)](#)
[![Python](https://img.shields.io/badge/Python-3.9+-yellow.svg)](#)
[![CatBoost](https://img.shields.io/badge/Surrogate-CatBoost_%7C_Scikit--Learn-green.svg)](#)

## Overview

This repository accompanies the B.Sc. research project of **Adib Saffarifard**, conducted at the **School of Mechanical Engineering, University of Tehran**, under the supervision of **Dr. Mehdi Moghimi Zand**.

The project investigates continuous, label-free separation of microparticles and biological cells using **tilted-angle standing surface acoustic waves (taSSAWs)**. It combines three-dimensional finite-element simulation in COMSOL Multiphysics with a two-stage machine-learning surrogate for predicting particle transport and identifying promising separation conditions.

The numerical framework incorporates piezoelectric actuation, first-order pressure acoustics, boundary-driven acoustic streaming, hydrodynamic transport, acoustic radiation forces, and three-dimensional particle tracing.

A structured parametric study generates **360 particle-level observations from 45 independent COMSOL simulations**, covering eight particle and cell species.

The machine-learning framework uses physically motivated features, leakage-controlled cross-validation, outlet-reachability classification, and pair-specific optimization to screen operating conditions without repeatedly solving the complete three-dimensional multiphysics model.

**The surrogate is intended for design-space screening and prioritization of high-fidelity simulations, not as a replacement for the COMSOL solver.**

## 1. Three-Dimensional Multiphysics Model

### 1.1 Device geometry and operating conditions

The modeled device consists of an anisotropic lithium-niobate piezoelectric substrate, interdigitated transducers (IDTs), and a tilted microfluidic channel.

| Parameter | Value |
|---|---|
| Piezoelectric material | 128° YX-cut LiNbO₃ |
| Substrate dimensions | 20.3 × 15 × 1.2 mm |
| Microchannel length | 9 mm |
| Microchannel width | 1 mm |
| **Microchannel height** | **0.075 mm (75 µm)** |
| Operating frequency | 10.06 MHz |
| Main inlet velocity | 6 mm/s |
| Tilt angles | 5°, 10°, 15°, 20°, 25° |
| Principal excitation voltages | 15, 30, 40, 50, 60, 70, 90, 110, 130 V |
| Independent operating runs | 45 |

The numerical geometry includes a particle-focusing region, a side-flow realignment stage, and an inclined-wave separation region.

Changing the channel inclination angle modifies the orientation of the pressure-node pattern relative to the primary flow direction and the position of the channel outlet.

### 1.2 Multiphysics coupling

The COMSOL implementation uses a staged solution procedure:

1. **Piezoelectric actuation:** Solid Mechanics and Electrostatics calculate the harmonic electromechanical response of the anisotropic LiNbO₃ substrate.
2. **First-order acoustics:** The substrate displacement excites the acoustic pressure and oscillatory velocity fields inside the liquid.
3. **Acoustic streaming:** A limiting-velocity formulation calculates the effective boundary slip and transfers it to a stationary Creeping Flow calculation.
4. **Particle dynamics:** Acoustic radiation and hydrodynamic drag determine the particle trajectories.
5. **Particle tracing:** Three-dimensional trajectories and terminal outlet positions are calculated for each particle or cell species.

The coupling is sequential rather than a fully simultaneous monolithic time-domain solution.

### 1.3 Boundary-driven acoustic streaming

Acoustic streaming is modeled using the limiting-velocity approach based on Nyborg's formulation and its subsequent developments.

The characteristic viscous acoustic boundary-layer thickness is

\[
\delta_v=\sqrt{\frac{2\nu}{\omega}}
\]

where \(\nu\) is the kinematic viscosity and \(\omega\) is the angular excitation frequency.

Instead of explicitly resolving the thin viscous acoustic boundary layer, the first-order acoustic field and vibrating-wall motion are used to calculate an effective slip velocity at the fluid-solid interface.

This velocity is supplied as a boundary condition to the Creeping Flow model.

The approach reduces the mesh requirements of the three-dimensional calculation, but it remains an approximation to the near-wall viscous flow.

### 1.4 Particle forces

Particle motion accounts for the acoustic radiation force and Stokes drag.

The radiation force follows the viscous small-particle formulation of **Settnes and Bruus**, incorporating monopole and dipole scattering effects and the density and compressibility contrasts between particle and fluid.

Hydrodynamic drag is represented by

\[
\mathbf F_{\mathrm{drag}}
=6\pi\eta r_p(\mathbf v_f-\mathbf v_p)
\]

where \(\eta\) is dynamic viscosity, \(r_p\) is particle radius, and \(\mathbf v_f\) and \(\mathbf v_p\) are fluid and particle velocities.

Under the quasi-steady particle approximation,

\[
\mathbf F_{\mathrm{rad}}+
\mathbf F_{\mathrm{drag}}\approx0
\]

Particle inertia, pairwise particle interactions, and near-wall corrections to Stokes drag are neglected.

### 1.5 Particle and cell properties

Eight particle or cell species are included in the parametric simulations.

| Particle / Cell | Radius (µm) | Diameter (µm) | Density (kg/m³) | Compressibility (Pa⁻¹) |
|---|---:|---:|---:|---|
| PS_r2 | 2.0 | 4.0 | 1050.0 | 1.72 × 10⁻¹⁰ |
| PS_r5 | 5.0 | 10.0 | 1050.0 | 1.72 × 10⁻¹⁰ |
| PS_d15 | 7.5 | 15.0 | 1050.0 | 1.72 × 10⁻¹⁰ |
| Platelet | 1.5 | 3.0 | 1058.0 | 3.48 × 10⁻¹⁰ |
| RBC | 3.5 | 7.0 | 1096.0 | 3.48 × 10⁻¹⁰ |
| WBC | 6.0 | 12.0 | 1019.0 | 3.99 × 10⁻¹⁰ |
| HeLa | 8.0 | 16.0 | 1035.7 | 3.55 × 10⁻¹⁰ |
| MCF-7 | 10.0 | 20.0 | 1068.0 | 4.22 × 10⁻¹⁰ |

The species are modeled as equivalent spheres with fixed acoustic and mechanical properties. Cell deformation and heterogeneous internal structures are not included.

## 2. Numerical Benchmark Against Published Results

The three-dimensional implementation is compared with the **sheathless taSSAW numerical separator** presented by Namnabat, Moghimi Zand, and Houshfar (2021).

The reference configuration contains focusing, realignment, and separation stages and uses:

- A 10° inclination angle
- A 40 V principal excitation voltage
- Polystyrene particles with diameters of 4 and 10 µm

These correspond to the particles named PS_r2 and PS_r5 in this project, where the numerical suffix denotes particle radius.

### Terminal separation comparison

| Quantity | Value |
|---|---:|
| Published numerical reference | 402.500 µm |
| Present COMSOL reconstruction | 406.201 µm |
| Absolute difference | 3.701 µm |
| Relative discrepancy | **0.92%** |

The relative discrepancy is calculated as

\[
E_{\mathrm{rel}}
=\frac{|406.201-402.5|}{402.5}\times100
\approx0.92\%
\]

**Interpretation:** The 402.5 µm reference value is a numerical result reported by Namnabat et al., not an independent experimental measurement.

The present model uses an operating frequency of 10.06 MHz, whereas the published reference sheathless case uses 9.63 MHz.

Agreement is therefore reported specifically for the **terminal interparticle separation distance**.

Force histories and particle trajectories are compared qualitatively. Differences remain in the \(ARF_y\) component and in the final position of each particle relative to the channel wall.

This numerical benchmark does not establish complete agreement in the acoustic fields, forces, individual trajectories, or mesh-independent solution accuracy.

### Mesh selection

The model uses local mesh refinement in the microchannel and acoustically active regions:

| Region | Nominal element size |
|---|---|
| General computational domain | 0.06–0.12 mm |
| Microchannel | 0.015–0.03 mm |
| Sensitive edges | Additional refinement, scale factor 0.35 |

The final mesh was selected with consideration of numerical stability, spatial resolution, computational cost, and agreement with the published terminal separation result.

**A systematic mesh-independence study was not performed.** The influence of systematic mesh refinement on acoustic pressure, streaming velocity, radiation force, and terminal particle position remains unquantified.

## 3. Physics-Guided Machine Learning Pipeline

The Python implementation, `particle_ml_pipeline.py`, uses numerical outputs and physically motivated feature engineering to estimate terminal particle displacement.

The approach is described as **physics-guided surrogate modeling** rather than a physics-constrained neural network. Physical relationships inform feature construction, but the regression objective does not directly enforce the governing partial differential equations.

### 3.1 Dataset

The database comprises:

| Quantity | Value |
|---|---:|
| Independent COMSOL runs | 45 |
| Particle/cell species per run | 8 |
| Total particle-level observations | 360 |
| Reached outlet | 346 |
| Did not reach outlet | 14 |
| Maximum pressure range | 0.255–3.061 MPa |
| Fixed frequency | 10.06 MHz |
| Fixed main inlet velocity | 6 mm/s |

The target displacement is defined relative to a reference channel wall:

\[
\Delta X=X_{\mathrm{particle}}-X_{\mathrm{wall}}
\]

with

\[
X_{\mathrm{wall}}(\theta)
=9.9-9\sin\theta\quad [\mathrm{mm}]
\]

where \(\theta\) is the channel inclination angle.

Using a geometry-relative target allows terminal positions to be compared between simulations with different channel orientations.

### 3.2 Physics-guided features

The regression features include:

- Channel inclination angle and trigonometric orientation descriptors
- Maximum first-order acoustic pressure \(P_{\max}\)
- Pressure-squared terms and pressure–particle-size interactions
- Particle radius, density, and compressibility
- Particle-to-fluid density and compressibility ratios
- Acoustic contrast descriptors and radiation-force scaling proxies
- Reference-wall position

A representative acoustic radiation scaling descriptor is

\[
Q_{\mathrm{ac}}\propto
P_{\max}^{2}r_p^{3}|\Phi|
\]

where \(\Phi\) represents an acoustic contrast descriptor.

The maximum acoustic pressure is extracted from the COMSOL solution rather than inferred from voltage alone.

Particle identity is excluded as a categorical input from the selected general regression model, enabling evaluation on held-out particle types using their physical properties.

### 3.3 Two-stage surrogate architecture

**Stage 1: Outlet reachability classification**

A HistGradientBoosting classifier estimates whether a particle reaches a valid outlet position.

The classification stage uses all 360 observations and accounts for the imbalance between reached and unreached samples.

Reported grouped-validation performance:

| Metric | Result |
|---|---:|
| ROC-AUC | 0.984 |
| Balanced accuracy | 0.920 |

**Stage 2: Terminal displacement regression**

The second stage predicts wall-relative lateral displacement only for the 346 observations with valid terminal positions.

Ten regression algorithms are compared using three validation protocols.

### 3.4 Leakage-controlled evaluation

The validation framework uses:

**Run-CV:** Five-fold GroupKFold validation, using the COMSOL run identifier to prevent observations from the same operating run appearing simultaneously in training and validation folds.

**Leave-One-Angle-Out (LOAO):** All samples belonging to one inclination angle are withheld to evaluate transfer to an unseen geometric setting.

**Leave-One-Particle-Out (LOPO):** One particle or cell species is withheld to evaluate generalization based on physical-property inputs.

These protocols evaluate different forms of generalization. In particular, predicting a new operating run for known particles is generally less demanding than predicting a previously unseen particle type.

### 3.5 Regression benchmark results

| Model | Run-CV R² | Run RMSE (µm) | LOAO RMSE (µm) | LOPO RMSE (µm) | Composite RMSE (µm) |
|---|---:|---:|---:|---:|---:|
| **CatBoost** | 0.942 | 63.79 | 98.66 | 134.28 | **98.91** |
| Random Forest | 0.934 | 67.76 | 99.22 | 156.51 | 107.83 |
| Gradient Boosting | 0.906 | 80.89 | 111.39 | 154.03 | 115.43 |
| XGBoost | 0.934 | 67.94 | 115.54 | 163.09 | 115.52 |
| HistGradientBoosting | 0.921 | 74.13 | 120.88 | 151.75 | 115.59 |
| Extra Trees | 0.946 | 61.52 | 93.60 | 199.27 | 118.13 |
| Decision Tree | 0.883 | 90.28 | 115.23 | 166.44 | 123.98 |
| SVR-RBF | 0.763 | 128.79 | 142.96 | 193.48 | 155.08 |
| MLP | 0.890 | 87.63 | 180.47 | 292.85 | 186.99 |
| Ridge | 0.604 | 166.32 | 174.23 | 240.55 | 193.70 |

The composite RMSE is the arithmetic mean of the Run-CV, LOAO, and LOPO RMSE values.

**CatBoost is selected for balanced performance across all three generalization protocols**, rather than because it has the lowest error in every individual test.

Extra Trees produces slightly lower Run-CV and LOAO errors but substantially larger LOPO error.

The evaluation results are cross-validation estimates, not a separate independent final test set.

## 4. Constrained Pairwise Separation Optimization

The optimization objective is the terminal separation between particles \(A\) and \(B\):

\[
S_{AB}=|\Delta X_A-\Delta X_B|
\]

provided both reach the outlet.

Candidates are accepted only when

\[
P_{\mathrm{reach},A}\ge0.80,\qquad
P_{\mathrm{reach},B}\ge0.80
\]

The search is constrained to the five simulated angles and the corresponding angle-specific ranges of acoustic pressure.

A total of **905 candidate operating states** are screened.

The selected pressure is converted to an equivalent excitation voltage using the angle-specific pressure–voltage relationship, followed by direct COMSOL verification.

### Optimization results

| Target pair | Angle | Pmax (MPa) | Equivalent voltage | Surrogate separation (µm) | COMSOL separation (µm) | Relative discrepancy |
|---|---:|---:|---:|---:|---:|---:|
| **PS_r2–PS_r5** | 20° | 1.886 | 111.47 V | 868.15 | **852.30** | **1.86%** |
| **WBC–MCF-7** | 15° | 1.484 | 62.92 V | 88.85 | **169.20** | **47.49%** |

### 4.1 Polystyrene pair

The highest separation observed directly among the original 45 runs is **857.60 µm**, at 20° and 90 V.

The surrogate-selected candidate produces 852.30 µm in direct COMSOL verification, which is 5.30 µm below the best original result.

Consequently, the surrogate identified a **high-performing operating region**, with 1.86% prediction discrepancy, rather than discovering a new maximum separation.

### 4.2 WBC–MCF-7 pair

The highest directly observed separation in the original parametric database is **169.64 µm**, at 5° and 50 V.

The surrogate-selected candidate produces 169.20 µm in direct COMSOL verification, only 0.44 µm below that observed maximum.

However, the surrogate predicts only 88.85 µm for this condition and therefore substantially underestimates the separation magnitude.

Pair-specific out-of-fold evaluation also shows a negative \(R^2\) for WBC–MCF-7, indicating difficulty in representing its sharply varying response.

This illustrates an important distinction: **identifying a promising operating condition is not equivalent to predicting its performance accurately**.

The 169.64 µm result is the largest observed value within the evaluated 45-run grid, not a demonstrated global physical optimum.

## 5. Limitations

The following limitations should be considered when interpreting the numerical and machine-learning results.

**Numerical modeling:** No systematic mesh-independence study was performed. Boundary-driven streaming is represented using a limiting-velocity approximation rather than directly resolving the acoustic viscous boundary layer.

**Reference comparison:** The 402.5 µm benchmark is a published numerical result. No independent experimental verification of the present model is reported, and agreement in terminal separation does not imply agreement in all force components or trajectories.

**Physical simplifications:** Particles and cells are represented as rigid equivalent spheres with constant properties. Cell deformation, near-wall drag corrections, and particle–particle interactions are not explicitly modeled.

**Dataset size:** The 360 particle observations come from only 45 distinct operating simulations. The reachability dataset also contains only 14 unreached observations.

**Generalization:** The current surrogate is evaluated over limited ranges of angle, pressure, and particle properties. The frequency and inlet velocity remain fixed. Extrapolation beyond the represented conditions is not established.

**Prediction uncertainty:** The reported cross-validation errors and pair-specific metrics do not constitute calibrated prediction intervals. Ensemble-based uncertainty estimates and empirical 95% prediction-interval coverage have not yet been reported.

**Engineering performance:** The objective measures final geometric separation, not experimentally measured sorting purity, throughput, recovery, or biological cell viability.

## 6. Future Work

Potential extensions include:

- Systematic mesh-refinement and mesh-independence assessment
- Independent experimental verification
- Additional COMSOL sampling near sharp biological-cell separation peaks
- Run-group-aware ensemble uncertainty estimation and calibrated prediction intervals
- Active-learning strategies to prioritize additional high-fidelity simulations
- Variation of operating frequency and inlet flow velocity
- Trajectory-aware surrogate models using local force histories and acoustic node-crossing information
- Evaluation of biological sorting purity, recovery, throughput, and viability

## 7. Usage and Reproducibility

The main machine-learning implementation is identified as:

`particle_ml_pipeline.py`

The pipeline is designed to process particle-level COMSOL outputs, construct machine-learning features, evaluate regression models, and screen pairwise operating conditions.

### Installation

Python 3.9 or later is the intended environment.

Install the main analysis dependencies:

```bash
pip install numpy pandas scikit-learn matplotlib joblib catboost xgboost
```

### Run the analysis

The supplied command-line interface uses CSV input files matching the specified pattern:

```bash
python particle_ml_pipeline.py \
  --data-glob "run_results_master*.csv" \
  --output-dir "particle_ml_outputs"
```

The pipeline is intended to perform data preparation, cross-validation, model comparison, pairwise optimization, and output generation.

**Reproducibility requirement:** The relevant observation-level CSV files must be available in the specified input location. The presence and completeness of those files in the public repository must be checked independently.

Reproducing the machine-learning analysis from exported COMSOL results is distinct from reproducing the original three-dimensional COMSOL simulations. The latter also requires the appropriate COMSOL model, material settings, boundary conditions, and software environment.

## 8. Research Contribution and Attribution

The underlying three-dimensional acoustofluidic approach builds upon established work, particularly:

- **Namnabat et al. (2021):** Three-dimensional modeling of SSAW-driven acoustophoretic motion incorporating boundary-driven streaming and limiting velocity.
- **Settnes and Bruus (2012):** Viscous acoustic radiation force formulation for small compressible particles.
- **Nyborg (1953):** Foundational acoustic streaming formulation.
- **Ding et al. (2014):** Tilted-angle SSAW separation of particles and biological cells.

The principal contribution of this project is the combination of a staged three-dimensional numerical implementation with a structured parametric database and a two-stage, physics-guided machine-learning framework for constrained, pair-specific separation screening.

The project does not claim to introduce a new acoustic radiation force equation or a new limiting-velocity theory.

## 9. References

**Namnabat, M. S., Moghimi Zand, M., and Houshfar, E. (2021).** 3D numerical simulation of acoustophoretic motion induced by boundary-driven acoustic streaming in standing surface acoustic wave microfluidics. *Scientific Reports*, 11, 13326.  
https://doi.org/10.1038/s41598-021-90825-z

**Settnes, M., and Bruus, H. (2012).** Forces acting on a small particle in an acoustical field in a viscous fluid. *Physical Review E*, 85, 016327.  
https://doi.org/10.1103/PhysRevE.85.016327

**Ding, X., et al. (2014).** Cell separation using tilted-angle standing surface acoustic waves. *Proceedings of the National Academy of Sciences*, 111, 12992–12997.  
https://doi.org/10.1073/pnas.1413325111

**Nyborg, W. L. (1953).** Acoustic streaming due to attenuated plane waves. *Journal of the Acoustical Society of America*, 25, 68–75.

## 10. Project Information

**Researcher:** Adib Saffarifard  
**Supervisor:** Dr. Mehdi Moghimi Zand  
**Institution:** School of Mechanical Engineering, University of Tehran  
**Research area:** Acoustofluidics, microfluidics, multiphysics simulation, and physics-guided machine learning  
**Project:** B.Sc. research thesis, 2026

**Repository:** https://github.com/adib00227800/taSSAW-Acoustofluidics-Surrogate-Modeling

---

*This repository describes a numerical research framework. The reported performance values are simulation-based unless explicitly stated otherwise. Physical device performance requires independent experimental assessment.*
