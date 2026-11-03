#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Physics-informed ML pipeline for acoustic particle separation.

What this script does
---------------------
1) Reads all run_results_master*.csv files.
2) Converts the wide tables into one row per (run, particle).
3) Computes:
      x_wall = 9.9 mm - 9 mm * sin(theta)
      target = x_particle - x_wall   [um]
4) Uses Pmax_MPa instead of V_in_V in the ML features.
5) Adds particle physics features (diameter, density, compressibility,
   viscous acoustic contrast, P^2 and size/pressure interaction proxies).
6) Compares several regression models under THREE validation protocols:
      - unseen operating run (GroupKFold by run_id)
      - unseen particle type (Leave-One-Particle-Out)
      - unseen angle (Leave-One-Angle-Out)
7) Selects the model with the lowest mean RMSE across the protocols.
8) Trains a separate reachability classifier because some particles do not
   reach the outlet; outlet position is undefined for those cases.
9) Optimizes pair separation |x_rel_A - x_rel_B| over theta and P, while:
      - staying inside the observed theta range,
      - staying inside the theta-dependent observed pressure envelope,
      - requiring both particles to have adequate predicted reach probability.
10) Can predict outlet displacement for NEW particles from their physical
    properties, without using particle name as a model input.

Important limitation
--------------------
The input CSVs contain FINAL outlet positions, not time-resolved trajectories.
Therefore this script predicts FINAL outlet x_rel, not a full x(t), y(t) path.
A trajectory model requires trajectory/time-series data from COMSOL.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import html
import warnings
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

# -------------------------------------------------------------------------
# HEADLESS / WINDOWS STABILITY
# -------------------------------------------------------------------------
# This pipeline only SAVES figures; it never needs a GUI window.
# Force Matplotlib to use the non-interactive Agg backend so Tkinter/Tcl
# is never created in worker threads on Windows (especially Python 3.13).
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import joblib
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
    roc_auc_score,
    balanced_accuracy_score,
    recall_score,
)
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

warnings.filterwarnings("ignore")

# Optional models. The pipeline works without them, but CatBoost/XGBoost are recommended.
try:
    from xgboost import XGBRegressor
    HAS_XGBOOST = True
except Exception:
    HAS_XGBOOST = False

try:
    from catboost import CatBoostRegressor
    HAS_CATBOOST = True
except Exception:
    HAS_CATBOOST = False


# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------

RANDOM_STATE = 42

# Geometry
X0_WALL_MM = 9.9
L_CH_MM = 9.0
W_CH_MM = 1.0
W_CH_UM = W_CH_MM * 1000.0

# Fluid properties from the provided COMSOL parameter file
RHO_F = 997.0                 # kg/m^3
KAPPA_F = 4.48e-10            # 1/Pa
ETA_F = 0.890e-3              # Pa.s

# Reachability threshold used during pair optimization.
# 0.80 is conservative enough for this small dataset; change if needed.
MIN_REACH_PROB = 0.80

# Dense constrained optimization grid.
N_THETA_GRID = 161
N_PRESSURE_FRACTION_GRID = 181

# If None -> optimize all possible particle pairs.
# Example:
# SPECIAL_PAIRS = [("WBC", "MCF7"), ("RBC", "Platelet")]
SPECIAL_PAIRS: Optional[List[Tuple[str, str]]] = None

# Particle properties. Diameter = 2 * characteristic radius from the source file.
# Names match the CSV column suffixes exactly.
PARTICLE_PROPERTIES: Dict[str, Dict[str, float]] = {
    "PS_r2": {
        "diameter_um": 4.0,
        "density_kg_m3": 1050.0,
        "compressibility_1_pa": 1.72e-10,
    },
    "PS_r5": {
        "diameter_um": 10.0,
        "density_kg_m3": 1050.0,
        "compressibility_1_pa": 1.72e-10,
    },
    "WBC": {
        "diameter_um": 12.0,
        "density_kg_m3": 1019.0,
        "compressibility_1_pa": 3.99e-10,
    },
    "MCF7": {
        "diameter_um": 20.0,
        "density_kg_m3": 1068.0,
        "compressibility_1_pa": 4.22e-10,
    },
    "HeLa": {
        "diameter_um": 16.0,
        "density_kg_m3": 1035.7,
        "compressibility_1_pa": 3.55e-10,
    },
    "RBC": {
        "diameter_um": 7.0,
        "density_kg_m3": 1096.0,
        "compressibility_1_pa": 3.48e-10,
    },
    "Platelet": {
        "diameter_um": 3.0,
        "density_kg_m3": 1058.0,
        "compressibility_1_pa": 3.48e-10,
    },
    "PS_d15": {
        "diameter_um": 15.0,
        "density_kg_m3": 1050.0,
        "compressibility_1_pa": 1.72e-10,
    },
}


# -----------------------------------------------------------------------------
# UTILITY / PHYSICS
# -----------------------------------------------------------------------------

def x_wall_um(theta_deg: np.ndarray | float) -> np.ndarray | float:
    """x_wall = 9.9 mm - 9 mm*sin(theta), returned in micrometers."""
    return (X0_WALL_MM - L_CH_MM * np.sin(np.deg2rad(theta_deg))) * 1000.0


def add_physics_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add physics-informed features.

    f1, f2 and Lambda follow the provided COMSOL definitions. We use the
    real-valued acoustic contrast proxy:
        phi = (1/3) * f1 + (1/2) * Re(f2)

    The model also receives P^2 and a size-pressure interaction proxy because
    the acoustic radiation force contains pressure/velocity-field energy terms
    and scales with particle volume (~a^3).
    """
    out = df.copy()

    diameter_um = out["diameter_um"].astype(float).to_numpy()
    radius_m = 0.5 * diameter_um * 1e-6
    rho_p = out["density_kg_m3"].astype(float).to_numpy()
    kappa_p = out["compressibility_1_pa"].astype(float).to_numpy()
    freq_hz = out["frequency_MHz"].astype(float).to_numpy() * 1e6

    omega = 2.0 * np.pi * freq_hz
    delta_v = np.sqrt(2.0 * (ETA_F / RHO_F) / omega)
    delta_tilde = delta_v / radius_m

    lambda_visc = -(3.0 / 2.0) * (1.0 + 1j * (1.0 + delta_tilde)) * delta_tilde
    rho_tilde = rho_p / RHO_F
    f1 = 1.0 - kappa_p / KAPPA_F
    f2 = (
        2.0
        * (1.0 - lambda_visc)
        * (rho_tilde - 1.0)
        / (2.0 * rho_tilde + 1.0 - 3.0 * lambda_visc)
    )
    phi = (1.0 / 3.0) * f1 + 0.5 * np.real(f2)

    theta_rad = np.deg2rad(out["theta_deg"].astype(float))
    out["sin_theta"] = np.sin(theta_rad)
    out["cos_theta"] = np.cos(theta_rad)

    out["P2_MPa2"] = out["Pmax_MPa"].astype(float) ** 2
    out["radius_um"] = out["diameter_um"].astype(float) / 2.0
    out["radius_cubed_um3"] = out["radius_um"] ** 3
    out["density_ratio"] = out["density_kg_m3"].astype(float) / RHO_F
    out["compressibility_ratio"] = out["compressibility_1_pa"].astype(float) / KAPPA_F
    out["contrast_phi"] = phi
    out["acoustic_proxy"] = (
        out["P2_MPa2"]
        * out["radius_cubed_um3"]
        * np.abs(out["contrast_phi"])
    )
    return out


def pressure_envelope(wide: pd.DataFrame) -> pd.DataFrame:
    """Observed theta-dependent P range used to prevent operating extrapolation."""
    return (
        wide.groupby("theta_deg")["Pmax_MPa"]
        .agg(["min", "max"])
        .sort_index()
        .reset_index()
    )


def pressure_from_fraction(
    theta_deg: np.ndarray,
    pressure_fraction: np.ndarray,
    envelope: pd.DataFrame,
) -> np.ndarray:
    """Map fraction [0,1] to the interpolated observed P range at each theta."""
    theta_obs = envelope["theta_deg"].to_numpy(float)
    pmin_obs = envelope["min"].to_numpy(float)
    pmax_obs = envelope["max"].to_numpy(float)
    pmin = np.interp(theta_deg, theta_obs, pmin_obs)
    pmax = np.interp(theta_deg, theta_obs, pmax_obs)
    return pmin + pressure_fraction * (pmax - pmin)


# -----------------------------------------------------------------------------
# DATA PREPARATION
# -----------------------------------------------------------------------------

def find_particles_from_columns(df: pd.DataFrame) -> List[str]:
    particles = []
    for p in PARTICLE_PROPERTIES:
        if f"x_{p}_um" in df.columns and f"reached_{p}" in df.columns:
            particles.append(p)
    return particles


def load_wide_data(data_glob: str) -> pd.DataFrame:
    files = sorted(Path().glob(data_glob)) if not os.path.isabs(data_glob) else []
    if os.path.isabs(data_glob):
        import glob as _glob
        files = [Path(p) for p in sorted(_glob.glob(data_glob))]

    if not files:
        raise FileNotFoundError(f"No CSV files matched: {data_glob}")

    frames = []
    for path in files:
        df = pd.read_csv(path)
        df["source_file"] = path.name
        frames.append(df)

    wide = pd.concat(frames, ignore_index=True)

    required = [
        "run_id",
        "theta_deg",
        "Pmax_MPa",
        "frequency_MHz",
        "inlet_velocity_mm_s",
    ]
    missing = [c for c in required if c not in wide.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    return wide


def make_long_dataset(wide: pd.DataFrame) -> pd.DataFrame:
    particles = find_particles_from_columns(wide)
    if not particles:
        raise ValueError("No recognized particle x_*/reached_* columns found.")

    rows: List[dict] = []
    for _, r in wide.iterrows():
        theta = float(r["theta_deg"])
        wall = float(x_wall_um(theta))

        for particle in particles:
            props = PARTICLE_PROPERTIES[particle]
            x_value = r.get(f"x_{particle}_um", np.nan)
            reached_raw = r.get(f"reached_{particle}", False)
            reached = bool(reached_raw) and pd.notna(x_value)

            rows.append(
                {
                    "run_id": str(r["run_id"]),
                    "source_file": r.get("source_file", ""),
                    "theta_deg": theta,
                    "Pmax_MPa": float(r["Pmax_MPa"]),
                    "frequency_MHz": float(r["frequency_MHz"]),
                    "inlet_velocity_mm_s": float(r["inlet_velocity_mm_s"]),
                    "particle": particle,
                    **props,
                    "reached": int(reached),
                    "x_um": float(x_value) if pd.notna(x_value) else np.nan,
                    "x_wall_um": wall,
                    "x_rel_um": (float(x_value) - wall) if reached else np.nan,
                }
            )

    long_df = pd.DataFrame(rows)
    return add_physics_features(long_df)


def audit_pressure_voltage(wide: pd.DataFrame) -> pd.DataFrame:
    """Audit V-P linearity only; voltage is NEVER used as a regression feature."""
    if "V_in_V" not in wide.columns:
        return pd.DataFrame()

    records = []
    for theta, g in wide.groupby("theta_deg"):
        if g["V_in_V"].nunique() < 2:
            continue
        slope, intercept = np.polyfit(g["V_in_V"], g["Pmax_MPa"], 1)
        pred = slope * g["V_in_V"] + intercept
        ss_res = float(np.sum((g["Pmax_MPa"] - pred) ** 2))
        ss_tot = float(np.sum((g["Pmax_MPa"] - g["Pmax_MPa"].mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
        records.append(
            {
                "theta_deg": theta,
                "P_per_V_slope_MPa_per_V": slope,
                "intercept_MPa": intercept,
                "R2_V_to_P": r2,
            }
        )
    return pd.DataFrame(records)


# -----------------------------------------------------------------------------
# FEATURES / MODELS
# -----------------------------------------------------------------------------

def choose_feature_columns(reg_df: pd.DataFrame) -> List[str]:
    features = [
        "theta_deg",
        "sin_theta",
        "cos_theta",
        "Pmax_MPa",
        "P2_MPa2",
        "diameter_um",
        "radius_cubed_um3",
        "density_ratio",
        "compressibility_ratio",
        "contrast_phi",
        "acoustic_proxy",
    ]

    # These can only be learned if they actually vary in the supplied tables.
    for c in ["frequency_MHz", "inlet_velocity_mm_s"]:
        if reg_df[c].nunique(dropna=True) > 1:
            features.append(c)
    return features


def build_regressors(fast_mode: bool = False) -> Dict[str, object]:
    n_trees = 160 if fast_mode else 300
    boost_iters = 180 if fast_mode else 350
    mlp_iters = 450 if fast_mode else 900

    models: Dict[str, object] = {
        "Ridge": Pipeline(
            [("scale", StandardScaler()), ("model", Ridge(alpha=1.0))]
        ),
        "DecisionTree": DecisionTreeRegressor(
            max_depth=6, min_samples_leaf=4, random_state=RANDOM_STATE
        ),
        "RandomForest": RandomForestRegressor(
            n_estimators=n_trees,
            min_samples_leaf=2,
            max_features=0.9,
            random_state=RANDOM_STATE,
            n_jobs=1,
        ),
        "ExtraTrees": ExtraTreesRegressor(
            n_estimators=n_trees,
            min_samples_leaf=2,
            max_features=0.9,
            random_state=RANDOM_STATE,
            n_jobs=1,
        ),
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=boost_iters,
            learning_rate=0.03,
            max_depth=3,
            min_samples_leaf=3,
            loss="huber",
            random_state=RANDOM_STATE,
        ),
        "HistGradientBoosting": HistGradientBoostingRegressor(
            max_iter=boost_iters,
            learning_rate=0.05,
            max_leaf_nodes=15,
            l2_regularization=1.0,
            random_state=RANDOM_STATE,
        ),
        "SVR_RBF": Pipeline(
            [
                ("scale", StandardScaler()),
                ("model", SVR(C=100.0, epsilon=5.0, gamma="scale")),
            ]
        ),
        "MLP_Neural": Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "model",
                    MLPRegressor(
                        hidden_layer_sizes=(64, 32, 16),
                        activation="relu",
                        solver="adam",
                        alpha=1e-3,
                        learning_rate_init=1e-3,
                        max_iter=mlp_iters,
                        early_stopping=True,
                        validation_fraction=0.15,
                        n_iter_no_change=30,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }

    if HAS_XGBOOST:
        models["XGBoost"] = XGBRegressor(
            n_estimators=n_trees,
            max_depth=4,
            learning_rate=0.03,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=2.0,
            objective="reg:squarederror",
            random_state=RANDOM_STATE,
            n_jobs=1,
        )

    if HAS_CATBOOST:
        models["CatBoost"] = CatBoostRegressor(
            iterations=n_trees,
            depth=6,
            learning_rate=0.03,
            loss_function="RMSE",
            verbose=False,
            random_seed=RANDOM_STATE,
            l2_leaf_reg=5.0,
            thread_count=1,
            allow_writing_files=False,
        )

    return models


def metric_dict(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "MAE_um": mean_absolute_error(y_true, y_pred),
        "RMSE_um": math.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred),
        "MedAE_um": median_absolute_error(y_true, y_pred),
    }


def oof_predict(
    model: object,
    X: pd.DataFrame,
    y: pd.Series,
    splits: Sequence[Tuple[np.ndarray, np.ndarray]],
) -> np.ndarray:
    pred = np.full(len(y), np.nan, dtype=float)
    for train_idx, test_idx in splits:
        m = clone(model)
        m.fit(X.iloc[train_idx], y.iloc[train_idx])
        pred[test_idx] = m.predict(X.iloc[test_idx])
    return pred


def evaluate_regressors(
    reg_df: pd.DataFrame,
    features: Sequence[str],
    models: Dict[str, object],
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    X = reg_df[list(features)].reset_index(drop=True)
    y = reg_df["x_rel_um"].reset_index(drop=True)

    run_groups = reg_df["run_id"].astype(str).to_numpy()
    particle_groups = reg_df["particle"].astype(str).to_numpy()
    angle_groups = reg_df["theta_deg"].astype(str).to_numpy()

    run_splits = list(
        GroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE).split(
            X, y, run_groups
        )
    )
    particle_splits = list(LeaveOneGroupOut().split(X, y, particle_groups))
    angle_splits = list(LeaveOneGroupOut().split(X, y, angle_groups))

    protocols = {
        "run_cv": run_splits,
        "particle_cv": particle_splits,
        "angle_cv": angle_splits,
    }

    records = []
    run_oof_by_model: Dict[str, np.ndarray] = {}

    for name, model in models.items():
        print(f"[model] evaluating {name} ...")
        for protocol_name, splits in protocols.items():
            pred = oof_predict(model, X, y, splits)
            mask = np.isfinite(pred)
            metrics = metric_dict(y.to_numpy()[mask], pred[mask])
            records.append(
                {
                    "model": name,
                    "protocol": protocol_name,
                    "N": int(mask.sum()),
                    **metrics,
                }
            )
            if protocol_name == "run_cv":
                run_oof_by_model[name] = pred.copy()

    result = pd.DataFrame(records)
    return result, run_oof_by_model


def select_model_name(metrics: pd.DataFrame) -> Tuple[str, pd.DataFrame]:
    score = (
        metrics.pivot(index="model", columns="protocol", values="RMSE_um")
        .copy()
    )
    score["mean_RMSE_um"] = score.mean(axis=1)
    score = score.sort_values("mean_RMSE_um")
    return str(score.index[0]), score.reset_index()


# -----------------------------------------------------------------------------
# REACHABILITY CLASSIFIER
# -----------------------------------------------------------------------------

def fit_reachability_classifier(
    long_df: pd.DataFrame, features: Sequence[str]
) -> HistGradientBoostingClassifier:
    clf = HistGradientBoostingClassifier(
        max_iter=350,
        learning_rate=0.05,
        max_leaf_nodes=12,
        l2_regularization=2.0,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )
    clf.fit(long_df[list(features)], long_df["reached"].astype(int))
    return clf


def evaluate_reachability_classifier(
    long_df: pd.DataFrame,
    features: Sequence[str],
    classifier: HistGradientBoostingClassifier,
) -> pd.DataFrame:
    X = long_df[list(features)].reset_index(drop=True)
    y = long_df["reached"].astype(int).reset_index(drop=True)
    groups = long_df["run_id"].astype(str).to_numpy()
    splits = list(
        GroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE).split(
            X, y, groups
        )
    )

    prob = np.full(len(y), np.nan)
    for tr, te in splits:
        m = clone(classifier)
        m.fit(X.iloc[tr], y.iloc[tr])
        prob[te] = m.predict_proba(X.iloc[te])[:, 1]

    mask = np.isfinite(prob)
    yt = y.to_numpy()[mask]
    yp = prob[mask]
    yh = (yp >= 0.5).astype(int)

    result = {
        "N": int(mask.sum()),
        "ROC_AUC_reached": roc_auc_score(yt, yp) if len(np.unique(yt)) == 2 else np.nan,
        "balanced_accuracy": balanced_accuracy_score(yt, yh),
        "recall_reached": recall_score(yt, yh, pos_label=1, zero_division=0),
        "recall_not_reached": recall_score(1 - yt, 1 - yh, pos_label=1, zero_division=0),
    }
    return pd.DataFrame([result])


# -----------------------------------------------------------------------------
# PREDICTION HELPERS
# -----------------------------------------------------------------------------

def make_condition_particle_frame(
    theta_deg: np.ndarray,
    p_mpa: np.ndarray,
    frequency_mhz: np.ndarray | float,
    inlet_velocity_mm_s: np.ndarray | float,
    particle_name: str,
    particle_props: Dict[str, float],
) -> pd.DataFrame:
    theta_deg = np.asarray(theta_deg, dtype=float)
    p_mpa = np.asarray(p_mpa, dtype=float)
    n = len(theta_deg)

    freq = np.broadcast_to(np.asarray(frequency_mhz, dtype=float), (n,))
    vin = np.broadcast_to(np.asarray(inlet_velocity_mm_s, dtype=float), (n,))

    df = pd.DataFrame(
        {
            "theta_deg": theta_deg,
            "Pmax_MPa": p_mpa,
            "frequency_MHz": freq,
            "inlet_velocity_mm_s": vin,
            "particle": particle_name,
            "diameter_um": float(particle_props["diameter_um"]),
            "density_kg_m3": float(particle_props["density_kg_m3"]),
            "compressibility_1_pa": float(particle_props["compressibility_1_pa"]),
        }
    )
    return add_physics_features(df)


def predict_particle(
    regressor: object,
    reach_classifier: object,
    features: Sequence[str],
    theta_deg: np.ndarray,
    p_mpa: np.ndarray,
    frequency_mhz: np.ndarray | float,
    inlet_velocity_mm_s: np.ndarray | float,
    particle_name: str,
    particle_props: Dict[str, float],
) -> Tuple[np.ndarray, np.ndarray]:
    frame = make_condition_particle_frame(
        theta_deg,
        p_mpa,
        frequency_mhz,
        inlet_velocity_mm_s,
        particle_name,
        particle_props,
    )
    pred = np.asarray(regressor.predict(frame[list(features)]), dtype=float)
    # x_rel must lie inside the 1-mm channel width in this geometry.
    pred = np.clip(pred, 0.0, W_CH_UM)
    reach_prob = reach_classifier.predict_proba(frame[list(features)])[:, 1]
    return pred, reach_prob


# -----------------------------------------------------------------------------
# PAIR OPTIMIZATION
# -----------------------------------------------------------------------------


def estimate_voltage_from_pressure(
    theta_deg: np.ndarray | float,
    p_mpa: np.ndarray | float,
    wide: pd.DataFrame,
) -> np.ndarray:
    """
    Estimate the equivalent drive voltage ONLY FOR REPORTING / COMSOL re-run setup.
    Voltage is never used as an ML feature.

    A linear P=aV+b relationship is fitted independently at each observed angle,
    then a(theta), b(theta) are interpolated if a non-observed theta is requested.
    """
    theta_arr = np.atleast_1d(np.asarray(theta_deg, dtype=float))
    p_arr = np.atleast_1d(np.asarray(p_mpa, dtype=float))
    theta_arr, p_arr = np.broadcast_arrays(theta_arr, p_arr)

    if "V_in_V" not in wide.columns or wide["V_in_V"].nunique(dropna=True) < 2:
        return np.full(theta_arr.shape, np.nan, dtype=float)

    audit = audit_pressure_voltage(wide).sort_values("theta_deg")
    if audit.empty:
        return np.full(theta_arr.shape, np.nan, dtype=float)

    th_obs = audit["theta_deg"].to_numpy(float)
    slopes = audit["P_per_V_slope_MPa_per_V"].to_numpy(float)
    intercepts = audit["intercept_MPa"].to_numpy(float)

    slope = np.interp(theta_arr, th_obs, slopes)
    intercept = np.interp(theta_arr, th_obs, intercepts)
    with np.errstate(divide="ignore", invalid="ignore"):
        v = (p_arr - intercept) / slope
    v[~np.isfinite(v)] = np.nan
    return v


def optimize_pairs(
    regressor: object,
    reach_classifier: object,
    features: Sequence[str],
    wide: pd.DataFrame,
    particle_properties: Dict[str, Dict[str, float]],
    pairs: Optional[Sequence[Tuple[str, str]]] = None,
    min_reach_prob: float = MIN_REACH_PROB,
    observed_angles_only: bool = True,
) -> pd.DataFrame:
    """
    Optimize pair separation in a constrained domain.

    Default is intentionally conservative: optimize only at angles that were
    actually simulated. With only five current angles, this avoids treating
    angle interpolation as equally trustworthy as a directly covered angle.
    Pressure is searched continuously but only inside the observed pressure
    envelope of each angle.

    Set observed_angles_only=False only if Leave-One-Angle-Out validation is
    sufficiently strong and continuous angle interpolation is scientifically
    justified.
    """
    envelope = pressure_envelope(wide).sort_values("theta_deg")

    if observed_angles_only:
        theta_grid = envelope["theta_deg"].to_numpy(float)
    else:
        theta_grid = np.linspace(
            float(envelope["theta_deg"].min()),
            float(envelope["theta_deg"].max()),
            N_THETA_GRID,
        )

    fraction_grid = np.linspace(0.0, 1.0, N_PRESSURE_FRACTION_GRID)
    T, F = np.meshgrid(theta_grid, fraction_grid, indexing="ij")
    theta_flat = T.ravel()
    frac_flat = F.ravel()
    p_flat = pressure_from_fraction(theta_flat, frac_flat, envelope)

    frequency_mhz = float(wide["frequency_MHz"].median())
    inlet_velocity = float(wide["inlet_velocity_mm_s"].median())

    pred_by_particle: Dict[str, np.ndarray] = {}
    reach_by_particle: Dict[str, np.ndarray] = {}

    for name, props in particle_properties.items():
        pred, reach = predict_particle(
            regressor,
            reach_classifier,
            features,
            theta_flat,
            p_flat,
            frequency_mhz,
            inlet_velocity,
            name,
            props,
        )
        pred_by_particle[name] = pred
        reach_by_particle[name] = reach

    if pairs is None:
        pair_list = list(combinations(particle_properties.keys(), 2))
    else:
        pair_list = list(pairs)

    rows = []
    for a, b in pair_list:
        if a not in particle_properties or b not in particle_properties:
            print(f"[warning] skipping unknown pair: {(a, b)}")
            continue

        xa = pred_by_particle[a]
        xb = pred_by_particle[b]
        pa = reach_by_particle[a]
        pb = reach_by_particle[b]
        separation = np.abs(xa - xb)
        feasible = (pa >= min_reach_prob) & (pb >= min_reach_prob)

        if feasible.any():
            candidate_idx = np.where(feasible)[0]
            idx = candidate_idx[np.argmax(separation[candidate_idx])]
            feasibility_mode = "reach_constrained"
        else:
            score = separation * np.minimum(pa, pb)
            idx = int(np.argmax(score))
            feasibility_mode = "fallback_weighted_by_reach"

        theta_opt = float(theta_flat[idx])
        p_opt = float(p_flat[idx])
        frac_opt = float(frac_flat[idx])
        v_eq = float(estimate_voltage_from_pressure(theta_opt, p_opt, wide)[0])

        # Boundary flag matters because optimum-at-boundary is more sensitive.
        hit_p_boundary = bool(frac_opt <= 1e-9 or frac_opt >= 1.0 - 1e-9)
        if observed_angles_only:
            hit_theta_boundary = bool(
                theta_opt == float(envelope["theta_deg"].min())
                or theta_opt == float(envelope["theta_deg"].max())
            )
        else:
            hit_theta_boundary = bool(
                abs(theta_opt - float(envelope["theta_deg"].min())) < 1e-9
                or abs(theta_opt - float(envelope["theta_deg"].max())) < 1e-9
            )

        rows.append(
            {
                "particle_A": a,
                "particle_B": b,
                "theta_deg_opt": theta_opt,
                "Pmax_MPa_opt": p_opt,
                "equivalent_voltage_V_report_only": v_eq,
                "pressure_fraction_in_observed_envelope": frac_opt,
                "pred_xrel_A_um": float(xa[idx]),
                "pred_xrel_B_um": float(xb[idx]),
                "pred_separation_um": float(separation[idx]),
                "reach_prob_A": float(pa[idx]),
                "reach_prob_B": float(pb[idx]),
                "optimization_mode": feasibility_mode,
                "observed_angles_only": bool(observed_angles_only),
                "hit_pressure_boundary": hit_p_boundary,
                "hit_theta_boundary": hit_theta_boundary,
                "hit_any_boundary": bool(hit_p_boundary or hit_theta_boundary),
            }
        )

    return pd.DataFrame(rows).sort_values("pred_separation_um", ascending=False)


# -----------------------------------------------------------------------------
# NEW PARTICLES
# -----------------------------------------------------------------------------

def create_new_particle_template(path: Path) -> None:
    if path.exists():
        return
    pd.DataFrame(
        columns=[
            "name",
            "diameter_um",
            "density_kg_m3",
            "compressibility_1_pa",
        ]
    ).to_csv(path, index=False)


def load_new_particles(path: Path) -> Dict[str, Dict[str, float]]:
    if not path.exists() or path.stat().st_size == 0:
        return {}
    df = pd.read_csv(path)
    if df.empty:
        return {}

    required = {
        "name",
        "diameter_um",
        "density_kg_m3",
        "compressibility_1_pa",
    }
    if not required.issubset(df.columns):
        raise ValueError(
            f"{path.name} must contain columns: {sorted(required)}"
        )

    result = {}
    for _, r in df.iterrows():
        result[str(r["name"])] = {
            "diameter_um": float(r["diameter_um"]),
            "density_kg_m3": float(r["density_kg_m3"]),
            "compressibility_1_pa": float(r["compressibility_1_pa"]),
        }
    return result


def particle_domain_flag(
    props: Dict[str, float], known_props: Dict[str, Dict[str, float]]
) -> Tuple[bool, str]:
    known = pd.DataFrame(known_props).T
    checks = []
    for key in ["diameter_um", "density_kg_m3", "compressibility_1_pa"]:
        value = float(props[key])
        lo = float(known[key].min())
        hi = float(known[key].max())
        checks.append(lo <= value <= hi)
    inside = bool(all(checks))
    return inside, ("inside_training_property_range" if inside else "PROPERTY_EXTRAPOLATION")


def predict_new_particles_grid(
    new_particles: Dict[str, Dict[str, float]],
    known_props: Dict[str, Dict[str, float]],
    regressor: object,
    reach_classifier: object,
    features: Sequence[str],
    wide: pd.DataFrame,
) -> pd.DataFrame:
    if not new_particles:
        return pd.DataFrame()

    envelope = pressure_envelope(wide)
    theta_grid = np.linspace(
        float(envelope["theta_deg"].min()),
        float(envelope["theta_deg"].max()),
        41,
    )
    fraction_grid = np.linspace(0.0, 1.0, 61)
    T, F = np.meshgrid(theta_grid, fraction_grid, indexing="ij")
    theta_flat = T.ravel()
    frac_flat = F.ravel()
    p_flat = pressure_from_fraction(theta_flat, frac_flat, envelope)

    freq = float(wide["frequency_MHz"].median())
    vin = float(wide["inlet_velocity_mm_s"].median())

    blocks = []
    for name, props in new_particles.items():
        pred, reach = predict_particle(
            regressor,
            reach_classifier,
            features,
            theta_flat,
            p_flat,
            freq,
            vin,
            name,
            props,
        )
        inside, flag = particle_domain_flag(props, known_props)
        blocks.append(
            pd.DataFrame(
                {
                    "particle": name,
                    "theta_deg": theta_flat,
                    "Pmax_MPa": p_flat,
                    "pressure_fraction": frac_flat,
                    "pred_x_rel_um": pred,
                    "pred_x_absolute_um": pred + x_wall_um(theta_flat),
                    "pred_reach_probability": reach,
                    "property_domain_flag": flag,
                    "properties_inside_training_range": inside,
                    "diameter_um": props["diameter_um"],
                    "density_kg_m3": props["density_kg_m3"],
                    "compressibility_1_pa": props["compressibility_1_pa"],
                }
            )
        )
    return pd.concat(blocks, ignore_index=True)


# -----------------------------------------------------------------------------
# PLOTS / OUTPUT
# -----------------------------------------------------------------------------

def save_model_comparison_plot(metrics: pd.DataFrame, out_path: Path) -> None:
    pivot = metrics.pivot(index="model", columns="protocol", values="RMSE_um")
    ax = pivot.plot(kind="bar", figsize=(12, 6))
    ax.set_ylabel("RMSE [um]")
    ax.set_title("Model comparison across validation protocols")
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def save_parity_plot(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    out_path: Path,
) -> None:
    mask = np.isfinite(y_pred)
    yt = y_true[mask]
    yp = y_pred[mask]
    lo = min(float(np.min(yt)), float(np.min(yp)))
    hi = max(float(np.max(yt)), float(np.max(yp)))

    plt.figure(figsize=(6.5, 6.5))
    plt.scatter(yt, yp, alpha=0.7)
    plt.plot([lo, hi], [lo, hi], "--")
    plt.xlabel("True x_rel [um]")
    plt.ylabel("Predicted x_rel [um]")
    plt.title(f"OOF parity - {model_name} (run-group CV)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def save_feature_importance(
    model: object,
    features: Sequence[str],
    out_csv: Path,
    out_png: Path,
) -> None:
    importances = None

    # Direct tree/CatBoost models
    if hasattr(model, "feature_importances_"):
        importances = np.asarray(model.feature_importances_, dtype=float)
    elif hasattr(model, "get_feature_importance"):
        importances = np.asarray(model.get_feature_importance(), dtype=float)

    if importances is None or len(importances) != len(features):
        return

    fi = pd.DataFrame(
        {"feature": list(features), "importance": importances}
    ).sort_values("importance", ascending=False)
    fi.to_csv(out_csv, index=False)

    plt.figure(figsize=(8, 6))
    plt.barh(fi["feature"][::-1], fi["importance"][::-1])
    plt.xlabel("Importance")
    plt.title("Selected model feature importance")
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()



# -----------------------------------------------------------------------------
# REPORTING SUITE - TABLES + FIGURES SIMILAR TO THE PREVIOUS RESULTS CHAPTER
# -----------------------------------------------------------------------------

KEY_REPORT_PAIRS: List[Tuple[str, str]] = [
    ("PS_r2", "PS_r5"),
    ("WBC", "MCF7"),
]


def ensure_report_dirs(out_dir: Path) -> Dict[str, Path]:
    dirs = {
        "root": out_dir,
        "data": out_dir / "00_data_quality",
        "physical": out_dir / "01_physical_results",
        "models": out_dir / "02_model_results",
        "reach": out_dir / "03_reachability",
        "optimization": out_dir / "04_optimization",
        "new": out_dir / "05_new_particles",
    }
    for p in dirs.values():
        p.mkdir(parents=True, exist_ok=True)
    return dirs


def save_dataframe_csv(df: pd.DataFrame, path: Path, digits: Optional[int] = None) -> None:
    out = df.copy()
    if digits is not None:
        num_cols = out.select_dtypes(include=[np.number]).columns
        out[num_cols] = out[num_cols].round(digits)
    out.to_csv(path, index=False)


def model_comparison_table(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model, g in metrics.groupby("model"):
        rec: Dict[str, float | str] = {"model": model}
        for protocol, prefix in [
            ("run_cv", "Run_CV"),
            ("angle_cv", "Leave_One_Angle_Out"),
            ("particle_cv", "Leave_One_Particle_Out"),
        ]:
            s = g[g["protocol"] == protocol]
            if not s.empty:
                r = s.iloc[0]
                rec[f"{prefix}_R2"] = float(r["R2"])
                rec[f"{prefix}_MAE_um"] = float(r["MAE_um"])
                rec[f"{prefix}_RMSE_um"] = float(r["RMSE_um"])
        vals = [
            rec.get("Run_CV_RMSE_um", np.nan),
            rec.get("Leave_One_Angle_Out_RMSE_um", np.nan),
            rec.get("Leave_One_Particle_Out_RMSE_um", np.nan),
        ]
        rec["Composite_RMSE_um"] = float(np.nanmean(vals))
        rows.append(rec)
    return pd.DataFrame(rows).sort_values("Composite_RMSE_um").reset_index(drop=True)


def selected_model_angle_cv_predictions(
    model: object,
    reg_df: pd.DataFrame,
    features: Sequence[str],
) -> pd.DataFrame:
    X = reg_df[list(features)].reset_index(drop=True)
    y = reg_df["x_rel_um"].reset_index(drop=True)
    groups = reg_df["theta_deg"].astype(str).to_numpy()
    pred = np.full(len(reg_df), np.nan, dtype=float)
    logo = LeaveOneGroupOut()
    for tr, te in logo.split(X, y, groups):
        m = clone(model)
        m.fit(X.iloc[tr], y.iloc[tr])
        pred[te] = m.predict(X.iloc[te])

    out = reg_df[
        ["run_id", "theta_deg", "Pmax_MPa", "particle", "x_rel_um"]
    ].reset_index(drop=True).copy()
    out["pred_x_rel_um_angle_cv"] = pred
    return out


def metrics_by_group(
    frame: pd.DataFrame,
    group_col: str,
    true_col: str,
    pred_col: str,
) -> pd.DataFrame:
    rows = []
    for group, g in frame.groupby(group_col):
        mask = g[true_col].notna() & g[pred_col].notna()
        if mask.sum() < 2:
            continue
        yt = g.loc[mask, true_col].to_numpy(float)
        yp = g.loc[mask, pred_col].to_numpy(float)
        md = metric_dict(yt, yp)
        rows.append({group_col: group, "N": int(mask.sum()), **md})
    return pd.DataFrame(rows)


def pair_cv_metrics_from_oof(oof: pd.DataFrame) -> pd.DataFrame:
    """
    Compute pair-separation validation directly from run-group OOF predictions.
    This is crucial: a model can predict each particle reasonably but still be
    weak for the difference between a specific pair.
    """
    true_w = oof.pivot_table(
        index="run_id", columns="particle", values="x_rel_um", aggfunc="first"
    )
    pred_w = oof.pivot_table(
        index="run_id", columns="particle", values="pred_x_rel_um_oof", aggfunc="first"
    )

    particles = sorted(set(true_w.columns) & set(pred_w.columns))
    rows = []
    for a, b in combinations(particles, 2):
        mask = (
            true_w[a].notna()
            & true_w[b].notna()
            & pred_w[a].notna()
            & pred_w[b].notna()
        )
        if mask.sum() < 3:
            continue

        true_sep = np.abs(true_w.loc[mask, a] - true_w.loc[mask, b]).to_numpy(float)
        pred_sep = np.abs(pred_w.loc[mask, a] - pred_w.loc[mask, b]).to_numpy(float)
        md = metric_dict(true_sep, pred_sep)
        p90 = float(np.percentile(np.abs(true_sep - pred_sep), 90))
        span = float(np.ptp(true_sep))
        nrmse = float(md["RMSE_um"] / span) if span > 1e-12 else np.nan

        if md["R2"] >= 0.70 and (np.isnan(nrmse) or nrmse <= 0.30):
            confidence = "HIGH"
        elif md["R2"] >= 0.30 and (np.isnan(nrmse) or nrmse <= 0.60):
            confidence = "MODERATE"
        else:
            confidence = "LOW"

        rows.append(
            {
                "particle_A": a,
                "particle_B": b,
                "N_common_runs": int(mask.sum()),
                "Pair_CV_R2": float(md["R2"]),
                "Pair_CV_MAE_um": float(md["MAE_um"]),
                "Pair_CV_RMSE_um": float(md["RMSE_um"]),
                "Pair_CV_P90_abs_error_um": p90,
                "Pair_CV_normalized_RMSE": nrmse,
                "surrogate_pair_confidence": confidence,
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["surrogate_pair_confidence", "Pair_CV_RMSE_um"],
        ascending=[True, True],
    )


def observed_pair_best_table(
    wide: pd.DataFrame,
    long_df: pd.DataFrame,
) -> pd.DataFrame:
    valid = long_df[long_df["reached"].eq(1) & long_df["x_rel_um"].notna()]
    xw = valid.pivot_table(index="run_id", columns="particle", values="x_rel_um", aggfunc="first")
    meta_cols = ["run_id", "theta_deg", "Pmax_MPa"]
    if "V_in_V" in wide.columns:
        meta_cols.append("V_in_V")
    meta = wide[meta_cols].drop_duplicates("run_id").set_index("run_id")

    rows = []
    particles = list(xw.columns)
    for a, b in combinations(particles, 2):
        mask = xw[a].notna() & xw[b].notna()
        if not mask.any():
            continue
        sep = (xw.loc[mask, a] - xw.loc[mask, b]).abs()
        run_id = str(sep.idxmax())
        rows.append(
            {
                "particle_A": a,
                "particle_B": b,
                "best_observed_run_id": run_id,
                "theta_deg_observed": float(meta.loc[run_id, "theta_deg"]),
                "Pmax_MPa_observed": float(meta.loc[run_id, "Pmax_MPa"]),
                "V_in_V_observed": (
                    float(meta.loc[run_id, "V_in_V"]) if "V_in_V" in meta.columns else np.nan
                ),
                "observed_xrel_A_um": float(xw.loc[run_id, a]),
                "observed_xrel_B_um": float(xw.loc[run_id, b]),
                "max_observed_separation_um": float(sep.loc[run_id]),
            }
        )
    return pd.DataFrame(rows).sort_values("max_observed_separation_um", ascending=False)


def merge_optimization_validation(
    opt: pd.DataFrame,
    pair_cv: pd.DataFrame,
    observed_best: pd.DataFrame,
) -> pd.DataFrame:
    out = opt.copy()

    # canonical key allows A-B or B-A matching
    def key_cols(df: pd.DataFrame) -> pd.Series:
        return df.apply(lambda r: "||".join(sorted([str(r["particle_A"]), str(r["particle_B"])])), axis=1)

    out["_pair_key"] = key_cols(out)
    pcv = pair_cv.copy()
    obs = observed_best.copy()
    if not pcv.empty:
        pcv["_pair_key"] = key_cols(pcv)
        keep = [c for c in pcv.columns if c not in {"particle_A", "particle_B"}]
        out = out.merge(pcv[keep], on="_pair_key", how="left")
    if not obs.empty:
        obs["_pair_key"] = key_cols(obs)
        keep = [c for c in obs.columns if c not in {"particle_A", "particle_B"}]
        out = out.merge(obs[keep], on="_pair_key", how="left")

    out["predicted_minus_best_observed_um"] = (
        out["pred_separation_um"] - out["max_observed_separation_um"]
    )
    out["needs_direct_COMSOL_confirmation"] = True

    # More conservative status when optimum hits boundary.
    def status(row: pd.Series) -> str:
        conf = str(row.get("surrogate_pair_confidence", "UNKNOWN"))
        boundary = bool(row.get("hit_any_boundary", False))
        if conf == "HIGH" and not boundary:
            return "HIGH_CONFIDENCE_INTERIOR"
        if conf in {"HIGH", "MODERATE"} and boundary:
            return "BOUNDARY_CONFIRM_WITH_COMSOL"
        if conf == "MODERATE":
            return "MODERATE_CONFIDENCE"
        return "LOW_CONFIDENCE_SURROGATE"

    out["result_status"] = out.apply(status, axis=1)
    return out.drop(columns=["_pair_key"], errors="ignore")


def plot_pressure_vs_voltage(wide: pd.DataFrame, path: Path) -> None:
    if "V_in_V" not in wide.columns:
        return
    plt.figure(figsize=(8.5, 5.5))
    for theta, g in wide.groupby("theta_deg"):
        g = g.sort_values("V_in_V")
        plt.plot(g["V_in_V"], g["Pmax_MPa"], marker="o", label=f"{theta:g} deg")
    plt.xlabel("Drive voltage [V] — reporting only, not an ML feature")
    plt.ylabel("Pmax [MPa]")
    plt.title("Acoustic pressure versus electrical drive")
    plt.legend(title="theta")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_displacement_vs_angle_selected_drive(
    wide: pd.DataFrame,
    long_df: pd.DataFrame,
    path: Path,
) -> None:
    if "V_in_V" not in wide.columns:
        return
    available = sorted(wide["V_in_V"].dropna().unique())
    targets = [15.0, 60.0, 110.0]
    selected = []
    for t in targets:
        if available:
            selected.append(min(available, key=lambda x: abs(float(x) - t)))
    selected = list(dict.fromkeys(selected))

    fig, axes = plt.subplots(1, len(selected), figsize=(6.2 * len(selected), 5.2), sharey=True)
    if len(selected) == 1:
        axes = [axes]

    v_by_run = wide.set_index("run_id")["V_in_V"].to_dict()
    df = long_df[long_df["reached"].eq(1)].copy()
    df["V_in_V"] = df["run_id"].map(v_by_run)

    for ax, v in zip(axes, selected):
        sub = df[np.isclose(df["V_in_V"].astype(float), float(v))]
        for particle, g in sub.groupby("particle"):
            g = g.sort_values("theta_deg")
            ax.plot(g["theta_deg"], g["x_rel_um"], marker="o", label=particle)
        ax.set_title(f"Drive = {v:g} V")
        ax.set_xlabel("theta [deg]")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("x - x_wall [um]")
    axes[-1].legend(fontsize=8, loc="best")
    fig.suptitle("Direct COMSOL displacement versus taSSAW angle")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_displacement_vs_pressure_particles(long_df: pd.DataFrame, path: Path) -> None:
    particles = [p for p in PARTICLE_PROPERTIES if p in set(long_df["particle"])]
    n = len(particles)
    ncols = 2
    nrows = int(math.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 3.8 * nrows), sharex=False)
    axes = np.atleast_1d(axes).ravel()

    valid = long_df[long_df["reached"].eq(1)]
    for ax, particle in zip(axes, particles):
        sub = valid[valid["particle"] == particle]
        for theta, g in sub.groupby("theta_deg"):
            g = g.sort_values("Pmax_MPa")
            ax.plot(g["Pmax_MPa"], g["x_rel_um"], marker="o", label=f"{theta:g} deg")
        ax.set_title(particle)
        ax.set_xlabel("Pmax [MPa]")
        ax.set_ylabel("x - x_wall [um]")
        ax.grid(alpha=0.25)
    for ax in axes[n:]:
        ax.axis("off")
    if n > 0:
        axes[0].legend(fontsize=7, title="theta")
    fig.suptitle("Direct COMSOL displacement versus acoustic pressure")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def _pair_observed_frame(long_df: pd.DataFrame, wide: pd.DataFrame, a: str, b: str) -> pd.DataFrame:
    valid = long_df[long_df["reached"].eq(1)]
    p = valid[valid["particle"].isin([a, b])].pivot_table(
        index="run_id", columns="particle", values="x_rel_um", aggfunc="first"
    )
    if a not in p.columns or b not in p.columns:
        return pd.DataFrame()
    p = p.dropna(subset=[a, b]).copy()
    p["separation_um"] = (p[a] - p[b]).abs()

    cols = ["run_id", "theta_deg", "Pmax_MPa"]
    if "V_in_V" in wide.columns:
        cols.append("V_in_V")
    meta = wide[cols].drop_duplicates("run_id").set_index("run_id")
    p = p.join(meta, how="left").reset_index()
    return p


def plot_pair_heatmap_observed(
    long_df: pd.DataFrame,
    wide: pd.DataFrame,
    a: str,
    b: str,
    path: Path,
) -> None:
    pf = _pair_observed_frame(long_df, wide, a, b)
    if pf.empty:
        return

    xcol = "V_in_V" if "V_in_V" in pf.columns else "Pmax_MPa"
    mat = pf.pivot_table(index="theta_deg", columns=xcol, values="separation_um", aggfunc="first")
    mat = mat.sort_index().sort_index(axis=1)

    plt.figure(figsize=(10, 5.5))
    im = plt.imshow(mat.to_numpy(), aspect="auto", origin="lower")
    plt.colorbar(im, label="Observed separation [um]")
    plt.xticks(range(len(mat.columns)), [f"{x:g}" for x in mat.columns], rotation=45)
    plt.yticks(range(len(mat.index)), [f"{x:g}" for x in mat.index])
    plt.xlabel("Drive voltage [V]" if xcol == "V_in_V" else "Pmax [MPa]")
    plt.ylabel("theta [deg]")
    plt.title(f"Observed separation map: {a} - {b}")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_pair_pressure_observed(
    long_df: pd.DataFrame,
    wide: pd.DataFrame,
    a: str,
    b: str,
    path: Path,
) -> None:
    pf = _pair_observed_frame(long_df, wide, a, b)
    if pf.empty:
        return
    plt.figure(figsize=(8.5, 5.5))
    for theta, g in pf.groupby("theta_deg"):
        g = g.sort_values("Pmax_MPa")
        plt.plot(g["Pmax_MPa"], g["separation_um"], marker="o", label=f"{theta:g} deg")
    plt.xlabel("Pmax [MPa]")
    plt.ylabel("Observed separation [um]")
    plt.title(f"Direct COMSOL pair separation: {a} - {b}")
    plt.legend(title="theta")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_observed_max_pair_matrix(observed_best: pd.DataFrame, path: Path) -> None:
    particles = list(PARTICLE_PROPERTIES.keys())
    mat = pd.DataFrame(np.nan, index=particles, columns=particles)
    for i in range(len(particles)):
        mat.iat[i, i] = 0.0
    for _, r in observed_best.iterrows():
        a, b = r["particle_A"], r["particle_B"]
        v = float(r["max_observed_separation_um"])
        if a in mat.index and b in mat.columns:
            mat.loc[a, b] = v
            mat.loc[b, a] = v

    plt.figure(figsize=(8.5, 7.2))
    im = plt.imshow(mat.to_numpy(float), aspect="equal")
    plt.colorbar(im, label="Maximum observed separation [um]")
    plt.xticks(range(len(particles)), particles, rotation=45, ha="right")
    plt.yticks(range(len(particles)), particles)
    plt.title("Maximum observed pairwise separation across COMSOL runs")
    for i in range(len(particles)):
        for j in range(len(particles)):
            val = mat.iloc[i, j]
            if np.isfinite(val):
                plt.text(j, i, f"{val:.0f}", ha="center", va="center", fontsize=7)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_best_observed_run_positions(
    long_df: pd.DataFrame,
    wide: pd.DataFrame,
    path: Path,
) -> None:
    valid = long_df[long_df["reached"].eq(1)].copy()
    spread = valid.groupby("run_id")["x_rel_um"].agg(lambda s: float(s.max() - s.min()))
    if spread.empty:
        return
    best_run = str(spread.idxmax())
    g = valid[valid["run_id"] == best_run].sort_values("x_rel_um")
    meta = wide[wide["run_id"].astype(str) == best_run].iloc[0]

    plt.figure(figsize=(9, 5.5))
    y = np.arange(len(g))
    plt.scatter(g["x_rel_um"], y, s=55)
    plt.yticks(y, g["particle"])
    plt.xlabel("x - x_wall [um]")
    title = f"Outlet positions — run {best_run} | theta={float(meta['theta_deg']):g} deg, Pmax={float(meta['Pmax_MPa']):.3f} MPa"
    if "V_in_V" in meta.index:
        title += f", V={float(meta['V_in_V']):g}"
    plt.title(title)
    plt.grid(axis="x", alpha=0.25)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_model_comparison_rich(table: pd.DataFrame, path: Path) -> None:
    cols = [
        "Run_CV_RMSE_um",
        "Leave_One_Angle_Out_RMSE_um",
        "Leave_One_Particle_Out_RMSE_um",
    ]
    existing = [c for c in cols if c in table.columns]
    p = table.set_index("model")[existing]
    ax = p.plot(kind="bar", figsize=(12.5, 6.2))
    ax.set_ylabel("RMSE [um]")
    ax.set_title("Regression model comparison across validation protocols")
    ax.grid(axis="y", alpha=0.25)
    plt.xticks(rotation=40, ha="right")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_selected_model_residuals(oof: pd.DataFrame, model_name: str, path: Path) -> None:
    g = oof.dropna(subset=["pred_x_rel_um_oof", "residual_um"])
    plt.figure(figsize=(8, 5.5))
    plt.scatter(g["pred_x_rel_um_oof"], g["residual_um"], alpha=0.65)
    plt.axhline(0.0, linestyle="--")
    plt.xlabel("OOF predicted x - x_wall [um]")
    plt.ylabel("Residual = true - predicted [um]")
    plt.title(f"OOF residuals — {model_name}")
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_group_rmse(table: pd.DataFrame, group_col: str, title: str, path: Path) -> None:
    if table.empty:
        return
    g = table.sort_values("RMSE_um")
    plt.figure(figsize=(9, 5.2))
    plt.bar(g[group_col].astype(str), g["RMSE_um"])
    plt.ylabel("RMSE [um]")
    plt.xlabel(group_col)
    plt.title(title)
    plt.xticks(rotation=35, ha="right")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def reachability_oof_predictions(
    long_df: pd.DataFrame,
    features: Sequence[str],
    classifier: HistGradientBoostingClassifier,
) -> pd.DataFrame:
    X = long_df[list(features)].reset_index(drop=True)
    y = long_df["reached"].astype(int).reset_index(drop=True)
    groups = long_df["run_id"].astype(str).to_numpy()
    splits = list(GroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE).split(X, y, groups))
    prob = np.full(len(y), np.nan)
    for tr, te in splits:
        m = clone(classifier)
        m.fit(X.iloc[tr], y.iloc[tr])
        prob[te] = m.predict_proba(X.iloc[te])[:, 1]

    out = long_df[["run_id", "theta_deg", "Pmax_MPa", "particle", "reached"]].reset_index(drop=True).copy()
    out["pred_reach_probability_oof"] = prob
    out["pred_reached_oof"] = (prob >= 0.5).astype(float)
    return out


def plot_reachability_by_particle(reach_oof: pd.DataFrame, path: Path) -> None:
    rows = []
    for p, g in reach_oof.groupby("particle"):
        if g["pred_reach_probability_oof"].notna().sum() == 0:
            continue
        rows.append({
            "particle": p,
            "actual_reach_rate": float(g["reached"].mean()),
            "mean_predicted_reach_probability": float(g["pred_reach_probability_oof"].mean()),
        })
    if not rows:
        return
    d = pd.DataFrame(rows).sort_values("particle")
    x = np.arange(len(d))
    w = 0.38
    plt.figure(figsize=(10, 5.4))
    plt.bar(x - w/2, d["actual_reach_rate"], width=w, label="Actual")
    plt.bar(x + w/2, d["mean_predicted_reach_probability"], width=w, label="OOF predicted")
    plt.xticks(x, d["particle"], rotation=35, ha="right")
    plt.ylim(0, 1.05)
    plt.ylabel("Reach fraction / probability")
    plt.title("Reachability by particle")
    plt.legend()
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def pair_surface_predictions(
    regressor: object,
    reach_classifier: object,
    features: Sequence[str],
    wide: pd.DataFrame,
    a: str,
    b: str,
    n_p: int = 161,
) -> pd.DataFrame:
    envelope = pressure_envelope(wide).sort_values("theta_deg")
    freq = float(wide["frequency_MHz"].median())
    vin = float(wide["inlet_velocity_mm_s"].median())
    blocks = []
    for _, row in envelope.iterrows():
        theta = float(row["theta_deg"])
        p = np.linspace(float(row["min"]), float(row["max"]), n_p)
        t = np.full_like(p, theta)
        xa, ra = predict_particle(
            regressor, reach_classifier, features, t, p, freq, vin, a, PARTICLE_PROPERTIES[a]
        )
        xb, rb = predict_particle(
            regressor, reach_classifier, features, t, p, freq, vin, b, PARTICLE_PROPERTIES[b]
        )
        blocks.append(pd.DataFrame({
            "theta_deg": t,
            "Pmax_MPa": p,
            "pred_xrel_A_um": xa,
            "pred_xrel_B_um": xb,
            "pred_separation_um": np.abs(xa - xb),
            "reach_prob_A": ra,
            "reach_prob_B": rb,
            "feasible": (ra >= MIN_REACH_PROB) & (rb >= MIN_REACH_PROB),
        }))
    return pd.concat(blocks, ignore_index=True)


def plot_pair_optimization_surface(
    surface: pd.DataFrame,
    opt_row: pd.Series,
    a: str,
    b: str,
    path: Path,
) -> None:
    plt.figure(figsize=(8.7, 5.8))
    for theta, g in surface.groupby("theta_deg"):
        g = g.sort_values("Pmax_MPa")
        y = g["pred_separation_um"].where(g["feasible"], np.nan)
        plt.plot(g["Pmax_MPa"], y, label=f"{theta:g} deg")
    plt.scatter(
        [float(opt_row["Pmax_MPa_opt"])],
        [float(opt_row["pred_separation_um"])],
        marker="x",
        s=90,
        label="Selected optimum",
    )
    plt.xlabel("Pmax [MPa]")
    plt.ylabel("Predicted separation [um]")
    plt.title(f"Pair optimization surface slice: {a} - {b}")
    plt.legend(title="theta")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_optimized_pair_matrix(opt_table: pd.DataFrame, path: Path) -> None:
    particles = list(PARTICLE_PROPERTIES.keys())
    mat = pd.DataFrame(np.nan, index=particles, columns=particles)
    for i in range(len(particles)):
        mat.iat[i, i] = 0.0
    for _, r in opt_table.iterrows():
        a, b = r["particle_A"], r["particle_B"]
        if a in mat.index and b in mat.columns:
            v = float(r["pred_separation_um"])
            mat.loc[a, b] = v
            mat.loc[b, a] = v
    plt.figure(figsize=(8.5, 7.2))
    im = plt.imshow(mat.to_numpy(float), aspect="equal")
    plt.colorbar(im, label="Predicted optimized separation [um]")
    plt.xticks(range(len(particles)), particles, rotation=45, ha="right")
    plt.yticks(range(len(particles)), particles)
    plt.title("Predicted optimum separation matrix")
    for i in range(len(particles)):
        for j in range(len(particles)):
            val = mat.iloc[i, j]
            if np.isfinite(val):
                plt.text(j, i, f"{val:.0f}", ha="center", va="center", fontsize=7)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_new_particle_prediction_grid(new_pred: pd.DataFrame, new_dir: Path) -> None:
    if new_pred.empty:
        return
    for name, sub in new_pred.groupby("particle"):
        plt.figure(figsize=(8.6, 5.6))
        for theta, g in sub.groupby("theta_deg"):
            g = g.sort_values("Pmax_MPa")
            y = g["pred_x_rel_um"].where(g["pred_reach_probability"] >= MIN_REACH_PROB, np.nan)
            plt.plot(g["Pmax_MPa"], y, label=f"{theta:g} deg")
        plt.xlabel("Pmax [MPa]")
        plt.ylabel("Predicted x - x_wall [um]")
        plt.title(f"New particle prediction: {name}")
        plt.legend(title="theta", fontsize=8)
        plt.grid(alpha=0.25)
        plt.tight_layout()
        safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in str(name))
        plt.savefig(new_dir / f"fig_new_particle_{safe}.png", dpi=200)
        plt.close()


def html_table(df: pd.DataFrame, max_rows: int = 30) -> str:
    if df is None or df.empty:
        return "<p>No data.</p>"
    show = df.head(max_rows).copy()
    num_cols = show.select_dtypes(include=[np.number]).columns
    show[num_cols] = show[num_cols].round(4)
    return show.to_html(index=False, border=0, classes="dataframe")


def write_html_report(
    out_dir: Path,
    best_name: str,
    summary: Dict[str, object],
    model_table: pd.DataFrame,
    per_particle: pd.DataFrame,
    per_angle: pd.DataFrame,
    pair_opt: pd.DataFrame,
    pair_cv: pd.DataFrame,
    reach_metrics: pd.DataFrame,
) -> None:
    rel = lambda p: str(Path(p).relative_to(out_dir)).replace("\\", "/")

    image_paths = [
        ("Pressure vs voltage", out_dir / "00_data_quality" / "fig_pressure_vs_voltage.png"),
        ("Displacement vs angle", out_dir / "01_physical_results" / "fig_displacement_vs_angle_selected_drive.png"),
        ("Displacement vs pressure", out_dir / "01_physical_results" / "fig_displacement_vs_pressure_particles.png"),
        ("Observed max pair matrix", out_dir / "01_physical_results" / "fig_max_observed_pair_separation_matrix.png"),
        ("Model comparison", out_dir / "02_model_results" / "fig_model_comparison_rmse.png"),
        ("OOF parity", out_dir / "02_model_results" / "fig_selected_model_oof_parity.png"),
        ("OOF residuals", out_dir / "02_model_results" / "fig_selected_model_residuals.png"),
        ("RMSE by particle", out_dir / "02_model_results" / "fig_selected_model_rmse_by_particle.png"),
        ("RMSE by left-out angle", out_dir / "02_model_results" / "fig_selected_model_rmse_by_angle.png"),
        ("Reachability", out_dir / "03_reachability" / "fig_reachability_by_particle.png"),
        ("Optimized pair matrix", out_dir / "04_optimization" / "fig_predicted_optimized_pair_matrix.png"),
    ]
    for a, b in KEY_REPORT_PAIRS:
        image_paths.extend([
            (f"Observed heatmap {a}-{b}", out_dir / "01_physical_results" / f"fig_observed_heatmap_{a}_{b}.png"),
            (f"Observed separation vs pressure {a}-{b}", out_dir / "01_physical_results" / f"fig_observed_pair_pressure_{a}_{b}.png"),
            (f"Optimization surface {a}-{b}", out_dir / "04_optimization" / f"fig_optimization_surface_{a}_{b}.png"),
        ])

    cards = []
    for title, p in image_paths:
        if p.exists():
            cards.append(
                f'<div class="card"><h3>{html.escape(title)}</h3>'
                f'<img src="{html.escape(rel(p))}" alt="{html.escape(title)}"></div>'
            )

    summary_items = "".join(
        f"<li><b>{html.escape(str(k))}:</b> {html.escape(str(v))}</li>"
        for k, v in summary.items()
    )

    doc = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Particle ML Results Report</title>
<style>
body{{font-family:Arial,sans-serif;max-width:1500px;margin:30px auto;padding:0 20px;line-height:1.5;color:#222}}
h1,h2{{margin-top:34px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(520px,1fr));gap:18px}}
.card{{border:1px solid #ddd;border-radius:10px;padding:14px;background:#fff}}
.card img{{width:100%;height:auto}}
.dataframe{{border-collapse:collapse;width:100%;font-size:13px;overflow:auto;display:block}}
.dataframe th,.dataframe td{{border:1px solid #ddd;padding:6px 8px;text-align:right;white-space:nowrap}}
.dataframe th{{background:#f2f2f2;position:sticky;top:0}}
.note{{background:#f7f7f7;border-left:4px solid #888;padding:10px 14px}}
</style>
</head>
<body>
<h1>Particle separation ML — Results dashboard</h1>
<p class="note">The regression model uses <b>Pmax_MPa</b>, not drive voltage. Voltage is retained only for descriptive plots and for converting an optimized pressure back to an approximate COMSOL drive setting.</p>
<h2>Executive summary</h2>
<ul>{summary_items}</ul>

<h2>Model comparison</h2>
{html_table(model_table, 20)}

<h2>Selected model error by particle</h2>
{html_table(per_particle, 20)}

<h2>Leave-one-angle-out error</h2>
{html_table(per_angle, 20)}

<h2>Pair-specific OOF validation</h2>
{html_table(pair_cv.sort_values("Pair_CV_RMSE_um"), 28)}

<h2>Optimization results</h2>
{html_table(pair_opt.sort_values("pred_separation_um", ascending=False), 28)}

<h2>Reachability</h2>
{html_table(reach_metrics, 10)}

<h2>Figures</h2>
<div class="grid">{''.join(cards)}</div>
</body>
</html>"""
    (out_dir / "RESULTS_REPORT.html").write_text(doc, encoding="utf-8")


def write_results_readme(out_dir: Path, best_name: str) -> None:
    text = f"""RESULTS OUTPUT GUIDE
====================

Selected regression model: {best_name}

Open first:
  RESULTS_REPORT.html

Folders:
  00_data_quality
    - dataset summary
    - P-V audit
    - Pmax versus voltage figure

  01_physical_results
    - direct COMSOL displacement versus angle
    - direct COMSOL displacement versus Pmax
    - observed pair separation heatmaps
    - observed pair separation versus Pmax
    - maximum observed pairwise separation matrix
    - best observed separation table

  02_model_results
    - model-comparison table with Run-CV, Leave-One-Angle-Out and Leave-One-Particle-Out
    - selected-model OOF parity plot
    - residual plot
    - error tables/figures by particle and angle
    - feature importance where available

  03_reachability
    - reachability metrics
    - OOF reachability predictions
    - reachability-by-particle figure

  04_optimization
    - pair-specific OOF validation
    - optimization results merged with confidence and observed best
    - pair optimization surface plots
    - optimized separation matrix

  05_new_particles
    - new_particles.csv template
    - prediction grid and plots after you add new particles

Important:
  * Pmax is the ML intensity feature.
  * Voltage is report-only / COMSOL-setting conversion.
  * By default optimization uses only observed theta values and P within each theta's observed envelope.
  * Any optimized point, especially one on a search boundary, should be re-run in COMSOL.
  * Current CSVs support final outlet-position prediction, not full time-resolved trajectory prediction.
"""
    (out_dir / "README_RESULTS.txt").write_text(text, encoding="utf-8")

# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-glob",
        default="run_results_master*.csv",
        help="CSV glob, e.g. '/path/run_results_master*.csv'",
    )
    parser.add_argument(
        "--output-dir",
        default="particle_ml_outputs",
        help="Directory for results",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Use fewer estimators/iterations for a quicker first run",
    )
    parser.add_argument(
        "--continuous-theta-optimization",
        action="store_true",
        help="Allow interpolation over continuous theta. Default is observed angles only.",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    dirs = ensure_report_dirs(out_dir)

    print("\n=== 1) Loading data ===")
    wide = load_wide_data(args.data_glob)
    long_df = make_long_dataset(wide)
    reg_df = long_df[(long_df["reached"] == 1) & long_df["x_rel_um"].notna()].copy()

    print(f"Wide runs: {len(wide)}")
    print(f"Long rows (run x particle): {len(long_df)}")
    print(f"Reached rows used for x_rel regression: {len(reg_df)}")
    print(f"Angles: {sorted(wide['theta_deg'].unique())}")

    # Root copies for compatibility + structured folder copies.
    long_df.to_csv(out_dir / "prepared_long_dataset.csv", index=False)
    long_df.to_csv(dirs["data"] / "prepared_long_dataset.csv", index=False)
    envelope = pressure_envelope(wide)
    envelope.to_csv(out_dir / "pressure_envelope_by_theta.csv", index=False)
    envelope.to_csv(dirs["data"] / "table_pressure_envelope_by_theta.csv", index=False)

    dataset_summary = pd.DataFrame([{
        "n_runs": len(wide),
        "n_particles": long_df["particle"].nunique(),
        "n_long_rows": len(long_df),
        "n_reached_rows": int(long_df["reached"].sum()),
        "n_not_reached_rows": int((1 - long_df["reached"]).sum()),
        "theta_min_deg": float(wide["theta_deg"].min()),
        "theta_max_deg": float(wide["theta_deg"].max()),
        "Pmax_min_MPa": float(wide["Pmax_MPa"].min()),
        "Pmax_max_MPa": float(wide["Pmax_MPa"].max()),
        "frequency_unique_count": int(wide["frequency_MHz"].nunique()),
        "inlet_velocity_unique_count": int(wide["inlet_velocity_mm_s"].nunique()),
    }])
    dataset_summary.to_csv(dirs["data"] / "table_dataset_summary.csv", index=False)

    print("\n=== 2) Auditing V -> P relationship (V is NOT used in model) ===")
    vp_audit = audit_pressure_voltage(wide)
    if not vp_audit.empty:
        vp_audit.to_csv(out_dir / "pressure_voltage_audit.csv", index=False)
        vp_audit.to_csv(dirs["data"] / "table_pressure_voltage_audit.csv", index=False)
        print(vp_audit.round(6).to_string(index=False))
        plot_pressure_vs_voltage(wide, dirs["data"] / "fig_pressure_vs_voltage.png")

    print("\n=== 3) Direct-data figures (same style of outputs as prior report) ===")
    observed_best = observed_pair_best_table(wide, long_df)
    observed_best.to_csv(dirs["physical"] / "table_best_observed_separation_all_pairs.csv", index=False)
    plot_displacement_vs_angle_selected_drive(
        wide, long_df, dirs["physical"] / "fig_displacement_vs_angle_selected_drive.png"
    )
    plot_displacement_vs_pressure_particles(
        long_df, dirs["physical"] / "fig_displacement_vs_pressure_particles.png"
    )
    for a, b in KEY_REPORT_PAIRS:
        if a in PARTICLE_PROPERTIES and b in PARTICLE_PROPERTIES:
            plot_pair_heatmap_observed(
                long_df, wide, a, b, dirs["physical"] / f"fig_observed_heatmap_{a}_{b}.png"
            )
            plot_pair_pressure_observed(
                long_df, wide, a, b, dirs["physical"] / f"fig_observed_pair_pressure_{a}_{b}.png"
            )
    plot_observed_max_pair_matrix(
        observed_best, dirs["physical"] / "fig_max_observed_pair_separation_matrix.png"
    )
    plot_best_observed_run_positions(
        long_df, wide, dirs["physical"] / "fig_outlet_positions_best_observed_run.png"
    )

    print("\n=== 4) Feature construction ===")
    features = choose_feature_columns(reg_df)
    print("Regression features:")
    for f in features:
        print(" -", f)
    if "frequency_MHz" not in features:
        print("[note] frequency_MHz is constant in current data -> not learnable independently.")
    if "inlet_velocity_mm_s" not in features:
        print("[note] inlet_velocity_mm_s is constant in current data -> not learnable independently.")
    print("[note] V_in_V is intentionally excluded; Pmax_MPa is used instead.")

    print("\n=== 5) Comparing regression models ===")
    models = build_regressors(fast_mode=args.fast)
    metrics, run_oof = evaluate_regressors(reg_df, features, models)
    metrics.to_csv(out_dir / "model_metrics_all_protocols.csv", index=False)
    metrics.to_csv(dirs["models"] / "table_model_metrics_long.csv", index=False)

    model_table = model_comparison_table(metrics)
    model_table.to_csv(dirs["models"] / "table_model_comparison.csv", index=False)

    best_name, ranking = select_model_name(metrics)
    ranking.to_csv(out_dir / "model_ranking.csv", index=False)
    ranking.to_csv(dirs["models"] / "table_model_ranking.csv", index=False)
    print("\nCombined ranking by mean RMSE across Run / Particle / Angle CV:")
    print(ranking.round(3).to_string(index=False))
    print(f"\nSELECTED MODEL: {best_name}")

    plot_model_comparison_rich(
        model_table, dirs["models"] / "fig_model_comparison_rmse.png"
    )
    # Backward-compatible filename.
    save_model_comparison_plot(metrics, out_dir / "model_comparison_rmse.png")

    best_model = clone(models[best_name])
    best_model.fit(reg_df[features], reg_df["x_rel_um"])

    # Run-group OOF predictions
    selected_oof = run_oof[best_name]
    oof_out = reg_df[
        ["run_id", "theta_deg", "Pmax_MPa", "particle", "x_rel_um"]
    ].reset_index(drop=True).copy()
    if "V_in_V" in wide.columns:
        vmap = wide.set_index("run_id")["V_in_V"].to_dict()
        oof_out["V_in_V_report_only"] = oof_out["run_id"].map(vmap)
    oof_out["pred_x_rel_um_oof"] = selected_oof
    oof_out["residual_um"] = oof_out["x_rel_um"] - oof_out["pred_x_rel_um_oof"]
    oof_out["abs_error_um"] = oof_out["residual_um"].abs()
    oof_out.to_csv(out_dir / "selected_model_oof_predictions.csv", index=False)
    oof_out.to_csv(dirs["models"] / "selected_model_oof_predictions.csv", index=False)

    save_parity_plot(
        reg_df["x_rel_um"].to_numpy(),
        selected_oof,
        best_name,
        dirs["models"] / "fig_selected_model_oof_parity.png",
    )
    save_parity_plot(
        reg_df["x_rel_um"].to_numpy(),
        selected_oof,
        best_name,
        out_dir / "selected_model_parity.png",
    )
    plot_selected_model_residuals(
        oof_out, best_name, dirs["models"] / "fig_selected_model_residuals.png"
    )
    save_feature_importance(
        best_model,
        features,
        dirs["models"] / "table_selected_model_feature_importance.csv",
        dirs["models"] / "fig_selected_model_feature_importance.png",
    )

    # Error by particle, directly from run-group OOF
    per_particle = metrics_by_group(
        oof_out, "particle", "x_rel_um", "pred_x_rel_um_oof"
    ).sort_values("RMSE_um")
    per_particle.to_csv(dirs["models"] / "table_selected_model_error_by_particle.csv", index=False)
    plot_group_rmse(
        per_particle, "particle",
        f"OOF RMSE by particle — {best_name}",
        dirs["models"] / "fig_selected_model_rmse_by_particle.png",
    )

    # Strict Leave-One-Angle-Out predictions and table
    angle_cv_out = selected_model_angle_cv_predictions(best_model, reg_df, features)
    angle_cv_out.to_csv(dirs["models"] / "selected_model_leave_one_angle_predictions.csv", index=False)
    per_angle = metrics_by_group(
        angle_cv_out, "theta_deg", "x_rel_um", "pred_x_rel_um_angle_cv"
    ).sort_values("theta_deg")
    per_angle.to_csv(dirs["models"] / "table_selected_model_leave_one_angle_out.csv", index=False)
    plot_group_rmse(
        per_angle, "theta_deg",
        f"Leave-one-angle-out RMSE — {best_name}",
        dirs["models"] / "fig_selected_model_rmse_by_angle.png",
    )

    print("\n=== 6) Reachability model ===")
    reach_clf = fit_reachability_classifier(long_df, features)
    reach_metrics = evaluate_reachability_classifier(long_df, features, reach_clf)
    reach_metrics.to_csv(out_dir / "reachability_metrics.csv", index=False)
    reach_metrics.to_csv(dirs["reach"] / "table_reachability_metrics.csv", index=False)
    reach_oof = reachability_oof_predictions(long_df, features, reach_clf)
    reach_oof.to_csv(dirs["reach"] / "reachability_oof_predictions.csv", index=False)
    plot_reachability_by_particle(
        reach_oof, dirs["reach"] / "fig_reachability_by_particle.png"
    )
    print(reach_metrics.round(4).to_string(index=False))

    print("\n=== 7) Pair-specific validation + optimization ===")
    pair_cv = pair_cv_metrics_from_oof(oof_out)
    pair_cv.to_csv(dirs["optimization"] / "table_pair_cv_metrics.csv", index=False)

    observed_angles_only = not args.continuous_theta_optimization
    pair_results = optimize_pairs(
        best_model,
        reach_clf,
        features,
        wide,
        PARTICLE_PROPERTIES,
        pairs=SPECIAL_PAIRS,
        min_reach_prob=MIN_REACH_PROB,
        observed_angles_only=observed_angles_only,
    )
    pair_results.to_csv(out_dir / "pair_optimization_results.csv", index=False)

    pair_results_full = merge_optimization_validation(
        pair_results, pair_cv, observed_best
    )
    pair_results_full.to_csv(
        dirs["optimization"] / "table_pair_optimization_results_with_confidence.csv",
        index=False,
    )
    pair_results_full.to_csv(out_dir / "pair_optimization_results_with_confidence.csv", index=False)
    print(pair_results_full.head(15).round(4).to_string(index=False))

    plot_optimized_pair_matrix(
        pair_results_full,
        dirs["optimization"] / "fig_predicted_optimized_pair_matrix.png",
    )

    # Always create the two report-style surface plots when those pairs exist.
    for a, b in KEY_REPORT_PAIRS:
        keymask = (
            ((pair_results_full["particle_A"] == a) & (pair_results_full["particle_B"] == b))
            | ((pair_results_full["particle_A"] == b) & (pair_results_full["particle_B"] == a))
        )
        if keymask.any():
            surface = pair_surface_predictions(
                best_model, reach_clf, features, wide, a, b
            )
            surface.to_csv(
                dirs["optimization"] / f"surface_predictions_{a}_{b}.csv",
                index=False,
            )
            opt_row = pair_results_full.loc[keymask].iloc[0]
            plot_pair_optimization_surface(
                surface, opt_row, a, b,
                dirs["optimization"] / f"fig_optimization_surface_{a}_{b}.png",
            )

    print("\n=== 8) New-particle prediction ===")
    new_particle_file = dirs["new"] / "new_particles.csv"
    create_new_particle_template(new_particle_file)
    # compatibility copy at root
    create_new_particle_template(out_dir / "new_particles.csv")

    # Prefer the file inside 05_new_particles if user filled it.
    new_particles = load_new_particles(new_particle_file)
    if not new_particles:
        root_new = load_new_particles(out_dir / "new_particles.csv")
        new_particles = root_new

    if new_particles:
        new_pred = predict_new_particles_grid(
            new_particles,
            PARTICLE_PROPERTIES,
            best_model,
            reach_clf,
            features,
            wide,
        )
        new_pred.to_csv(dirs["new"] / "new_particle_prediction_grid.csv", index=False)
        new_pred.to_csv(out_dir / "new_particle_prediction_grid.csv", index=False)
        plot_new_particle_prediction_grid(new_pred, dirs["new"])
        print(f"Predictions saved for {len(new_particles)} new particle(s).")

        combined_props = {**PARTICLE_PROPERTIES, **new_particles}
        combined_opt = optimize_pairs(
            best_model,
            reach_clf,
            features,
            wide,
            combined_props,
            pairs=SPECIAL_PAIRS,
            min_reach_prob=MIN_REACH_PROB,
            observed_angles_only=observed_angles_only,
        )
        combined_opt.to_csv(
            dirs["new"] / "pair_optimization_including_new_particles.csv", index=False
        )
    else:
        print(
            f"Template created: {new_particle_file}\n"
            "Fill name, diameter_um, density_kg_m3, compressibility_1_pa and rerun."
        )

    print("\n=== 9) Saving models / report ===")
    joblib.dump(best_model, out_dir / "selected_regression_model.joblib")
    joblib.dump(reach_clf, out_dir / "reachability_model.joblib")

    selected_run_metrics = metrics[
        (metrics["model"] == best_name) & (metrics["protocol"] == "run_cv")
    ].iloc[0]
    selected_angle_metrics = metrics[
        (metrics["model"] == best_name) & (metrics["protocol"] == "angle_cv")
    ].iloc[0]
    selected_particle_metrics = metrics[
        (metrics["model"] == best_name) & (metrics["protocol"] == "particle_cv")
    ].iloc[0]

    summary = {
        "Selected model": best_name,
        "Target": "x_rel = x_particle - x_wall",
        "X_wall formula": "(9.9 - 9*sin(theta)) mm",
        "Run-CV RMSE [um]": round(float(selected_run_metrics["RMSE_um"]), 3),
        "Leave-One-Angle-Out RMSE [um]": round(float(selected_angle_metrics["RMSE_um"]), 3),
        "Leave-One-Particle-Out RMSE [um]": round(float(selected_particle_metrics["RMSE_um"]), 3),
        "Optimization theta policy": "observed angles only" if observed_angles_only else "continuous interpolation",
        "Optimization intensity variable": "Pmax_MPa",
        "Reach probability threshold": MIN_REACH_PROB,
        "New-particle input": "diameter + density + compressibility",
    }

    metadata = {
        "selected_model": best_name,
        "features": list(features),
        "target": "x_rel_um = x_particle_um - x_wall_um",
        "x_wall_formula": "(9.9 - 9*sin(theta_deg)) mm",
        "channel_width_um": W_CH_UM,
        "min_reach_probability_for_optimization": MIN_REACH_PROB,
        "pressure_feature_used_instead_of_voltage": True,
        "voltage_role": "report-only conversion back to COMSOL drive setting",
        "observed_angles_only_optimization": observed_angles_only,
        "frequency_unique_values": sorted(map(float, wide["frequency_MHz"].unique())),
        "inlet_velocity_unique_values": sorted(map(float, wide["inlet_velocity_mm_s"].unique())),
        "limitations": [
            "Current data predict final outlet position, not full trajectory.",
            "Frequency/inlet velocity cannot be independently learned if constant.",
            "New-particle predictions are less reliable outside the training property range.",
            "Pair optimization confidence must be judged with pair-specific OOF metrics.",
            "Boundary optima require direct COMSOL confirmation.",
        ],
    }
    with open(out_dir / "model_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    write_html_report(
        out_dir,
        best_name,
        summary,
        model_table,
        per_particle,
        per_angle,
        pair_results_full,
        pair_cv,
        reach_metrics,
    )
    write_results_readme(out_dir, best_name)

    print("\n============================================================")
    print("DONE")
    print(f"Open this first: {(out_dir / 'RESULTS_REPORT.html').resolve()}")
    print(f"All outputs: {out_dir.resolve()}")
    print("============================================================")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        tb = traceback.format_exc()
        try:
            Path("PIPELINE_ERROR.txt").write_text(tb, encoding="utf-8")
        except Exception:
            pass
        print("\nPIPELINE FAILED. Full traceback saved to PIPELINE_ERROR.txt\n")
        print(tb)
        raise
