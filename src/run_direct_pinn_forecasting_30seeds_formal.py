
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Formal 30-seed direct PINN-LV forecasting experiment.

Goal
----
Forecast four future quarters by directly evaluating a solution-network PINN
at future time coordinates. The formal experiment does not use the previous
Euler/RK recursive rollout.

Compared models
---------------
1. M1-Direct-K50 / M1-Direct-K75 / M1-Direct-K100
   - Network input: continuous time t.
   - Network output: normalized states [x_A(t), x_B(t)].
   - Historical observations supervise the data loss.
   - The overlap-constrained LV residual is enforced over the complete domain
     from the first training quarter through the final forecast quarter.
   - Future overlap follows last-observed persistence by default, preventing
     future-graph leakage.

2. M0-Direct-NoOverlap
   - The same direct solution-network PINN without the overlap interaction.

3. CNN, NBEATS, InformerLite
   - Purely data-driven forecasting baselines under the same train/test split.

Formal protocol
---------------
- Default horizon: four quarters.
- Default seeds: 20260221--20260250 (30 matched runs).
- Raw counts are Min-Max normalized using training-period statistics only.
- No logarithmic transformation.
- Predictions are inverse transformed before evaluation.
- Primary relative metric: Macro-NMAE (%), computed independently for the two
  fields and then macro-averaged.
- Paired tests use matched random seeds.

Paper figures
-------------
The script automatically exports:
1. A 2 x 3 six-sequence four-quarter forecasting figure using the pair-specific
   M1 variant with the lowest 30-seed mean Macro-NMAE.
2. A three-panel Macro-NMAE mean +/- standard deviation bar chart containing
   all M1 node budgets and all baselines.
Both PNG and vector PDF versions are generated.
"""

from __future__ import annotations

import argparse
import copy
import math
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score as sk_r2_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


DTYPE = torch.float64


# ============================================================
# 1. Basic utilities
# ============================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def clean_numeric(x):
    if pd.isna(x):
        return np.nan
    if isinstance(x, str):
        return float(x.replace(",", "").strip())
    return float(x)


def inv_softplus(value: float) -> float:
    value = max(float(value), 1.0e-8)
    return math.log(math.expm1(value)) if value < 20 else value


def estimate_exp_rate(t: np.ndarray, x: np.ndarray) -> float:
    x = np.clip(x.astype(np.float64), 1.0e-6, None)
    try:
        slope, _ = np.polyfit(t.astype(np.float64), np.log(x), deg=1)
        if np.isfinite(slope):
            return float(np.clip(slope, 1.0e-4, 10.0))
    except Exception:
        pass
    return 0.2


def ccc_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)

    if y_true.size < 2:
        return float("nan")

    mean_true = float(np.mean(y_true))
    mean_pred = float(np.mean(y_pred))
    var_true = float(np.var(y_true))
    var_pred = float(np.var(y_pred))
    cov = float(np.mean((y_true - mean_true) * (y_pred - mean_pred)))

    denom = var_true + var_pred + (mean_true - mean_pred) ** 2
    return float(2.0 * cov / denom) if denom > 0 else float("nan")


def nmae_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    epsilon: float = 1.0e-12,
) -> float:
    """
    Mean-normalized MAE, returned as a ratio.

    NMAE = sum(|y - y_hat|) / sum(|y|)

    This is equivalent to MAE divided by the mean absolute ground-truth
    magnitude when the two vectors have the same length.
    """
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)

    numerator = float(np.sum(np.abs(y_true - y_pred)))
    denominator = float(np.sum(np.abs(y_true)))

    if denominator <= epsilon:
        return float("nan")
    return numerator / denominator


def compute_metrics_vector(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)

    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(sk_r2_score(y_true, y_pred)) if len(y_true) > 1 else float("nan"),
        "ccc": ccc_score(y_true, y_pred),
        "nmae": nmae_score(y_true, y_pred),
    }


def compute_pair_metrics(true_raw: np.ndarray, pred_raw: np.ndarray) -> dict[str, float]:
    true_raw = np.asarray(true_raw, dtype=np.float64)
    pred_raw = np.asarray(pred_raw, dtype=np.float64)

    curve = compute_metrics_vector(true_raw, pred_raw)
    left = compute_metrics_vector(true_raw[:, 0], pred_raw[:, 0])
    right = compute_metrics_vector(true_raw[:, 1], pred_raw[:, 1])

    # Calculate NMAE separately for the two fields, then macro-average them.
    # Multiplication by 100 makes the exported values directly readable as %.
    left_nmae_percent = 100.0 * left["nmae"]
    right_nmae_percent = 100.0 * right["nmae"]
    macro_nmae_percent = float(np.nanmean([left_nmae_percent, right_nmae_percent]))

    return {
        "raw_curve_mae": curve["mae"],
        "raw_curve_rmse": curve["rmse"],
        "raw_curve_r2": curve["r2"],
        "raw_curve_ccc": curve["ccc"],
        "macro_nmae_percent": macro_nmae_percent,

        "left_raw_mae": left["mae"],
        "left_raw_rmse": left["rmse"],
        "left_r2": left["r2"],
        "left_ccc": left["ccc"],
        "left_nmae_percent": left_nmae_percent,

        "right_raw_mae": right["mae"],
        "right_raw_rmse": right["rmse"],
        "right_r2": right["r2"],
        "right_ccc": right["ccc"],
        "right_nmae_percent": right_nmae_percent,
    }


# ============================================================
# 2. Data and scaling
# ============================================================

@dataclass
class ScaleInfo:
    min_value: float
    max_value: float
    low: float
    high: float

    def transform(self, values: np.ndarray) -> np.ndarray:
        span = self.max_value - self.min_value
        if span <= 0:
            raise ValueError("Cannot Min-Max normalize a constant series.")
        return self.low + (self.high - self.low) * (values - self.min_value) / span

    def inverse(self, values: np.ndarray) -> np.ndarray:
        return self.min_value + (values - self.low) / (self.high - self.low) * (self.max_value - self.min_value)


def minmax_counts_train_only(
    full_values: np.ndarray,
    train_end_idx: int,
    low: float,
    high: float,
) -> tuple[np.ndarray, ScaleInfo]:
    full_values = np.asarray(full_values, dtype=np.float64)
    train_values = full_values[: train_end_idx + 1]
    scale = ScaleInfo(
        min_value=float(np.min(train_values)),
        max_value=float(np.max(train_values)),
        low=float(low),
        high=float(high),
    )
    return scale.transform(full_values), scale


def normalize_overlap(values: np.ndarray, mode: str, train_end_idx: int) -> np.ndarray:
    vals = values.astype(np.float64)

    if mode == "none":
        return vals

    if mode == "auto":
        if vals.min() >= -1.0e-12 and vals.max() <= 1.0 + 1.0e-12:
            return vals
        mode = "train_minmax"

    if mode == "train_minmax":
        train_vals = vals[: train_end_idx + 1]
        span = train_vals.max() - train_vals.min()
        if span <= 0:
            return np.zeros_like(vals)
        return (vals - train_vals.min()) / span

    if mode == "full_minmax":
        span = vals.max() - vals.min()
        if span <= 0:
            return np.zeros_like(vals)
        return (vals - vals.min()) / span

    raise ValueError(f"Unknown overlap normalization mode: {mode}")


PAIR_SPECS = {
    "cnn_gnn": {
        "left_label": "CNN",
        "right_label": "GNN",
        "left_candidates": ["CNN_count", "CNN数量", "CNN", "cnn_count"],
        "right_candidates": ["GNN_count", "GNN数量", "GNN", "gnn_count"],
        "overlap_candidates": ["O_CNN_GNN", "cnn_gnn", "O_CNN-GNN", "O_CNN–GNN"],
    },
    "cnn_transformer": {
        "left_label": "CNN",
        "right_label": "Transformer",
        "left_candidates": ["CNN_count", "CNN数量", "CNN", "cnn_count"],
        "right_candidates": ["Transformer_count", "transformer数量", "Transformer数量", "Transformer", "transformer_count"],
        "overlap_candidates": ["O_CNN_Transformer", "cnn_transformer", "O_CNN-Transformer", "O_CNN–Transformer"],
    },
    "gnn_transformer": {
        "left_label": "GNN",
        "right_label": "Transformer",
        "left_candidates": ["GNN_count", "GNN数量", "GNN", "gnn_count"],
        "right_candidates": ["Transformer_count", "transformer数量", "Transformer数量", "Transformer", "transformer_count"],
        "overlap_candidates": ["O_GNN_Transformer", "gnn_transformer", "O_GNN-Transformer", "O_GNN–Transformer"],
    },
}


def find_col(df: pd.DataFrame, candidates: list[str]) -> str:
    cols = list(df.columns)
    lower_map = {str(c).strip().lower(): c for c in cols}

    for cand in candidates:
        key = cand.strip().lower()
        if key in lower_map:
            return lower_map[key]

    for cand in candidates:
        ck = cand.strip().lower().replace("_", "").replace("-", "").replace("–", "")
        for col in cols:
            k = str(col).strip().lower().replace("_", "").replace("-", "").replace("–", "")
            if k == ck:
                return col

    raise ValueError(f"Cannot find any of columns {candidates}. Existing columns: {cols}")


@dataclass
class PairData:
    pair: str
    quarters: list[str]
    t: np.ndarray
    labels: tuple[str, str]
    raw_values: np.ndarray
    norm_values: np.ndarray
    overlap: np.ndarray
    left_scale: ScaleInfo
    right_scale: ScaleInfo


def load_pair_data(
    csv_path: str,
    pair: str,
    train_end_idx: int,
    minmax_low: float,
    minmax_high: float,
    overlap_normalization: str,
) -> PairData:
    if pair not in PAIR_SPECS:
        raise ValueError(f"Unknown pair={pair}. Valid: {list(PAIR_SPECS.keys())}")

    raw = pd.read_csv(csv_path)

    quarter_col = find_col(raw, ["quarter", "时间", "季度"])
    spec = PAIR_SPECS[pair]
    left_col = find_col(raw, spec["left_candidates"])
    right_col = find_col(raw, spec["right_candidates"])
    overlap_col = find_col(raw, spec["overlap_candidates"])

    df = pd.DataFrame({
        "quarter": raw[quarter_col].astype(str),
        "left_raw": raw[left_col].apply(clean_numeric),
        "right_raw": raw[right_col].apply(clean_numeric),
        "overlap_raw": raw[overlap_col].apply(clean_numeric),
    }).dropna().reset_index(drop=True)

    quarters = df["quarter"].tolist()
    t = np.arange(len(df), dtype=np.float64) / 4.0

    left_x, left_scale = minmax_counts_train_only(
        df["left_raw"].to_numpy(np.float64),
        train_end_idx=train_end_idx,
        low=minmax_low,
        high=minmax_high,
    )

    right_x, right_scale = minmax_counts_train_only(
        df["right_raw"].to_numpy(np.float64),
        train_end_idx=train_end_idx,
        low=minmax_low,
        high=minmax_high,
    )

    overlap = normalize_overlap(
        df["overlap_raw"].to_numpy(np.float64),
        mode=overlap_normalization,
        train_end_idx=train_end_idx,
    )

    return PairData(
        pair=pair,
        quarters=quarters,
        t=t,
        labels=(spec["left_label"], spec["right_label"]),
        raw_values=df[["left_raw", "right_raw"]].to_numpy(np.float64),
        norm_values=np.column_stack([left_x, right_x]),
        overlap=overlap,
        left_scale=left_scale,
        right_scale=right_scale,
    )


def inverse_pair(data: PairData, pred_norm: np.ndarray) -> np.ndarray:
    left = data.left_scale.inverse(pred_norm[:, 0])
    right = data.right_scale.inverse(pred_norm[:, 1])
    return np.column_stack([left, right])


# ============================================================
# 3. M1 overlap-only PINN
# ============================================================

class M1OverlapOnlyPINN(nn.Module):
    def __init__(
        self,
        hidden_width: int,
        rate_left_init: float,
        rate_right_init: float,
        t_min: float,
        t_max: float,
        output_low: float,
        output_high: float,
    ) -> None:
        super().__init__()

        self.output_low = float(output_low)
        self.output_high = float(output_high)

        self.register_buffer("t_min", torch.tensor(float(t_min), dtype=DTYPE))
        self.register_buffer("t_max", torch.tensor(float(t_max), dtype=DTYPE))

        self.network = nn.Sequential(
            nn.Linear(1, hidden_width),
            nn.Tanh(),
            nn.Linear(hidden_width, hidden_width),
            nn.Tanh(),
            nn.Linear(hidden_width, 2),
        )

        self._init_xavier()

        self.rate_left_raw = nn.Parameter(torch.tensor(inv_softplus(rate_left_init), dtype=DTYPE))
        self.rate_right_raw = nn.Parameter(torch.tensor(inv_softplus(rate_right_init), dtype=DTYPE))

    def _init_xavier(self) -> None:
        linear_layers = [m for m in self.network.modules() if isinstance(m, nn.Linear)]
        for layer in linear_layers[:-1]:
            nn.init.xavier_uniform_(layer.weight, gain=nn.init.calculate_gain("tanh"))
            nn.init.zeros_(layer.bias)
        nn.init.xavier_uniform_(linear_layers[-1].weight, gain=1.0)
        nn.init.zeros_(linear_layers[-1].bias)

    def scale_time(self, t: torch.Tensor) -> torch.Tensor:
        return 2.0 * (t - self.t_min) / (self.t_max - self.t_min) - 1.0

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        raw = self.network(self.scale_time(t))
        return self.output_low + (self.output_high - self.output_low) * torch.sigmoid(raw)

    def rates(self) -> tuple[torch.Tensor, torch.Tensor]:
        return F.softplus(self.rate_left_raw) + 1.0e-8, F.softplus(self.rate_right_raw) + 1.0e-8


def interpolate(query_t: torch.Tensor, observed_t: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    flat = query_t.reshape(-1)
    nodes = observed_t.reshape(-1)
    vals = values.reshape(-1)

    flat_clamped = torch.clamp(flat, nodes[0], nodes[-1])
    right = torch.searchsorted(nodes, flat_clamped, right=True).clamp(1, nodes.numel() - 1)
    left = right - 1
    fraction = (flat_clamped - nodes[left]) / (nodes[right] - nodes[left])

    return (vals[left] + fraction * (vals[right] - vals[left])).reshape_as(query_t)


def competition_rhs_np(x: np.ndarray, coeff: float, rate_left: float, rate_right: float) -> np.ndarray:
    left, right = float(x[0]), float(x[1])
    interaction = float(coeff) * left * right
    return np.array([
        rate_left * left - interaction,
        rate_right * right - interaction,
    ], dtype=np.float64)


def m1_rhs_np(x: np.ndarray, overlap: float, rate_left: float, rate_right: float) -> np.ndarray:
    return competition_rhs_np(x=x, coeff=overlap, rate_left=rate_left, rate_right=rate_right)


def competition_rhs_torch(states: torch.Tensor, coeff: torch.Tensor, rate_left: torch.Tensor, rate_right: torch.Tensor) -> torch.Tensor:
    left = states[:, 0]
    right = states[:, 1]
    c = coeff.reshape(-1)
    interaction = c * left * right

    return torch.column_stack((
        rate_left * left - interaction,
        rate_right * right - interaction,
    ))


def m1_rhs_torch(states: torch.Tensor, overlap: torch.Tensor, rate_left: torch.Tensor, rate_right: torch.Tensor) -> torch.Tensor:
    return competition_rhs_torch(states=states, coeff=overlap, rate_left=rate_left, rate_right=rate_right)


def train_m1_pinn(
    data: PairData,
    train_end_idx: int,
    seed: int,
    device: str,
    hidden_width: int,
    collocation_points: int,
    warmup_steps: int,
    full_steps: int,
    learning_rate: float,
    lr_decay_steps: int,
    lr_decay_factor: float,
    curve_weight: float,
    ode_weight: float,
    rate_l2_weight: float,
    grad_clip: float,
    minmax_low: float,
    minmax_high: float,
) -> dict:
    set_seed(seed)
    torch.set_num_threads(1)

    t_train = data.t[: train_end_idx + 1]
    x_train = data.norm_values[: train_end_idx + 1]
    o_train = data.overlap[: train_end_idx + 1]

    rate_left_init = estimate_exp_rate(t_train, x_train[:, 0])
    rate_right_init = estimate_exp_rate(t_train, x_train[:, 1])

    model = M1OverlapOnlyPINN(
        hidden_width=hidden_width,
        rate_left_init=rate_left_init,
        rate_right_init=rate_right_init,
        t_min=float(t_train[0]),
        t_max=float(t_train[-1]),
        output_low=minmax_low,
        output_high=minmax_high,
    ).to(device=device, dtype=DTYPE)

    anchor_t = torch.tensor(t_train[:, None], dtype=DTYPE, device=device)
    anchor_x = torch.tensor(x_train, dtype=DTYPE, device=device)
    overlap_t = torch.tensor(t_train[:, None], dtype=DTYPE, device=device)
    overlap_values = torch.tensor(o_train[:, None], dtype=DTYPE, device=device)
    collocation_t_base = torch.linspace(
        float(t_train[0]),
        float(t_train[-1]),
        collocation_points,
        dtype=DTYPE,
        device=device,
    )[:, None]

    warmup = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for _ in range(warmup_steps):
        warmup.zero_grad()
        pred = model(anchor_t)
        loss = torch.mean(torch.sum((pred - anchor_x).square(), dim=1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        warmup.step()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=lr_decay_steps,
        gamma=lr_decay_factor,
    )

    best_curve = float("inf")
    best_state = None

    for _ in range(full_steps):
        optimizer.zero_grad()

        pred = model(anchor_t)
        curve_loss = torch.mean(torch.sum((pred - anchor_x).square(), dim=1))

        collocation_t = collocation_t_base.detach().clone().requires_grad_(True)
        states = model(collocation_t)

        derivatives = [
            torch.autograd.grad(states[:, i].sum(), collocation_t, create_graph=True)[0][:, 0]
            for i in range(2)
        ]

        overlap = interpolate(collocation_t, overlap_t, overlap_values)
        rate_left, rate_right = model.rates()
        target = m1_rhs_torch(states, overlap, rate_left, rate_right)
        ode_loss = torch.mean(torch.sum((torch.column_stack(derivatives) - target).square(), dim=1))
        rate_loss = rate_left.square() + rate_right.square()

        loss = curve_weight * curve_loss + ode_weight * ode_loss + rate_l2_weight * rate_loss

        curve_value = float(curve_loss.detach().cpu())
        if curve_value < best_curve:
            best_curve = curve_value
            best_state = copy.deepcopy(model.state_dict())

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        scheduler.step()

    if best_state is not None:
        model.load_state_dict(best_state)

    with torch.no_grad():
        rate_left, rate_right = model.rates()
        train_fit_norm = model(anchor_t).detach().cpu().numpy()

    return {
        "model": "M1_overlap_only_PINN",
        "seed": seed,
        "train_fit_norm": train_fit_norm,
        "rate_left": float(rate_left.detach().cpu()),
        "rate_right": float(rate_right.detach().cpu()),
        "curve_loss": float(best_curve),
    }


def forecast_m1_recursive(
    data: PairData,
    train_end_idx: int,
    horizon: int,
    rate_left: float,
    rate_right: float,
    future_o_mode: str,
    clip_norm: bool,
    clip_low: float,
    clip_high: float,
) -> np.ndarray:
    dt = float(np.median(np.diff(data.t)))
    x = data.norm_values[train_end_idx].astype(np.float64).copy()
    last_o = float(data.overlap[train_end_idx])

    preds = []

    for k in range(horizon):
        target_idx = train_end_idx + 1 + k

        if future_o_mode == "last":
            o_t = last_o
        elif future_o_mode == "true":
            o_t = float(data.overlap[target_idx])
        else:
            raise ValueError("--future-o-mode must be last or true")

        dx = m1_rhs_np(x, o_t, rate_left, rate_right)
        x = x + dt * dx

        if clip_norm:
            x = np.clip(x, clip_low, clip_high)

        preds.append(x.copy())

    return np.asarray(preds, dtype=np.float64)


def fixed_c_model_name(c: float) -> str:
    if abs(c - 0.0) < 1e-12:
        return "M0_fixed_c0"
    if abs(c - 0.5) < 1e-12:
        return "M0_fixed_c0p5"
    if abs(c - 1.0) < 1e-12:
        return "M0_fixed_c1"
    txt = str(c).replace(".", "p").replace("-", "m")
    return f"M0_fixed_c{txt}"


def train_m0_fixed_pinn(
    data: PairData,
    train_end_idx: int,
    fixed_c: float,
    seed: int,
    device: str,
    hidden_width: int,
    collocation_points: int,
    warmup_steps: int,
    full_steps: int,
    learning_rate: float,
    lr_decay_steps: int,
    lr_decay_factor: float,
    curve_weight: float,
    ode_weight: float,
    rate_l2_weight: float,
    grad_clip: float,
    minmax_low: float,
    minmax_high: float,
) -> dict:
    """
    Fixed coefficient competition baseline:

        dx_A/dt = r_A x_A - c x_A x_B
        dx_B/dt = r_B x_B - c x_A x_B

    This uses the same PINN training protocol as M1, but replaces O_AB(t)
    with a constant coefficient c.
    """
    set_seed(seed)
    torch.set_num_threads(1)

    t_train = data.t[: train_end_idx + 1]
    x_train = data.norm_values[: train_end_idx + 1]

    rate_left_init = estimate_exp_rate(t_train, x_train[:, 0])
    rate_right_init = estimate_exp_rate(t_train, x_train[:, 1])

    model = M1OverlapOnlyPINN(
        hidden_width=hidden_width,
        rate_left_init=rate_left_init,
        rate_right_init=rate_right_init,
        t_min=float(t_train[0]),
        t_max=float(t_train[-1]),
        output_low=minmax_low,
        output_high=minmax_high,
    ).to(device=device, dtype=DTYPE)

    anchor_t = torch.tensor(t_train[:, None], dtype=DTYPE, device=device)
    anchor_x = torch.tensor(x_train, dtype=DTYPE, device=device)
    collocation_t_base = torch.linspace(
        float(t_train[0]),
        float(t_train[-1]),
        collocation_points,
        dtype=DTYPE,
        device=device,
    )[:, None]

    warmup = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for _ in range(warmup_steps):
        warmup.zero_grad()
        pred = model(anchor_t)
        loss = torch.mean(torch.sum((pred - anchor_x).square(), dim=1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        warmup.step()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=lr_decay_steps,
        gamma=lr_decay_factor,
    )

    best_curve = float("inf")
    best_state = None

    for _ in range(full_steps):
        optimizer.zero_grad()

        pred = model(anchor_t)
        curve_loss = torch.mean(torch.sum((pred - anchor_x).square(), dim=1))

        collocation_t = collocation_t_base.detach().clone().requires_grad_(True)
        states = model(collocation_t)

        derivatives = [
            torch.autograd.grad(states[:, i].sum(), collocation_t, create_graph=True)[0][:, 0]
            for i in range(2)
        ]

        rate_left, rate_right = model.rates()
        fixed_coeff_tensor = torch.full_like(collocation_t[:, 0], float(fixed_c))
        target = competition_rhs_torch(states, fixed_coeff_tensor, rate_left, rate_right)
        ode_loss = torch.mean(torch.sum((torch.column_stack(derivatives) - target).square(), dim=1))
        rate_loss = rate_left.square() + rate_right.square()

        loss = curve_weight * curve_loss + ode_weight * ode_loss + rate_l2_weight * rate_loss

        curve_value = float(curve_loss.detach().cpu())
        if curve_value < best_curve:
            best_curve = curve_value
            best_state = copy.deepcopy(model.state_dict())

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        scheduler.step()

    if best_state is not None:
        model.load_state_dict(best_state)

    with torch.no_grad():
        rate_left, rate_right = model.rates()
        train_fit_norm = model(anchor_t).detach().cpu().numpy()

    return {
        "model": fixed_c_model_name(fixed_c),
        "seed": seed,
        "fixed_c": float(fixed_c),
        "train_fit_norm": train_fit_norm,
        "rate_left": float(rate_left.detach().cpu()),
        "rate_right": float(rate_right.detach().cpu()),
        "curve_loss": float(best_curve),
    }


def forecast_m0_fixed_recursive(
    data: PairData,
    train_end_idx: int,
    horizon: int,
    fixed_c: float,
    rate_left: float,
    rate_right: float,
    clip_norm: bool,
    clip_low: float,
    clip_high: float,
) -> np.ndarray:
    dt = float(np.median(np.diff(data.t)))
    x = data.norm_values[train_end_idx].astype(np.float64).copy()

    preds = []
    for _ in range(horizon):
        dx = competition_rhs_np(x=x, coeff=fixed_c, rate_left=rate_left, rate_right=rate_right)
        x = x + dt * dx

        if clip_norm:
            x = np.clip(x, clip_low, clip_high)

        preds.append(x.copy())

    return np.asarray(preds, dtype=np.float64)



# ============================================================
# 3B. Direct solution-network PINN extrapolation
# ============================================================

class DirectExtrapolationPINN(nn.Module):
    """
    A direct solution-network PINN.

    Input
    -----
    t: continuous time coordinate, shape [N, 1]

    Output
    ------
    Positive normalized states [x_A(t), x_B(t)], shape [N, 2].

    Unlike the original bounded sigmoid output, Softplus leaves the upper
    range unbounded, so future normalized states may exceed the training-period
    maximum of 0.95 when the learned dynamics require growth.
    """

    def __init__(
        self,
        hidden_width: int,
        hidden_layers: int,
        rate_left_init: float,
        rate_right_init: float,
        t_min: float,
        t_max: float,
    ) -> None:
        super().__init__()

        if hidden_layers < 1:
            raise ValueError("hidden_layers must be at least 1")
        if t_max <= t_min:
            raise ValueError("t_max must be greater than t_min")

        self.register_buffer("t_min", torch.tensor(float(t_min), dtype=DTYPE))
        self.register_buffer("t_max", torch.tensor(float(t_max), dtype=DTYPE))

        layers: list[nn.Module] = [nn.Linear(1, hidden_width), nn.Tanh()]
        for _ in range(hidden_layers - 1):
            layers.extend([nn.Linear(hidden_width, hidden_width), nn.Tanh()])
        layers.append(nn.Linear(hidden_width, 2))
        self.network = nn.Sequential(*layers)

        self._init_xavier()
        self.rate_left_raw = nn.Parameter(
            torch.tensor(inv_softplus(rate_left_init), dtype=DTYPE)
        )
        self.rate_right_raw = nn.Parameter(
            torch.tensor(inv_softplus(rate_right_init), dtype=DTYPE)
        )

    def _init_xavier(self) -> None:
        linear_layers = [m for m in self.network.modules() if isinstance(m, nn.Linear)]
        for layer in linear_layers[:-1]:
            nn.init.xavier_uniform_(layer.weight, gain=nn.init.calculate_gain("tanh"))
            nn.init.zeros_(layer.bias)
        nn.init.xavier_uniform_(linear_layers[-1].weight, gain=1.0)
        nn.init.zeros_(linear_layers[-1].bias)

    def scale_time(self, t: torch.Tensor) -> torch.Tensor:
        # The complete history + forecast domain maps to [-1, 1].
        return 2.0 * (t - self.t_min) / (self.t_max - self.t_min) - 1.0

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        raw = self.network(self.scale_time(t))
        return F.softplus(raw) + 1.0e-8

    def rates(self) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            F.softplus(self.rate_left_raw) + 1.0e-8,
            F.softplus(self.rate_right_raw) + 1.0e-8,
        )


def _make_overlap_interpolator_inputs(
    data: PairData,
    train_end_idx: int,
    horizon: int,
    future_o_mode: str,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Build the overlap nodes used by the physics residual.

    last: use historical overlap nodes only. The existing interpolate()
          function clamps future query times to the final node, implementing
          last-observed persistence without test leakage.
    true: include true future overlap nodes through the forecast horizon.
          This mode is diagnostic and leaks test-period graph information.
    """
    if future_o_mode == "last":
        end_idx = train_end_idx
    elif future_o_mode == "true":
        end_idx = train_end_idx + horizon
    else:
        raise ValueError("--future-o-mode must be last or true")

    overlap_t = torch.tensor(
        data.t[: end_idx + 1, None], dtype=DTYPE, device=device
    )
    overlap_values = torch.tensor(
        data.overlap[: end_idx + 1, None], dtype=DTYPE, device=device
    )
    return overlap_t, overlap_values


def train_direct_pinn(
    data: PairData,
    train_end_idx: int,
    horizon: int,
    seed: int,
    device: str,
    use_overlap: bool,
    future_o_mode: str,
    hidden_width: int,
    hidden_layers: int,
    collocation_points: int,
    warmup_steps: int,
    full_steps: int,
    learning_rate: float,
    lr_decay_steps: int,
    lr_decay_factor: float,
    curve_weight: float,
    ode_weight: float,
    boundary_weight: float,
    rate_l2_weight: float,
    grad_clip: float,
) -> dict:
    """
    Train a direct solution-network PINN over history + forecast time.

    Data loss is evaluated only on observed training quarters. Physics loss is
    evaluated on collocation points covering the complete extrapolation domain.
    The final observed state is included as an explicit boundary anchor.
    """
    set_seed(seed)
    torch.set_num_threads(1)

    forecast_end_idx = train_end_idx + horizon
    if forecast_end_idx >= len(data.t):
        raise ValueError("Forecast horizon extends beyond available timestamps.")

    t_train = data.t[: train_end_idx + 1]
    x_train = data.norm_values[: train_end_idx + 1]
    t_forecast_end = float(data.t[forecast_end_idx])

    rate_left_init = estimate_exp_rate(t_train, x_train[:, 0])
    rate_right_init = estimate_exp_rate(t_train, x_train[:, 1])

    model = DirectExtrapolationPINN(
        hidden_width=hidden_width,
        hidden_layers=hidden_layers,
        rate_left_init=rate_left_init,
        rate_right_init=rate_right_init,
        t_min=float(t_train[0]),
        t_max=t_forecast_end,
    ).to(device=device, dtype=DTYPE)

    anchor_t = torch.tensor(t_train[:, None], dtype=DTYPE, device=device)
    anchor_x = torch.tensor(x_train, dtype=DTYPE, device=device)
    boundary_t = anchor_t[-1:]
    boundary_x = anchor_x[-1:]

    overlap_t = None
    overlap_values = None
    if use_overlap:
        overlap_t, overlap_values = _make_overlap_interpolator_inputs(
            data=data,
            train_end_idx=train_end_idx,
            horizon=horizon,
            future_o_mode=future_o_mode,
            device=device,
        )

    # Warm up only on observed data so the network first learns the measured
    # trajectory before the physics residual is introduced.
    warmup = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for _ in range(warmup_steps):
        warmup.zero_grad()
        pred_hist = model(anchor_t)
        warmup_loss = torch.mean(torch.sum((pred_hist - anchor_x).square(), dim=1))
        warmup_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        warmup.step()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=lr_decay_steps,
        gamma=lr_decay_factor,
    )

    best_total = float("inf")
    best_state = None
    best_stats: dict[str, float] = {}

    t_start = float(t_train[0])
    t_end = t_forecast_end

    for _ in range(full_steps):
        optimizer.zero_grad()

        pred_hist = model(anchor_t)
        curve_loss = torch.mean(torch.sum((pred_hist - anchor_x).square(), dim=1))

        # Random collocation points are redrawn every step. Include both domain
        # endpoints explicitly to stabilize the boundary and forecast end.
        if collocation_points < 2:
            raise ValueError("collocation_points must be at least 2")
        random_t = t_start + (t_end - t_start) * torch.rand(
            collocation_points - 2, 1, dtype=DTYPE, device=device
        )
        endpoint_t = torch.tensor([[t_start], [t_end]], dtype=DTYPE, device=device)
        collocation_t = torch.cat([endpoint_t, random_t], dim=0)
        collocation_t = collocation_t.detach().clone().requires_grad_(True)

        states = model(collocation_t)
        derivatives = [
            torch.autograd.grad(
                states[:, i].sum(), collocation_t, create_graph=True
            )[0][:, 0]
            for i in range(2)
        ]

        rate_left, rate_right = model.rates()
        if use_overlap:
            assert overlap_t is not None and overlap_values is not None
            coeff = interpolate(collocation_t, overlap_t, overlap_values)
        else:
            coeff = torch.zeros_like(collocation_t)

        target = competition_rhs_torch(
            states=states,
            coeff=coeff,
            rate_left=rate_left,
            rate_right=rate_right,
        )
        ode_loss = torch.mean(
            torch.sum((torch.column_stack(derivatives) - target).square(), dim=1)
        )

        boundary_pred = model(boundary_t)
        boundary_loss = torch.mean(torch.sum((boundary_pred - boundary_x).square(), dim=1))
        rate_loss = rate_left.square() + rate_right.square()

        loss = (
            curve_weight * curve_loss
            + ode_weight * ode_loss
            + boundary_weight * boundary_loss
            + rate_l2_weight * rate_loss
        )

        total_value = float(loss.detach().cpu())
        if np.isfinite(total_value) and total_value < best_total:
            best_total = total_value
            best_state = copy.deepcopy(model.state_dict())
            best_stats = {
                "curve_loss": float(curve_loss.detach().cpu()),
                "ode_loss": float(ode_loss.detach().cpu()),
                "boundary_loss": float(boundary_loss.detach().cpu()),
                "total_loss": total_value,
            }

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        scheduler.step()

    if best_state is None:
        raise RuntimeError("Direct PINN training did not produce a finite checkpoint.")
    model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        rate_left, rate_right = model.rates()
        train_fit_norm = model(anchor_t).cpu().numpy()

    model_name = "M1-Direct" if use_overlap else "M0-Direct-NoOverlap"
    return {
        "model": model_name,
        "seed": seed,
        "torch_model": model,
        "train_fit_norm": train_fit_norm,
        "rate_left": float(rate_left.cpu()),
        "rate_right": float(rate_right.cpu()),
        **best_stats,
    }


@torch.no_grad()
def forecast_direct_pinn(
    model: DirectExtrapolationPINN,
    data: PairData,
    train_end_idx: int,
    horizon: int,
    device: str,
    clip_norm: bool,
    clip_low: float,
    clip_high: float,
) -> np.ndarray:
    """Evaluate the trained solution network directly at future quarters."""
    future_t_np = data.t[train_end_idx + 1: train_end_idx + 1 + horizon]
    future_t = torch.tensor(future_t_np[:, None], dtype=DTYPE, device=device)
    model.eval()
    pred = model(future_t).cpu().numpy().astype(np.float64)

    if clip_norm:
        pred = np.clip(pred, clip_low, clip_high)
    return pred


# ============================================================
# 4. CNN and InformerLite forecasting baselines
# ============================================================

class OneStepTrainDataset(Dataset):
    def __init__(self, values_norm: np.ndarray, train_end_idx: int, seq_len: int):
        self.x = []
        self.y = []

        values = np.asarray(values_norm[: train_end_idx + 1], dtype=np.float32)
        T = len(values)

        for i in range(T - seq_len):
            self.x.append(values[i:i + seq_len])
            self.y.append(values[i + seq_len])

        self.x = np.asarray(self.x, dtype=np.float32)
        self.y = np.asarray(self.y, dtype=np.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return torch.from_numpy(self.x[idx]), torch.from_numpy(self.y[idx])


class CNNOneStep(nn.Module):
    """
    1D temporal CNN one-step forecasting baseline.

    Input shape : [B, T, 2]
    Conv input  : [B, 2, T]
    Output      : [B, 2]
    """
    def __init__(
        self,
        input_dim=2,
        hidden_dim=64,
        output_dim=2,
        kernel_size=3,
        dropout=0.1,
    ):
        super().__init__()

        padding = kernel_size // 2

        self.conv = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
        )

        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        # x: [B, T, 2]
        z = x.transpose(1, 2)  # [B, 2, T]
        z = self.conv(z)       # [B, H, T]

        last_feat = z[:, :, -1]
        mean_feat = z.mean(dim=-1)
        feat = torch.cat([last_feat, mean_feat], dim=1)

        return self.head(feat)



class RNNOneStep(nn.Module):
    """
    GRU-based one-step forecasting baseline.

    Input shape : [B, T, 2]
    Output      : [B, 2]
    """
    def __init__(
        self,
        input_dim=2,
        hidden_dim=64,
        output_dim=2,
        num_layers=1,
        dropout=0.1,
    ):
        super().__init__()

        rnn_dropout = dropout if num_layers > 1 else 0.0

        self.rnn = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=rnn_dropout,
            batch_first=True,
        )

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        out, _ = self.rnn(x)
        last = out[:, -1, :]
        return self.head(last)



class InformerLiteOneStep(nn.Module):
    def __init__(
        self,
        input_dim=2,
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_feedforward=128,
        output_dim=2,
        max_len=64,
        dropout=0.1,
    ):
        super().__init__()

        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, max_len, d_model) * 0.01)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )

        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)

        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, output_dim),
        )

    def forward(self, x):
        B, T, _ = x.shape
        z = self.input_proj(x)
        z = z + self.pos_emb[:, :T, :]
        z = self.encoder(z)
        pooled = z[:, -1, :]
        return self.head(pooled)


def train_one_step_model(
    model_name: str,
    data: PairData,
    train_end_idx: int,
    seed: int,
    device: str,
    seq_len: int,
    epochs: int,
    learning_rate: float,
    hidden_dim: int,
    nhead: int,
    layers: int,
    dropout: float,
    cnn_kernel_size: int,
    grad_clip: float,
) -> dict:
    set_seed(seed)

    if model_name == "CNN":
        model = CNNOneStep(
            input_dim=2,
            hidden_dim=hidden_dim,
            output_dim=2,
            kernel_size=cnn_kernel_size,
            dropout=dropout,
        )
    elif model_name == "RNN":
        model = RNNOneStep(
            input_dim=2,
            hidden_dim=hidden_dim,
            output_dim=2,
            num_layers=max(1, layers),
            dropout=dropout,
        )
    elif model_name == "InformerLite":
        model = InformerLiteOneStep(
            input_dim=2,
            d_model=hidden_dim,
            nhead=nhead,
            num_layers=layers,
            dim_feedforward=hidden_dim * 2,
            output_dim=2,
            max_len=max(64, seq_len + 8),
            dropout=dropout,
        )
    else:
        raise ValueError(f"Unknown model_name={model_name}")

    dataset = OneStepTrainDataset(data.norm_values, train_end_idx, seq_len)

    if len(dataset) <= 0:
        raise ValueError(
            f"Not enough train samples: train_len={train_end_idx + 1}, seq_len={seq_len}"
        )

    loader = DataLoader(
        dataset,
        batch_size=min(8, len(dataset)),
        shuffle=True,
    )

    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=1e-4,
    )
    loss_fn = nn.MSELoss()

    best_loss = float("inf")
    best_state = None

    model.train()
    for _ in range(epochs):
        running = 0.0
        count = 0

        for x, y in loader:
            x = x.float().to(device)
            y = y.float().to(device)

            pred = model(x)
            loss = loss_fn(pred, y)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            running += float(loss.detach().cpu()) * len(x)
            count += len(x)

        epoch_loss = running / max(count, 1)
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_state = copy.deepcopy(model.state_dict())

    if best_state is not None:
        model.load_state_dict(best_state)

    return {
        "model": model_name,
        "seed": seed,
        "torch_model": model,
        "curve_loss": best_loss,
    }


@torch.no_grad()
def forecast_neural_recursive(
    model: nn.Module,
    data: PairData,
    train_end_idx: int,
    horizon: int,
    seq_len: int,
    device: str,
    clip_norm: bool,
    clip_low: float,
    clip_high: float,
) -> np.ndarray:
    model.eval()

    history = data.norm_values[: train_end_idx + 1].astype(np.float32).copy()
    preds = []

    for _ in range(horizon):
        x = history[-seq_len:]
        x_tensor = torch.from_numpy(x[None, :, :]).float().to(device)
        pred = model(x_tensor).cpu().numpy()[0].astype(np.float64)

        if clip_norm:
            pred = np.clip(pred, clip_low, clip_high)

        preds.append(pred.copy())
        history = np.vstack([history, pred[None, :].astype(np.float32)])

    return np.asarray(preds, dtype=np.float64)




# ============================================================
# 5. N-BEATS direct multi-step forecasting baseline
# ============================================================

class DirectHorizonDataset(Dataset):
    """
    Direct multi-step training dataset for N-BEATS.

    Input : previous seq_len normalized states, shape [seq_len, 2]
    Target: next horizon normalized states, shape [horizon, 2]
    """
    def __init__(self, values_norm: np.ndarray, train_end_idx: int, seq_len: int, horizon: int):
        self.x = []
        self.y = []

        values = np.asarray(values_norm[: train_end_idx + 1], dtype=np.float32)
        T = len(values)

        max_start = T - seq_len - horizon + 1
        for i in range(max_start):
            self.x.append(values[i:i + seq_len])
            self.y.append(values[i + seq_len:i + seq_len + horizon])

        self.x = np.asarray(self.x, dtype=np.float32)
        self.y = np.asarray(self.y, dtype=np.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return torch.from_numpy(self.x[idx]), torch.from_numpy(self.y[idx])


class NBEATSBlock(nn.Module):
    """
    Generic N-BEATS block.

    It maps a flattened input vector to:
    - backcast: reconstruction of the current residual input
    - forecast: direct multi-step forecast vector
    """
    def __init__(
        self,
        input_size: int,
        forecast_size: int,
        hidden_dim: int = 64,
        num_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        layers = []
        in_dim = input_size
        for _ in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim

        self.mlp = nn.Sequential(*layers)
        self.backcast_head = nn.Linear(hidden_dim, input_size)
        self.forecast_head = nn.Linear(hidden_dim, forecast_size)

    def forward(self, residual):
        h = self.mlp(residual)
        backcast = self.backcast_head(h)
        forecast = self.forecast_head(h)
        return backcast, forecast


class NBEATSDirect(nn.Module):
    """
    Small generic N-BEATS for direct multi-step forecasting.

    Input shape : [B, seq_len, 2]
    Output shape: [B, horizon, 2]
    """
    def __init__(
        self,
        seq_len: int,
        horizon: int,
        input_dim: int = 2,
        output_dim: int = 2,
        hidden_dim: int = 64,
        num_blocks: int = 3,
        num_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.seq_len = seq_len
        self.horizon = horizon
        self.input_dim = input_dim
        self.output_dim = output_dim

        input_size = seq_len * input_dim
        forecast_size = horizon * output_dim

        self.blocks = nn.ModuleList([
            NBEATSBlock(
                input_size=input_size,
                forecast_size=forecast_size,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                dropout=dropout,
            )
            for _ in range(num_blocks)
        ])

    def forward(self, x):
        # x: [B, seq_len, 2]
        residual = x.reshape(x.shape[0], -1)
        forecast = torch.zeros(
            x.shape[0],
            self.horizon * self.output_dim,
            device=x.device,
            dtype=x.dtype,
        )

        for block in self.blocks:
            backcast, block_forecast = block(residual)
            residual = residual - backcast
            forecast = forecast + block_forecast

        return forecast.reshape(x.shape[0], self.horizon, self.output_dim)


def train_nbeats_direct(
    data: PairData,
    train_end_idx: int,
    horizon: int,
    seed: int,
    device: str,
    seq_len: int,
    epochs: int,
    learning_rate: float,
    hidden_dim: int,
    num_blocks: int,
    num_layers: int,
    dropout: float,
    grad_clip: float,
    clip_norm: bool,
    clip_low: float,
    clip_high: float,
) -> dict:
    set_seed(seed)

    model = NBEATSDirect(
        seq_len=seq_len,
        horizon=horizon,
        input_dim=2,
        output_dim=2,
        hidden_dim=hidden_dim,
        num_blocks=num_blocks,
        num_layers=num_layers,
        dropout=dropout,
    )

    dataset = DirectHorizonDataset(
        values_norm=data.norm_values,
        train_end_idx=train_end_idx,
        seq_len=seq_len,
        horizon=horizon,
    )

    if len(dataset) <= 0:
        raise ValueError(
            f"Not enough train samples for NBEATS: "
            f"train_len={train_end_idx + 1}, seq_len={seq_len}, horizon={horizon}"
        )

    loader = DataLoader(
        dataset,
        batch_size=min(8, len(dataset)),
        shuffle=True,
    )

    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=1e-4,
    )
    loss_fn = nn.MSELoss()

    best_loss = float("inf")
    best_state = None

    model.train()
    for _ in range(epochs):
        running = 0.0
        count = 0

        for x, y in loader:
            x = x.float().to(device)
            y = y.float().to(device)

            pred = model(x)
            loss = loss_fn(pred, y)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            running += float(loss.detach().cpu()) * len(x)
            count += len(x)

        epoch_loss = running / max(count, 1)
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_state = copy.deepcopy(model.state_dict())

    if best_state is not None:
        model.load_state_dict(best_state)

    # Direct horizon forecast from the last seq_len training points.
    model.eval()
    history = data.norm_values[: train_end_idx + 1].astype(np.float32)
    x = history[-seq_len:]
    x_tensor = torch.from_numpy(x[None, :, :]).float().to(device)

    with torch.no_grad():
        future_norm = model(x_tensor).cpu().numpy()[0].astype(np.float64)

    if clip_norm:
        future_norm = np.clip(future_norm, clip_low, clip_high)

    return {
        "model": "NBEATS",
        "seed": seed,
        "future_norm": future_norm,
        "curve_loss": best_loss,
    }


# ============================================================
# 6. Plotting
# ============================================================

def plot_forecast_curves(
    outdir: Path,
    data: PairData,
    train_end_idx: int,
    horizon: int,
    plot_pred_raw: dict[str, np.ndarray],
) -> None:
    plot_dir = outdir / "plots" / f"h{horizon}"
    ensure_dir(plot_dir)

    end_idx = train_end_idx + horizon
    quarters = data.quarters[: end_idx + 1]
    true_full = data.raw_values[: end_idx + 1]
    train_raw = data.raw_values[: train_end_idx + 1]

    x = np.arange(len(quarters))
    tick_idx = np.arange(0, len(quarters), 2)

    def full_curve(pred_future_raw: np.ndarray) -> np.ndarray:
        return np.vstack([train_raw, pred_future_raw])

    def draw(dim: int, label: str):
        plt.figure(figsize=(12, 4.8))
        plt.plot(x, true_full[:, dim], marker="o", label="True")

        for model_name in ["M1_overlap_only_PINN", "M0_fixed_c0", "M0_fixed_c0p5", "M0_fixed_c1", "CNN", "RNN", "NBEATS", "InformerLite"]:
            if model_name in plot_pred_raw:
                model_full = full_curve(plot_pred_raw[model_name])
                plt.plot(
                    x,
                    model_full[:, dim],
                    marker="x",
                    linestyle="--",
                    label=model_name,
                )

        plt.axvline(train_end_idx + 0.5, linestyle=":", linewidth=1)
        plt.text(
            train_end_idx + 0.6,
            np.nanmax(true_full[:, dim]) * 0.95,
            "Forecast",
            fontsize=9,
        )

        plt.xticks(tick_idx, [quarters[i] for i in tick_idx], rotation=55, ha="right")
        plt.xlabel("Quarter")
        plt.ylabel("Quarterly publication count")
        plt.title(f"{data.pair} | {label} | Forecast comparison | h={horizon}")
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(plot_dir / f"{data.pair}_{label}_forecast_h{horizon}.png", dpi=300)
        plt.close()

    draw(0, data.labels[0])
    draw(1, data.labels[1])


# ============================================================
# 7. Formal direct-PINN experiment
# ============================================================

SELECTED_M1_KS = ["K50", "K75", "K100"]
DIRECT_M1_MODELS = [f"M1-Direct-{k}" for k in SELECTED_M1_KS]
BASELINE_MODELS = ["M0-Direct-NoOverlap", "CNN", "NBEATS", "InformerLite"]
ORDERED_MODELS = DIRECT_M1_MODELS + BASELINE_MODELS


def plot_forecast_curves(
    outdir: Path,
    data: PairData,
    train_end_idx: int,
    horizon: int,
    plot_pred_raw: dict[str, np.ndarray],
) -> None:
    plot_dir = outdir / "plots" / f"h{horizon}"
    ensure_dir(plot_dir)

    end_idx = train_end_idx + horizon
    quarters = data.quarters[: end_idx + 1]
    true_full = data.raw_values[: end_idx + 1]
    train_raw = data.raw_values[: train_end_idx + 1]

    x = np.arange(len(quarters))
    tick_idx = np.arange(0, len(quarters), 2)

    def full_curve(pred_future_raw: np.ndarray) -> np.ndarray:
        return np.vstack([train_raw, pred_future_raw])

    def draw(dim: int, label: str) -> None:
        plt.figure(figsize=(12, 4.8))
        plt.plot(x, true_full[:, dim], marker="o", label="True")

        for model_name in ORDERED_MODELS:
            if model_name in plot_pred_raw:
                model_full = full_curve(plot_pred_raw[model_name])
                plt.plot(
                    x,
                    model_full[:, dim],
                    marker="x",
                    linestyle="--",
                    label=model_name,
                )

        plt.axvline(train_end_idx + 0.5, linestyle=":", linewidth=1)
        plt.text(
            train_end_idx + 0.6,
            np.nanmax(true_full[:, dim]) * 0.95,
            "Forecast",
            fontsize=9,
        )
        plt.xticks(tick_idx, [quarters[i] for i in tick_idx], rotation=55, ha="right")
        plt.xlabel("Quarter")
        plt.ylabel("Quarterly publication count")
        plt.title(f"{data.pair} | {label} | Direct PINN | h={horizon}")
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(
            plot_dir / f"{data.pair}_{label}_direct_pinn_h{horizon}.png",
            dpi=300,
        )
        plt.close()

    draw(0, data.labels[0])
    draw(1, data.labels[1])


def get_k_csv_map(args) -> dict[str, str]:
    explicit = {
        "K50": args.csv_k50,
        "K75": args.csv_k75,
        "K100": args.csv_k100,
    }
    resolved: dict[str, str] = {}
    for k in SELECTED_M1_KS:
        path = explicit.get(k)
        if path is None:
            path = args.csv_template.format(K=k, k=k.lower())
        if not Path(path).exists():
            raise FileNotFoundError(
                f"Cannot find CSV for {k}: {path}. "
                f"Please pass --csv-{k.lower()} explicitly."
            )
        resolved[k] = path
    return resolved


def paired_ttest_1d(
    candidate_values: np.ndarray,
    baseline_values: np.ndarray,
) -> dict[str, float]:
    candidate_values = np.asarray(candidate_values, dtype=np.float64).reshape(-1)
    baseline_values = np.asarray(baseline_values, dtype=np.float64).reshape(-1)
    if candidate_values.shape != baseline_values.shape:
        raise ValueError("Paired t-test requires matched vectors with the same shape.")

    diff = baseline_values - candidate_values
    valid = np.isfinite(diff)
    diff = diff[valid]
    candidate_values = candidate_values[valid]
    baseline_values = baseline_values[valid]
    n = int(diff.size)

    if n < 2:
        return {
            "n": n,
            "mean_diff_baseline_minus_candidate": float(np.mean(diff)) if n else np.nan,
            "std_diff": np.nan,
            "t_stat": np.nan,
            "p_two_sided": np.nan,
            "p_one_sided_candidate_better": np.nan,
        }

    mean_diff = float(np.mean(diff))
    std_diff = float(np.std(diff, ddof=1))
    if std_diff <= 1.0e-15:
        if abs(mean_diff) <= 1.0e-15:
            t_stat, p_two, p_one = 0.0, 1.0, 0.5
        elif mean_diff > 0:
            t_stat, p_two, p_one = float("inf"), 0.0, 0.0
        else:
            t_stat, p_two, p_one = float("-inf"), 0.0, 1.0
    else:
        t_stat = mean_diff / (std_diff / math.sqrt(n))
        try:
            from scipy import stats
            p_two = float(
                stats.ttest_rel(baseline_values, candidate_values, nan_policy="omit").pvalue
            )
            p_one = float(stats.t.sf(t_stat, df=n - 1))
        except Exception:
            p_two = np.nan
            p_one = np.nan

    return {
        "n": n,
        "mean_diff_baseline_minus_candidate": mean_diff,
        "std_diff": std_diff,
        "t_stat": float(t_stat),
        "p_two_sided": p_two,
        "p_one_sided_candidate_better": p_one,
    }


def build_ttest_table(
    result_df: pd.DataFrame,
    pair: str,
    horizon: int,
    train_end_quarter: str,
    test_quarters: list[str],
    metric: str,
) -> pd.DataFrame:
    rows: list[dict] = []
    for candidate_model in DIRECT_M1_MODELS:
        candidate_df = result_df[result_df["model"] == candidate_model][
            ["seed", metric]
        ].rename(columns={metric: "candidate_value"})

        for baseline_model in BASELINE_MODELS:
            baseline_df = result_df[result_df["model"] == baseline_model][
                ["seed", metric]
            ].rename(columns={metric: "baseline_value"})

            merged = pd.merge(candidate_df, baseline_df, on="seed", how="inner").sort_values("seed")
            stats_row = paired_ttest_1d(
                candidate_values=merged["candidate_value"].to_numpy(np.float64),
                baseline_values=merged["baseline_value"].to_numpy(np.float64),
            )
            rows.append({
                "pair": pair,
                "horizon": horizon,
                "train_end_quarter": train_end_quarter,
                "test_quarters": ";".join(test_quarters),
                "metric": metric,
                "candidate_model": candidate_model,
                "baseline_model": baseline_model,
                "candidate_mean": float(merged["candidate_value"].mean()) if len(merged) else np.nan,
                "baseline_mean": float(merged["baseline_value"].mean()) if len(merged) else np.nan,
                **stats_row,
            })
    return pd.DataFrame(rows)


def summarize_seedlevel(
    result_df: pd.DataFrame,
    pair: str,
    horizon: int,
    train_end_quarter: str,
    test_quarters: list[str],
) -> pd.DataFrame:
    metric_cols = [
        "raw_curve_mae",
        "macro_nmae_percent",
        "raw_curve_rmse",
        "raw_curve_r2",
        "raw_curve_ccc",
        "left_raw_mae",
        "left_nmae_percent",
        "left_raw_rmse",
        "left_r2",
        "left_ccc",
        "right_raw_mae",
        "right_nmae_percent",
        "right_raw_rmse",
        "right_r2",
        "right_ccc",
    ]

    summary_rows: list[dict] = []
    for (k_label, model_name), g in result_df.groupby(["K", "model"], sort=False):
        row = {
            "K": k_label,
            "pair": pair,
            "model": model_name,
            "horizon": horizon,
            "runs": int(len(g)),
            "train_end_quarter": train_end_quarter,
            "test_quarters": ";".join(test_quarters),
        }
        for col in metric_cols:
            vals = pd.to_numeric(g[col], errors="coerce").dropna().to_numpy()
            row[col + "_mean"] = float(np.mean(vals)) if len(vals) else np.nan
            row[col + "_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            row[col + "_median"] = float(np.median(vals)) if len(vals) else np.nan
        summary_rows.append(row)

    order_map = {name: i for i, name in enumerate(ORDERED_MODELS)}
    summary_df = pd.DataFrame(summary_rows)
    summary_df["model_order"] = summary_df["model"].map(order_map).fillna(99)
    return summary_df.sort_values(
        ["pair", "horizon", "raw_curve_mae_mean", "model_order"]
    ).drop(columns=["model_order"])


def run_pair_horizon(
    args,
    pair: str,
    horizon: int,
    common_train_end_idx: int | None,
) -> tuple[list[dict], list[dict]]:
    k_csv_map = args.k_csv_map
    base_csv = k_csv_map["K100"] if "K100" in k_csv_map else next(iter(k_csv_map.values()))

    raw = pd.read_csv(base_csv)
    quarter_col = find_col(raw, ["quarter", "时间", "季度"])
    quarters = raw[quarter_col].astype(str).tolist()

    if args.train_mode == "rolling_end":
        train_end_idx = len(quarters) - horizon - 1
    else:
        if common_train_end_idx is None:
            raise RuntimeError("common_train_end_idx is required for train_mode=common")
        train_end_idx = common_train_end_idx

    if train_end_idx < args.seq_len:
        raise ValueError(
            f"train_end_idx={train_end_idx}, train_len={train_end_idx + 1}, "
            f"seq_len={args.seq_len}; not enough history."
        )
    if train_end_idx + horizon >= len(quarters):
        raise ValueError(
            f"Not enough future quarters: train_end={quarters[train_end_idx]}, horizon={horizon}"
        )

    data_by_k = {
        k_label: load_pair_data(
            csv_path=csv_path,
            pair=pair,
            train_end_idx=train_end_idx,
            minmax_low=args.minmax_low,
            minmax_high=args.minmax_high,
            overlap_normalization=args.overlap_normalization,
        )
        for k_label, csv_path in k_csv_map.items()
    }
    data_base = data_by_k["K100"] if "K100" in data_by_k else next(iter(data_by_k.values()))

    outdir = Path(args.outdir) / pair / f"h{horizon}"
    ensure_dir(outdir)

    for k_label, data in data_by_k.items():
        pd.DataFrame({
            "quarter": data.quarters,
            "t": data.t,
            "left_raw": data.raw_values[:, 0],
            "right_raw": data.raw_values[:, 1],
            "x_left": data.norm_values[:, 0],
            "x_right": data.norm_values[:, 1],
            "overlap": data.overlap,
            "is_train": np.arange(len(data.quarters)) <= train_end_idx,
        }).to_csv(
            outdir / f"preprocessed_timeseries_{k_label}.csv",
            index=False,
            encoding="utf-8-sig",
        )

    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    train_end_quarter = data_base.quarters[train_end_idx]
    test_quarters = data_base.quarters[train_end_idx + 1: train_end_idx + 1 + horizon]
    true_future_raw = data_base.raw_values[train_end_idx + 1: train_end_idx + 1 + horizon]

    print(
        f"\n=== pair={pair} h={horizon} train_end={train_end_quarter} "
        f"test={test_quarters[0]}--{test_quarters[-1]} device={device} ===",
        flush=True,
    )

    rows: list[dict] = []
    detail_rows: list[dict] = []
    pred_raw_by_model_seed: dict[tuple[str, int], np.ndarray] = {}

    for seed in args.seeds:
        # M1 direct solution-network PINN for K50/K75/K100.
        for k_label in SELECTED_M1_KS:
            data = data_by_k[k_label]
            result = train_direct_pinn(
                data=data,
                train_end_idx=train_end_idx,
                horizon=horizon,
                seed=seed,
                device=device,
                use_overlap=True,
                future_o_mode=args.future_o_mode,
                hidden_width=args.pinn_hidden_width,
                hidden_layers=args.pinn_hidden_layers,
                collocation_points=args.collocation_points,
                warmup_steps=args.pinn_warmup_steps,
                full_steps=args.pinn_full_steps,
                learning_rate=args.pinn_lr,
                lr_decay_steps=args.pinn_lr_decay_steps,
                lr_decay_factor=args.pinn_lr_decay_factor,
                curve_weight=args.curve_weight,
                ode_weight=args.ode_weight,
                boundary_weight=args.boundary_weight,
                rate_l2_weight=args.rate_l2_weight,
                grad_clip=args.grad_clip,
            )
            future_norm = forecast_direct_pinn(
                model=result["torch_model"],
                data=data,
                train_end_idx=train_end_idx,
                horizon=horizon,
                device=device,
                clip_norm=args.clip_norm,
                clip_low=args.clip_low,
                clip_high=args.clip_high,
            )
            future_raw = np.maximum(inverse_pair(data, future_norm), 0.0)
            metrics = compute_pair_metrics(true_future_raw, future_raw)
            model_label = f"M1-Direct-{k_label}"

            row = {
                "K": k_label,
                "pair": pair,
                "model": model_label,
                "seed": seed,
                "horizon": horizon,
                "train_end_quarter": train_end_quarter,
                "test_quarters": ";".join(test_quarters),
                "forecast_mode": "direct_network_evaluation",
                "future_o_mode": args.future_o_mode,
                "overlap_csv": k_csv_map[k_label],
                "curve_loss": result["curve_loss"],
                "ode_loss": result["ode_loss"],
                "boundary_loss": result["boundary_loss"],
                "total_loss": result["total_loss"],
                "rate_left": result["rate_left"],
                "rate_right": result["rate_right"],
                **metrics,
            }
            rows.append(row)
            pred_raw_by_model_seed[(model_label, seed)] = future_raw
            print(
                f"[{pair} h={horizon}] {model_label} seed={seed} "
                f"MAE={row['raw_curve_mae']:.3f} "
                f"Macro-NMAE={row['macro_nmae_percent']:.3f}%",
                flush=True,
            )

        # M0 direct no-overlap PINN with exactly the same direct protocol.
        result = train_direct_pinn(
            data=data_base,
            train_end_idx=train_end_idx,
            horizon=horizon,
            seed=seed,
            device=device,
            use_overlap=False,
            future_o_mode="last",
            hidden_width=args.pinn_hidden_width,
            hidden_layers=args.pinn_hidden_layers,
            collocation_points=args.collocation_points,
            warmup_steps=args.pinn_warmup_steps,
            full_steps=args.pinn_full_steps,
            learning_rate=args.pinn_lr,
            lr_decay_steps=args.pinn_lr_decay_steps,
            lr_decay_factor=args.pinn_lr_decay_factor,
            curve_weight=args.curve_weight,
            ode_weight=args.ode_weight,
            boundary_weight=args.boundary_weight,
            rate_l2_weight=args.rate_l2_weight,
            grad_clip=args.grad_clip,
        )
        future_norm = forecast_direct_pinn(
            model=result["torch_model"],
            data=data_base,
            train_end_idx=train_end_idx,
            horizon=horizon,
            device=device,
            clip_norm=args.clip_norm,
            clip_low=args.clip_low,
            clip_high=args.clip_high,
        )
        future_raw = np.maximum(inverse_pair(data_base, future_norm), 0.0)
        metrics = compute_pair_metrics(true_future_raw, future_raw)
        model_label = "M0-Direct-NoOverlap"
        row = {
            "K": "-",
            "pair": pair,
            "model": model_label,
            "seed": seed,
            "horizon": horizon,
            "train_end_quarter": train_end_quarter,
            "test_quarters": ";".join(test_quarters),
            "forecast_mode": "direct_network_evaluation",
            "future_o_mode": "not_used",
            "overlap_csv": "not_used",
            "curve_loss": result["curve_loss"],
            "ode_loss": result["ode_loss"],
            "boundary_loss": result["boundary_loss"],
            "total_loss": result["total_loss"],
            "rate_left": result["rate_left"],
            "rate_right": result["rate_right"],
            **metrics,
        }
        rows.append(row)
        pred_raw_by_model_seed[(model_label, seed)] = future_raw
        print(
            f"[{pair} h={horizon}] {model_label} seed={seed} "
            f"MAE={row['raw_curve_mae']:.3f} "
            f"Macro-NMAE={row['macro_nmae_percent']:.3f}%",
            flush=True,
        )

        # Existing data-driven baselines.
        for model_name in ["CNN", "NBEATS", "InformerLite"]:
            if model_name == "NBEATS":
                nn_result = train_nbeats_direct(
                    data=data_base,
                    train_end_idx=train_end_idx,
                    horizon=horizon,
                    seed=seed,
                    device=device,
                    seq_len=args.seq_len,
                    epochs=args.nbeats_epochs,
                    learning_rate=args.nbeats_lr,
                    hidden_dim=args.nbeats_hidden_dim,
                    num_blocks=args.nbeats_blocks,
                    num_layers=args.nbeats_layers,
                    dropout=args.dropout,
                    grad_clip=args.grad_clip,
                    clip_norm=args.clip_norm,
                    clip_low=args.clip_low,
                    clip_high=args.clip_high,
                )
                future_norm = nn_result["future_norm"]
            else:
                nn_result = train_one_step_model(
                    model_name=model_name,
                    data=data_base,
                    train_end_idx=train_end_idx,
                    seed=seed,
                    device=device,
                    seq_len=args.seq_len,
                    epochs=args.nn_epochs,
                    learning_rate=args.nn_lr,
                    hidden_dim=args.nn_hidden_dim,
                    nhead=args.nhead,
                    layers=args.layers,
                    dropout=args.dropout,
                    cnn_kernel_size=args.cnn_kernel_size,
                    grad_clip=args.grad_clip,
                )
                future_norm = forecast_neural_recursive(
                    model=nn_result["torch_model"],
                    data=data_base,
                    train_end_idx=train_end_idx,
                    horizon=horizon,
                    seq_len=args.seq_len,
                    device=device,
                    clip_norm=args.clip_norm,
                    clip_low=args.clip_low,
                    clip_high=args.clip_high,
                )

            future_raw = np.maximum(inverse_pair(data_base, future_norm), 0.0)
            metrics = compute_pair_metrics(true_future_raw, future_raw)
            row = {
                "K": "-",
                "pair": pair,
                "model": model_name,
                "seed": seed,
                "horizon": horizon,
                "train_end_quarter": train_end_quarter,
                "test_quarters": ";".join(test_quarters),
                "forecast_mode": "neural_baseline",
                "future_o_mode": "not_used",
                "overlap_csv": "not_used",
                "curve_loss": nn_result["curve_loss"],
                "ode_loss": np.nan,
                "boundary_loss": np.nan,
                "total_loss": np.nan,
                "rate_left": np.nan,
                "rate_right": np.nan,
                **metrics,
            }
            rows.append(row)
            pred_raw_by_model_seed[(model_name, seed)] = future_raw
            print(
                f"[{pair} h={horizon}] {model_name} seed={seed} "
                f"MAE={row['raw_curve_mae']:.3f} "
                f"Macro-NMAE={row['macro_nmae_percent']:.3f}%",
                flush=True,
            )

    for (model_name, seed), pred_raw in pred_raw_by_model_seed.items():
        model_k = model_name.rsplit("-", 1)[-1] if model_name.startswith("M1-Direct-K") else "-"
        for i, quarter in enumerate(test_quarters):
            detail_rows.append({
                "K": model_k,
                "pair": pair,
                "model": model_name,
                "seed": seed,
                "horizon": horizon,
                "train_end_quarter": train_end_quarter,
                "quarter": quarter,
                "left_field": data_base.labels[0],
                "right_field": data_base.labels[1],
                "true_left_raw": true_future_raw[i, 0],
                "true_right_raw": true_future_raw[i, 1],
                "pred_left_raw": pred_raw[i, 0],
                "pred_right_raw": pred_raw[i, 1],
            })

    result_df = pd.DataFrame(rows)
    detail_df = pd.DataFrame(detail_rows)
    summary_df = summarize_seedlevel(
        result_df, pair, horizon, train_end_quarter, test_quarters
    )
    ttest_df = build_ttest_table(
        result_df=result_df,
        pair=pair,
        horizon=horizon,
        train_end_quarter=train_end_quarter,
        test_quarters=test_quarters,
        metric=args.ttest_metric,
    )

    result_df.to_csv(outdir / "forecast_seedlevel_direct_pinn.csv", index=False, encoding="utf-8-sig")
    detail_df.to_csv(outdir / "forecast_detail_direct_pinn.csv", index=False, encoding="utf-8-sig")
    summary_df.to_csv(outdir / "forecast_summary_direct_pinn.csv", index=False, encoding="utf-8-sig")
    ttest_df.to_csv(outdir / "forecast_ttests_direct_pinn.csv", index=False, encoding="utf-8-sig")

    plot_pred_raw: dict[str, np.ndarray] = {}
    for model_name in ORDERED_MODELS:
        stack = [
            pred
            for (candidate_name, _), pred in pred_raw_by_model_seed.items()
            if candidate_name == model_name
        ]
        if stack:
            plot_pred_raw[model_name] = np.mean(np.stack(stack, axis=0), axis=0)

    # Per-pair predictions are retained in CSV. The publication-ready combined
    # forecasting figure is generated after all field pairs have finished.
    return summary_df.to_dict(orient="records"), ttest_df.to_dict(orient="records")



# ============================================================
# 8. Publication-ready aggregate figures
# ============================================================

PAIR_DISPLAY = {
    "cnn_gnn": "CNN–GNN",
    "cnn_transformer": "CNN–Transformer",
    "gnn_transformer": "GNN–Transformer",
}

PAPER_MODEL_ORDER = [
    "M1-Direct-K50",
    "M1-Direct-K75",
    "M1-Direct-K100",
    "M0-Direct-NoOverlap",
    "CNN",
    "NBEATS",
    "InformerLite",
]

PAPER_MODEL_LABELS = {
    "M1-Direct-K50": "M1-K50",
    "M1-Direct-K75": "M1-K75",
    "M1-Direct-K100": "M1-K100",
    "M0-Direct-NoOverlap": "M0",
    "CNN": "CNN",
    "NBEATS": "N-BEATS",
    "InformerLite": "InformerLite",
}

MACRO_NMAE_MODEL_LABELS = {
    "M1-Direct-K50": "M1-K50",
    "M1-Direct-K75": "M1-K75",
    "M1-Direct-K100": "M1-K100",
    "M0-Direct-NoOverlap": "M0",
    "CNN": "CNN",
    "NBEATS": "N-BEATS",
    "InformerLite": "InformerLite",
}

# Colors and markers follow the supplied paper-figure example.
PAPER_STYLES = {
    "True": dict(color="black", marker="o", linestyle="-", linewidth=2.25, markersize=5.1),
    "Best M1": dict(color="#1F6FB5", marker="s", linestyle="--", linewidth=2.65, markersize=5.5),
    "M0-Direct-NoOverlap": dict(color="#8A8A8A", marker="s", linestyle="--", linewidth=2.25, markersize=4.9, alpha=0.72),
    "CNN": dict(color="#2A9D4B", marker="^", linestyle="-.", linewidth=2.25, markersize=5.2, alpha=0.72),
    "NBEATS": dict(color="#E23D35", marker="D", linestyle=":", linewidth=2.25, markersize=4.8, alpha=0.72),
    "InformerLite": dict(color="#8F61D3", marker="v", linestyle="--", linewidth=2.25, markersize=5.2, alpha=0.72),
}

BAR_COLORS = {
    "M1-Direct-K50": "#6E8FB2",
    "M1-Direct-K75": "#7DA494",
    "M1-Direct-K100": "#EAB67A",
    "M0-Direct-NoOverlap": "#E5A79A",
    "CNN": "#C16E71",
    "NBEATS": "#ABC8E5",
    "InformerLite": "#D8A0C1",
}

MACRO_NMAE_STYLE = {
    "figsize": (7.16, 2.75),
    "bar_width": 0.66,
    "bar_edge": "#555555",
    "error_color": "#333333",
    "spine_color": "#333333",
    "panel_title_size": 9.0,
    "axis_label_size": 8.5,
    "tick_label_size": 7.5,
    "error_linewidth": 0.75,
    "cap_size": 3.5,
}


def _set_paper_rcparams() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "STIXGeneral"],
        "mathtext.fontset": "stix",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.titlesize": MACRO_NMAE_STYLE["panel_title_size"],
        "axes.labelsize": MACRO_NMAE_STYLE["axis_label_size"],
        "xtick.labelsize": MACRO_NMAE_STYLE["tick_label_size"],
        "ytick.labelsize": MACRO_NMAE_STYLE["tick_label_size"],
        "legend.fontsize": 9.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def _read_pair_outputs(output_root: Path, pair: str, horizon: int):
    pair_dir = output_root / pair / f"h{horizon}"
    summary_path = pair_dir / "forecast_summary_direct_pinn.csv"
    detail_path = pair_dir / "forecast_detail_direct_pinn.csv"
    preprocessed_path = pair_dir / "preprocessed_timeseries_K100.csv"

    for path in [summary_path, detail_path, preprocessed_path]:
        if not path.exists():
            raise FileNotFoundError(f"Required figure input does not exist: {path}")

    summary = pd.read_csv(summary_path)
    detail = pd.read_csv(detail_path)
    preprocessed = pd.read_csv(preprocessed_path)
    return summary, detail, preprocessed


def _select_best_m1(summary: pd.DataFrame, metric: str) -> str:
    candidates = summary[summary["model"].isin(DIRECT_M1_MODELS)].copy()
    if candidates.empty:
        raise ValueError("No M1 direct-PINN rows found when selecting Best M1.")
    if metric not in candidates.columns:
        raise ValueError(f"Selection metric {metric!r} is absent from summary columns.")
    candidates[metric] = pd.to_numeric(candidates[metric], errors="coerce")
    candidates = candidates.dropna(subset=[metric]).sort_values([metric, "model"])
    if candidates.empty:
        raise ValueError(f"No finite M1 values are available for {metric}.")
    return str(candidates.iloc[0]["model"])


def _macro_nmae_tick_step(ymax: float) -> float:
    for step in (2.0, 2.5, 5.0):
        if ymax / step <= 6:
            return step
    return 5.0


def plot_macro_nmae_panel(
    ax,
    means: np.ndarray,
    errors: np.ndarray,
    labels: list[str],
    title: str,
    y_top: float,
) -> None:
    if len(means) != len(labels):
        raise ValueError("Macro-NMAE means and labels must have identical length.")
    if len(means) == 0 or not np.all(np.isfinite(means)):
        raise ValueError(f"Macro-NMAE panel {title!r} contains NaN or empty values.")
    if len(errors) != len(means):
        raise ValueError("Macro-NMAE means and errors must have identical length.")
    if not np.all(np.isfinite(errors)):
        raise ValueError(f"Macro-NMAE panel {title!r} contains NaN error values.")

    style = MACRO_NMAE_STYLE
    x = np.arange(len(labels))
    colors = [BAR_COLORS[m] for m in PAPER_MODEL_ORDER]

    ax.bar(
        x,
        means,
        yerr=errors,
        width=style["bar_width"],
        color=colors,
        edgecolor=style["bar_edge"],
        linewidth=0.55,
        capsize=style["cap_size"],
        error_kw={
            "ecolor": style["error_color"],
            "elinewidth": style["error_linewidth"],
            "capthick": style["error_linewidth"],
        },
    )

    ax.set_title(title, fontsize=style["panel_title_size"], fontweight="bold", pad=4.0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", rotation_mode="anchor")
    ax.set_ylim(0.0, y_top)
    ax.grid(axis="y", linestyle="--", linewidth=0.45, alpha=0.35)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", labelsize=style["tick_label_size"], width=0.55, length=2.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for spine_name in ("left", "bottom"):
        ax.spines[spine_name].set_linewidth(0.7)
        ax.spines[spine_name].set_color(style["spine_color"])


def _mean_future_prediction(
    detail: pd.DataFrame,
    model_name: str,
    quarters: list[str],
    field_side: str,
) -> np.ndarray:
    pred_col = "pred_left_raw" if field_side == "left" else "pred_right_raw"
    subset = detail[detail["model"] == model_name].copy()
    if subset.empty:
        raise ValueError(f"No prediction rows found for model={model_name}.")
    means = subset.groupby("quarter", sort=False)[pred_col].mean()
    missing = [q for q in quarters if q not in means.index]
    if missing:
        raise ValueError(f"Missing prediction quarters for {model_name}: {missing}")
    return means.reindex(quarters).to_numpy(np.float64)


def plot_six_sequence_forecasting_curves(
    output_root: Path,
    pairs: list[str],
    horizon: int,
    selection_metric: str,
    history_quarters: int,
    dpi: int,
) -> pd.DataFrame:
    """Create the requested 2 x 3 four-quarter forecasting figure."""
    if len(pairs) != 3:
        raise ValueError(
            "The six-panel paper figure requires exactly three pairs in the desired column order."
        )
    if history_quarters < 1:
        raise ValueError("history_quarters must be positive.")

    _set_paper_rcparams()
    figure_dir = output_root / "paper_figures"
    ensure_dir(figure_dir)

    fig, axes = plt.subplots(2, 3, figsize=(15.8, 8.2), constrained_layout=False)
    panel_letters = [["a", "c", "e"], ["b", "d", "f"]]
    selection_rows: list[dict] = []

    legend_handles = None
    legend_labels = None

    for col, pair in enumerate(pairs):
        summary, detail, preprocessed = _read_pair_outputs(output_root, pair, horizon)
        best_m1 = _select_best_m1(summary, selection_metric)

        train_end_quarter = str(summary.iloc[0]["train_end_quarter"])
        test_quarters = str(summary.iloc[0]["test_quarters"]).split(";")
        if len(test_quarters) != horizon:
            raise ValueError(
                f"Expected {horizon} test quarters for {pair}, got {test_quarters}."
            )

        quarter_list = preprocessed["quarter"].astype(str).tolist()
        if train_end_quarter not in quarter_list:
            raise ValueError(f"train_end_quarter={train_end_quarter} absent for {pair}.")
        train_end_idx = quarter_list.index(train_end_quarter)
        display_end_idx = train_end_idx + horizon
        display_quarters = quarter_list[train_end_idx + 1: display_end_idx + 1]

        selection_row = summary[summary["model"] == best_m1].iloc[0]
        selection_rows.append({
            "pair": pair,
            "pair_display": PAIR_DISPLAY.get(pair, pair),
            "horizon": horizon,
            "selection_metric": selection_metric,
            "best_m1_model": best_m1,
            "best_m1_K": best_m1.rsplit("-", 1)[-1],
            "best_m1_metric_mean": float(selection_row[selection_metric]),
        })

        baseline_models = [
            ("M0-Direct-NoOverlap", "M0-Direct-NoOverlap"),
            ("CNN", "CNN"),
            ("NBEATS", "NBEATS"),
            ("InformerLite", "InformerLite"),
        ]

        field_specs = [
            (0, "left", PAIR_SPECS[pair]["left_label"], "left_raw"),
            (1, "right", PAIR_SPECS[pair]["right_label"], "right_raw"),
        ]

        for row, side, field_label, raw_col in field_specs:
            ax = axes[row, col]
            x = np.arange(len(display_quarters))
            true_values = preprocessed[raw_col].to_numpy(np.float64)[
                train_end_idx + 1: display_end_idx + 1
            ]
            panel_values = [true_values]

            # Draw baselines first so the proposed M1 and true curve remain visually prominent.
            for legend_name, model_name in baseline_models:
                future_pred = _mean_future_prediction(
                    detail=detail,
                    model_name=model_name,
                    quarters=test_quarters,
                    field_side=side,
                )
                panel_values.append(future_pred)
                style = PAPER_STYLES[legend_name]
                ax.plot(x, future_pred, label=legend_name, zorder=2, **style)

            best_m1_future = _mean_future_prediction(
                detail=detail,
                model_name=best_m1,
                quarters=test_quarters,
                field_side=side,
            )
            panel_values.append(best_m1_future)
            ax.plot(
                x,
                best_m1_future,
                label="Best M1",
                zorder=7,
                **PAPER_STYLES["Best M1"],
            )

            ax.plot(
                x,
                true_values,
                label="True",
                zorder=6,
                **PAPER_STYLES["True"],
            )

            panel = panel_letters[row][col]
            pair_name = PAIR_DISPLAY.get(pair, pair)
            ax.set_title(
                f"({panel}) {pair_name}: {field_label}",
                fontsize=17,
                fontweight="bold",
                pad=8,
            )
            ax.set_xticks(x)
            ax.set_xticklabels([str(i) for i in range(1, len(display_quarters) + 1)], fontsize=18)
            if row == 1:
                ax.set_xlabel("Quarter", fontsize=20)
            ax.tick_params(axis="y", labelsize=18)
            if col == 0:
                ax.set_ylabel("Publication count", fontsize=20)
            ax.grid(True, axis="both", linestyle="-", linewidth=0.45, alpha=0.28)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.margins(x=0.02)
            y_values = np.concatenate(panel_values)
            y_min = float(np.nanmin(y_values))
            y_max = float(np.nanmax(y_values))
            y_pad = max((y_max - y_min) * 0.055, y_max * 0.015)
            ax.set_ylim(y_min - y_pad, y_max + y_pad)

            if legend_handles is None:
                handles, labels = ax.get_legend_handles_labels()
                desired = ["True", "Best M1", "M0-Direct-NoOverlap", "CNN", "NBEATS", "InformerLite"]
                mapping = dict(zip(labels, handles))
                legend_handles = [mapping[name] for name in desired]
                legend_labels = [
                    "True", "Best M1", "M0-NoOverlap", "CNN", "N-BEATS", "InformerLite"
                ]

    axes[0, 0].legend(
        legend_handles,
        legend_labels,
        loc="upper left",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.015, 0.985),
        borderaxespad=0.0,
        handlelength=2.45,
        columnspacing=1.05,
        labelspacing=0.32,
        fontsize=13.0,
    )
    fig.subplots_adjust(left=0.07, right=0.992, top=0.92, bottom=0.11, wspace=0.18, hspace=0.48)

    png_path = figure_dir / f"paper_four_quarter_forecasting_curves_h{horizon}.png"
    pdf_path = figure_dir / f"paper_four_quarter_forecasting_curves_h{horizon}.pdf"
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    selection_df = pd.DataFrame(selection_rows)
    selection_df.to_csv(
        figure_dir / f"best_m1_by_pair_h{horizon}.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return selection_df


def plot_macro_nmae_bars(
    output_root: Path,
    pairs: list[str],
    horizon: int,
    dpi: int,
) -> pd.DataFrame:
    """Create a three-panel publication-ready Macro-NMAE mean +/- std bar chart."""
    _set_paper_rcparams()
    from matplotlib.ticker import MaxNLocator

    figure_dir = output_root / "paper_figures"
    ensure_dir(figure_dir)

    rows: list[pd.DataFrame] = []
    for pair in pairs:
        summary, _, _ = _read_pair_outputs(output_root, pair, horizon)
        subset = summary[summary["model"].isin(PAPER_MODEL_ORDER)].copy()
        subset["pair"] = pair
        rows.append(subset)

    plot_df = pd.concat(rows, ignore_index=True)
    metric_mean = "macro_nmae_percent_mean"
    metric_std = "macro_nmae_percent_std"
    plot_df[metric_mean] = pd.to_numeric(plot_df[metric_mean], errors="coerce")
    plot_df[metric_std] = pd.to_numeric(plot_df[metric_std], errors="coerce")
    if plot_df.empty or plot_df[metric_mean].isna().any():
        raise ValueError("Macro-NMAE plot data are empty or contain NaN mean values.")
    if plot_df[metric_std].isna().any():
        raise ValueError("Macro-NMAE plot data contain NaN error values.")

    fig, axes = plt.subplots(
        1,
        len(pairs),
        figsize=MACRO_NMAE_STYLE["figsize"],
        sharey=False,
        constrained_layout=False,
        facecolor="white",
    )
    if len(pairs) == 1:
        axes = [axes]

    panel_letters = ["a", "b", "c"]

    for panel_idx, (ax, pair) in enumerate(zip(axes, pairs)):
        pair_df = plot_df[plot_df["pair"] == pair].set_index("model")
        missing = [m for m in PAPER_MODEL_ORDER if m not in pair_df.index]
        if missing:
            raise ValueError(f"Missing Macro-NMAE rows for {pair}: {missing}")
        pair_df = pair_df.loc[PAPER_MODEL_ORDER]

        means = pair_df[metric_mean].to_numpy(np.float64)
        errors = pair_df[metric_std].to_numpy(np.float64)
        labels = [MACRO_NMAE_MODEL_LABELS[m] for m in PAPER_MODEL_ORDER]
        panel_max = float(np.nanmax(means + errors))
        y_top = max(panel_max / 0.88, 1.0)
        plot_macro_nmae_panel(
            ax=ax,
            means=means,
            errors=errors,
            labels=labels,
            title=f"({panel_letters[panel_idx]}) {PAIR_DISPLAY.get(pair, pair)}",
            y_top=y_top,
        )
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        if panel_idx == 0:
            ax.set_ylabel("Macro-NMAE (%)", fontsize=MACRO_NMAE_STYLE["axis_label_size"])

    fig.subplots_adjust(left=0.075, right=0.995, bottom=0.36, top=0.91, wspace=0.16)

    pdf_path = figure_dir / "four_quarter_macro_nmae_barplot.pdf"
    png_path = figure_dir / "four_quarter_macro_nmae_barplot.png"
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.02, facecolor="white")
    fig.savefig(png_path, dpi=600, bbox_inches="tight", pad_inches=0.02, facecolor="white")
    plt.close(fig)

    export_cols = [
        "pair", "model", "runs",
        "macro_nmae_percent_mean", "macro_nmae_percent_std",
        "raw_curve_mae_mean", "raw_curve_mae_std",
        "raw_curve_r2_mean", "raw_curve_ccc_mean",
    ]
    export_df = plot_df[export_cols].copy()
    export_df["model_order"] = export_df["model"].map(
        {name: i for i, name in enumerate(PAPER_MODEL_ORDER)}
    )
    export_df["pair_order"] = export_df["pair"].map(
        {name: i for i, name in enumerate(pairs)}
    )
    export_df = export_df.sort_values(["pair_order", "model_order"]).drop(
        columns=["pair_order", "model_order"]
    )
    export_df.to_csv(
        figure_dir / f"paper_macro_nmae_table_h{horizon}.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return export_df


def generate_paper_figures(args) -> None:
    output_root = Path(args.outdir)
    for horizon in args.horizons:
        selection_df = plot_six_sequence_forecasting_curves(
            output_root=output_root,
            pairs=list(args.pairs),
            horizon=horizon,
            selection_metric=args.best_m1_metric,
            history_quarters=args.plot_history_quarters,
            dpi=args.figure_dpi,
        )
        plot_macro_nmae_bars(
            output_root=output_root,
            pairs=list(args.pairs),
            horizon=horizon,
            dpi=args.figure_dpi,
        )
        print(f"\nPaper figures generated for h={horizon}:")
        print(f"  {output_root / 'paper_figures' / f'paper_four_quarter_forecasting_curves_h{horizon}.pdf'}")
        print(f"  {output_root / 'paper_figures' / 'four_quarter_macro_nmae_barplot.pdf'}")
        print("Best M1 selections:")
        for row in selection_df.to_dict(orient="records"):
            print(
                f"  {row['pair_display']}: {row['best_m1_model']} "
                f"({row['selection_metric']}={row['best_m1_metric_mean']:.6f})"
            )


# ============================================================
# 9. CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--csv-k50", type=str, default=None)
    parser.add_argument("--csv-k75", type=str, default=None)
    parser.add_argument("--csv-k100", type=str, default=None)
    parser.add_argument(
        "--csv-template",
        type=str,
        default="quarterly_counts_and_overlap_{K}_2018Q1_2025Q4.csv",
        help="Used when explicit K-specific paths are omitted.",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="outputs_direct_pinn_30seed_formal",
    )
    parser.add_argument(
        "--pairs",
        nargs="+",
        default=["cnn_gnn", "cnn_transformer", "gnn_transformer"],
        choices=list(PAIR_SPECS.keys()),
    )
    parser.add_argument("--horizons", type=int, nargs="+", default=[4])
    parser.add_argument(
        "--train-mode",
        type=str,
        default="rolling_end",
        choices=["rolling_end", "common"],
    )
    parser.add_argument("--common-train-end", type=str, default="auto")

    parser.add_argument("--minmax-low", type=float, default=0.05)
    parser.add_argument("--minmax-high", type=float, default=0.95)
    parser.add_argument(
        "--overlap-normalization",
        type=str,
        default="auto",
        choices=["auto", "none", "train_minmax", "full_minmax"],
    )
    parser.add_argument(
        "--future-o-mode",
        type=str,
        default="last",
        choices=["last", "true"],
        help="Use last for the leakage-free experiment. true is diagnostic only.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(range(20260221, 20260251)),
    )
    parser.add_argument("--seq-len", type=int, default=8)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--ttest-metric", type=str, default="macro_nmae_percent")

    # Direct PINN settings.
    parser.add_argument("--pinn-hidden-width", type=int, default=32)
    parser.add_argument("--pinn-hidden-layers", type=int, default=2)
    parser.add_argument("--collocation-points", type=int, default=128)
    parser.add_argument("--pinn-warmup-steps", type=int, default=1000)
    parser.add_argument("--pinn-full-steps", type=int, default=6000)
    parser.add_argument("--pinn-lr", type=float, default=1e-3)
    parser.add_argument("--pinn-lr-decay-steps", type=int, default=2000)
    parser.add_argument("--pinn-lr-decay-factor", type=float, default=0.7)
    parser.add_argument("--curve-weight", type=float, default=1.0)
    parser.add_argument("--ode-weight", type=float, default=1.0)
    parser.add_argument("--boundary-weight", type=float, default=1.0)
    parser.add_argument("--rate-l2-weight", type=float, default=0.0)

    # CNN / InformerLite settings.
    parser.add_argument("--nn-hidden-dim", type=int, default=64)
    parser.add_argument("--nn-epochs", type=int, default=500)
    parser.add_argument("--nn-lr", type=float, default=1e-3)
    parser.add_argument("--cnn-kernel-size", type=int, default=3)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)

    # N-BEATS settings.
    parser.add_argument("--nbeats-hidden-dim", type=int, default=64)
    parser.add_argument("--nbeats-blocks", type=int, default=3)
    parser.add_argument("--nbeats-layers", type=int, default=4)
    parser.add_argument("--nbeats-epochs", type=int, default=800)
    parser.add_argument("--nbeats-lr", type=float, default=1e-3)

    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--clip-norm", action="store_true")
    parser.add_argument("--clip-low", type=float, default=0.0)
    parser.add_argument("--clip-high", type=float, default=1.5)

    # Publication-ready figure settings.
    parser.add_argument(
        "--best-m1-metric",
        type=str,
        default="macro_nmae_percent_mean",
        choices=["macro_nmae_percent_mean", "raw_curve_mae_mean"],
        help="Metric used to choose the pair-specific Best M1 curve.",
    )
    parser.add_argument("--plot-history-quarters", type=int, default=4)
    parser.add_argument("--figure-dpi", type=int, default=300)
    parser.add_argument("--skip-paper-figures", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.k_csv_map = get_k_csv_map(args)
    ensure_dir(args.outdir)

    common_train_end_idx = None
    if args.train_mode == "common":
        base_csv = args.k_csv_map["K100"] if "K100" in args.k_csv_map else next(iter(args.k_csv_map.values()))
        raw = pd.read_csv(base_csv)
        quarter_col = find_col(raw, ["quarter", "时间", "季度"])
        quarters = raw[quarter_col].astype(str).tolist()

        if args.common_train_end.lower() == "auto":
            common_train_end_idx = len(quarters) - max(args.horizons) - 1
        else:
            if args.common_train_end not in quarters:
                raise ValueError(f"common_train_end={args.common_train_end} not found.")
            common_train_end_idx = quarters.index(args.common_train_end)

    print("Formal 30-seed direct-PINN comparison models:")
    print("  " + ", ".join(ORDERED_MODELS))
    print(f"Seeds: {args.seeds}  (n={len(args.seeds)})")
    print(f"Future overlap mode: {args.future_o_mode}")
    print("CSV map:")
    for k, path in args.k_csv_map.items():
        print(f"  {k}: {path}")

    all_summary_rows: list[dict] = []
    all_ttest_rows: list[dict] = []
    for pair in args.pairs:
        for horizon in args.horizons:
            summary_rows, ttest_rows = run_pair_horizon(
                args, pair, horizon, common_train_end_idx
            )
            all_summary_rows.extend(summary_rows)
            all_ttest_rows.extend(ttest_rows)

    summary_path = Path(args.outdir) / "all_pairs_forecast_summary_direct_pinn_30seed.csv"
    ttest_path = Path(args.outdir) / "all_pairs_forecast_ttests_direct_pinn_30seed.csv"
    pd.DataFrame(all_summary_rows).to_csv(summary_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(all_ttest_rows).to_csv(ttest_path, index=False, encoding="utf-8-sig")

    if not args.skip_paper_figures:
        generate_paper_figures(args)

    print("\nDone.")
    print(f"All-pair 30-seed summary      : {summary_path}")
    print(f"Formal paired t-tests        : {ttest_path}")
    print(f"Output root                  : {args.outdir}")


if __name__ == "__main__":
    main()
