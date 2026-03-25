from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .utils.metrics import coverage, interval_width, metric_bundle, pinball_loss


def predict_deterministic(model: torch.nn.Module, loader: DataLoader, task: str, device: str):
    model.eval()
    preds, trues, bases = [], [], []
    with torch.no_grad():
        for batch in loader:
            if task == "residual":
                x, baseline, residual = batch
                x = x.to(device)
                baseline = baseline.to(device)
                residual = residual.to(device)
                pred_resid = model(x)
                pred = baseline + pred_resid
                true = baseline + residual
                bases.append(baseline.cpu().numpy())
            else:
                x, y = batch
                x = x.to(device)
                y = y.to(device)
                pred = model(x)
                true = y
            preds.append(pred.cpu().numpy())
            trues.append(true.cpu().numpy())
    result = {
        "y_pred": np.concatenate(preds, axis=0),
        "y_true": np.concatenate(trues, axis=0),
    }
    if bases:
        result["baseline"] = np.concatenate(bases, axis=0)
    return result


@torch.no_grad()
def predict_diffusion(model: torch.nn.Module, loader: DataLoader, device: str, n_samples: int = 50):
    model.eval()
    mean_preds, trues, p10s, p50s, p90s, bases = [], [], [], [], [], []
    for x, baseline, residual in loader:
        x = x.to(device)
        baseline = baseline.to(device)
        residual = residual.to(device)
        samples = model.sample(x, n_samples=n_samples)
        sampled_pred = baseline.unsqueeze(1) + samples
        mean_pred = sampled_pred.mean(dim=1)
        p10 = sampled_pred.quantile(0.1, dim=1)
        p50 = sampled_pred.quantile(0.5, dim=1)
        p90 = sampled_pred.quantile(0.9, dim=1)
        true = baseline + residual
        mean_preds.append(mean_pred.cpu().numpy())
        p10s.append(p10.cpu().numpy())
        p50s.append(p50.cpu().numpy())
        p90s.append(p90.cpu().numpy())
        trues.append(true.cpu().numpy())
        bases.append(baseline.cpu().numpy())
    return {
        "y_pred": np.concatenate(mean_preds, axis=0),
        "y_true": np.concatenate(trues, axis=0),
        "p10": np.concatenate(p10s, axis=0),
        "p50": np.concatenate(p50s, axis=0),
        "p90": np.concatenate(p90s, axis=0),
        "baseline": np.concatenate(bases, axis=0),
    }


def flatten_predictions(arr: np.ndarray) -> np.ndarray:
    return arr.reshape(-1)


def make_regime_flags(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    out = df.copy()
    out["is_peak"] = out[target_col] >= out[target_col].quantile(0.9)
    out["ramp_abs"] = out[target_col].diff().abs().fillna(0.0)
    out["is_ramp"] = out["ramp_abs"] >= out["ramp_abs"].quantile(0.9)
    out["is_hot"] = out.filter(like="temperature").max(axis=1) >= out.filter(like="temperature").max(axis=1).quantile(0.9) if len(out.filter(like="temperature").columns) else False
    if "is_public_holiday" not in out.columns:
        out["is_public_holiday"] = False
    if "is_weekend" not in out.columns:
        out["is_weekend"] = out.index.dayofweek >= 5
    return out


def regime_metrics(aligned_df: pd.DataFrame, pred_col: str, true_col: str = "actual") -> Dict[str, Dict[str, float]]:
    regimes = {
        "overall": np.ones(len(aligned_df), dtype=bool),
        "peak": aligned_df["is_peak"].to_numpy(dtype=bool),
        "ramp": aligned_df["is_ramp"].to_numpy(dtype=bool),
        "weekend": aligned_df["is_weekend"].to_numpy(dtype=bool),
        "holiday": aligned_df["is_public_holiday"].to_numpy(dtype=bool),
        "hot": aligned_df["is_hot"].to_numpy(dtype=bool) if isinstance(aligned_df["is_hot"], pd.Series) else np.zeros(len(aligned_df), dtype=bool),
    }
    results: Dict[str, Dict[str, float]] = {}
    for name, mask in regimes.items():
        if mask.sum() == 0:
            continue
        y_true = aligned_df.loc[mask, true_col].to_numpy()
        y_pred = aligned_df.loc[mask, pred_col].to_numpy()
        results[name] = metric_bundle(y_true, y_pred)
    return results


def probabilistic_metrics(y_true: np.ndarray, p10: np.ndarray, p50: np.ndarray, p90: np.ndarray) -> Dict[str, float]:
    flat_true = flatten_predictions(y_true)
    flat_p10 = flatten_predictions(p10)
    flat_p50 = flatten_predictions(p50)
    flat_p90 = flatten_predictions(p90)
    return {
        "coverage_80": coverage(flat_true, flat_p10, flat_p90),
        "interval_width_80": interval_width(flat_p10, flat_p90),
        "pinball_10": pinball_loss(flat_true, flat_p10, 0.1),
        "pinball_50": pinball_loss(flat_true, flat_p50, 0.5),
        "pinball_90": pinball_loss(flat_true, flat_p90, 0.9),
    }
