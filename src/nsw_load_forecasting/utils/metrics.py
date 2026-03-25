from __future__ import annotations

from typing import Dict, Iterable

import numpy as np


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_pred - y_true)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))


def smape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    num = 2.0 * np.abs(y_pred - y_true)
    den = np.abs(y_pred) + np.abs(y_true) + eps
    return float(100.0 * np.mean(num / den))


def mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    mask = np.abs(y_true) > eps
    if not np.any(mask):
        return float("nan")
    return float(100.0 * np.mean(np.abs((y_pred[mask] - y_true[mask]) / y_true[mask])))


def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, q: float) -> float:
    diff = y_true - y_pred
    return float(np.mean(np.maximum(q * diff, (q - 1) * diff)))


def coverage(y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    inside = (y_true >= lower) & (y_true <= upper)
    return float(np.mean(inside))


def interval_width(lower: np.ndarray, upper: np.ndarray) -> float:
    return float(np.mean(upper - lower))


def metric_bundle(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "mape": mape(y_true, y_pred),
        "smape": smape(y_true, y_pred),
    }
