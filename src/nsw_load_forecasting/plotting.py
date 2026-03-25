from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import pandas as pd


def plot_daily_comparison(df: pd.DataFrame, pred_cols: Dict[str, str], out_path: str | Path) -> None:
    daily = df.resample("D").mean(numeric_only=True)
    plt.figure(figsize=(16, 6))
    plt.plot(daily.index, daily["actual"], label="Actual", linewidth=2)
    for label, col in pred_cols.items():
        plt.plot(daily.index, daily[col], label=label, linewidth=1.5)
    plt.title("NSW Load Forecasting: Daily Average Comparison")
    plt.xlabel("Date")
    plt.ylabel("Load (MW)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_error_histograms(df: pd.DataFrame, pred_cols: Dict[str, str], out_path: str | Path) -> None:
    fig, axes = plt.subplots(1, len(pred_cols), figsize=(6 * len(pred_cols), 4))
    if len(pred_cols) == 1:
        axes = [axes]
    for ax, (label, col) in zip(axes, pred_cols.items()):
        err = df[col] - df["actual"]
        ax.hist(err, bins=60, alpha=0.8)
        ax.axvline(0, linestyle="--")
        ax.set_title(label)
        ax.set_xlabel("Forecast error (MW)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
