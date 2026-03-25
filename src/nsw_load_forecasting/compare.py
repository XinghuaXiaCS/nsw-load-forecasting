from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pandas as pd

from .config import Config
from .data import load_feature_frame
from .evaluation import make_regime_flags, probabilistic_metrics, regime_metrics
from .plotting import plot_daily_comparison, plot_error_histograms
from .utils.io import save_csv, save_json


def compare_runs(cfg: Config) -> Dict[str, str]:
    output_root = cfg.output_root
    feature_df = load_feature_frame(cfg.data["feature_file"], cfg.data["target_col"])
    feature_df = feature_df.rename(columns={cfg.data["target_col"]: "actual"})
    feature_df = make_regime_flags(feature_df, target_col="actual")

    run_dirs = [p for p in output_root.iterdir() if p.is_dir() and (p / "predictions.csv").exists()]
    if not run_dirs:
        raise RuntimeError("No saved runs found in results/")

    comparison_rows = []
    aligned = None
    pred_cols: Dict[str, str] = {}
    regime_results = {}
    probabilistic = {}

    for run_dir in sorted(run_dirs):
        model_name = run_dir.name
        pred_df = pd.read_csv(run_dir / "predictions.csv")
        pred_df["timestamp"] = pd.to_datetime(pred_df["timestamp"])
        pred_df = pred_df.drop_duplicates(subset=["timestamp"]).set_index("timestamp").sort_index()
        col_name = f"pred_{model_name}"
        pred_cols[model_name] = col_name

        joined = feature_df[["actual", "is_peak", "is_ramp", "is_weekend", "is_public_holiday", "is_hot"]].join(pred_df[["prediction"]].rename(columns={"prediction": col_name}), how="inner")
        if aligned is None:
            aligned = joined.copy()
        else:
            aligned = aligned.join(joined[[col_name]], how="inner")

        metrics = json_or_empty(run_dir / "metrics.json")
        metrics["model"] = model_name
        comparison_rows.append(metrics)
        regime_results[model_name] = regime_metrics(joined.dropna(), pred_col=col_name)
        if {"p10", "p50", "p90"}.issubset(pred_df.columns):
            temp = feature_df[["actual"]].join(pred_df[["p10", "p50", "p90"]], how="inner").dropna()
            probabilistic[model_name] = probabilistic_metrics(temp["actual"].to_numpy(), temp["p10"].to_numpy(), temp["p50"].to_numpy(), temp["p90"].to_numpy())

    comparison_df = pd.DataFrame(comparison_rows).sort_values("mae")
    save_csv(comparison_df, output_root / "comparison_metrics.csv")
    save_json(regime_results, output_root / "comparison_regimes.json")
    if probabilistic:
        save_json(probabilistic, output_root / "comparison_probabilistic.json")

    aligned = aligned.dropna()
    plot_daily_comparison(aligned, pred_cols=pred_cols, out_path=output_root / "comparison_daily.png")
    plot_error_histograms(aligned, pred_cols=pred_cols, out_path=output_root / "comparison_errors.png")
    return {"result_dir": str(output_root)}


def json_or_empty(path: Path) -> Dict:
    if not path.exists():
        return {}
    import json
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
