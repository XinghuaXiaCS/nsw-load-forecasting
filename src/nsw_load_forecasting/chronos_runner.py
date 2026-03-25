from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from chronos import Chronos2Pipeline

from .config import Config
from .data import build_split_spec, build_window_spec, load_feature_frame
from .utils.io import save_csv, save_json
from .utils.metrics import metric_bundle


def run_chronos(cfg: Config) -> Dict[str, str]:
    data_cfg = cfg.data
    split_spec = build_split_spec(cfg.splits)
    window_spec = build_window_spec(cfg.window)

    # Load the feature dataframe and restore timestamp as a normal column.
    feature_df = load_feature_frame(
        data_cfg["feature_file"],
        data_cfg["target_col"],
        timestamp_col=data_cfg["timestamp_col"],
    ).copy()

    feature_df = feature_df.reset_index()
    if data_cfg["timestamp_col"] not in feature_df.columns:
        feature_df = feature_df.rename(columns={"index": data_cfg["timestamp_col"]})

    if data_cfg["target_col"] not in feature_df.columns:
        raise KeyError(f"Target column '{data_cfg['target_col']}' not found in feature dataframe")

    # Chronos predict_df requires an id column even for a single series.
    feature_df["id"] = "nsw_load"

    # Keep only future covariates that actually exist in the dataframe.
    known_future_covariates: List[str] = [
        c for c in data_cfg.get("known_future_covariates", [])
        if c in feature_df.columns
    ]

    # Create and place the pipeline on GPU if available.
    pipeline = Chronos2Pipeline.from_pretrained(
        "amazon/chronos-2",
        device_map="cuda" if torch.cuda.is_available() else "cpu",
    )

    full_df = feature_df.copy().reset_index(drop=True)
    ts = pd.to_datetime(full_df[data_cfg["timestamp_col"]])

    lookback = window_spec.lookback
    horizon = window_spec.horizon
    stride = window_spec.stride

    all_preds = []
    all_true = []
    all_timestamps = []

    max_start = len(full_df) - horizon
    for forecast_start_idx in range(lookback, max_start + 1, stride):
        forecast_start_ts = ts.iloc[forecast_start_idx]
        forecast_end_ts = ts.iloc[forecast_start_idx + horizon - 1]

        # Only evaluate windows fully contained in the configured test period.
        if forecast_start_ts < split_spec.test_start or forecast_end_ts > split_spec.test_end:
            continue

        context = full_df.iloc[forecast_start_idx - lookback : forecast_start_idx].copy()
        future = full_df.iloc[forecast_start_idx : forecast_start_idx + horizon].copy()

        # Chronos requires that future_df columns are a subset of df/context_df columns.
        context_cols = ["id", data_cfg["timestamp_col"], data_cfg["target_col"]] + known_future_covariates
        context_cols = [c for c in context_cols if c in context.columns]

        future_cols = ["id", data_cfg["timestamp_col"]] + known_future_covariates
        future_cols = [c for c in future_cols if c in future.columns]

        context_df = context[context_cols].copy()
        future_df = future[future_cols].copy()

        pred_df = pipeline.predict_df(
            context_df,  # first positional argument: historical dataframe
            future_df=future_df,
            prediction_length=horizon,
            quantile_levels=[0.5],
            id_column="id",
            timestamp_column=data_cfg["timestamp_col"],
            target=data_cfg["target_col"],
        )

        # Be robust to different output column names across Chronos versions.
        pred_col = None
        for c in ["predictions", "0.5", "prediction", "median"]:
            if c in pred_df.columns:
                pred_col = c
                break

        if pred_col is None:
            candidate_cols = [
                c for c in pred_df.columns
                if c not in {"id", data_cfg["timestamp_col"]}
            ]
            if not candidate_cols:
                raise ValueError("Chronos output does not contain a recognizable prediction column")
            pred_col = candidate_cols[0]

        y_pred = pred_df[pred_col].to_numpy()
        y_true = future[data_cfg["target_col"]].to_numpy()
        timestamps = future[data_cfg["timestamp_col"]].tolist()

        all_preds.append(y_pred)
        all_true.append(y_true)
        all_timestamps.extend(timestamps)

    if not all_preds:
        raise ValueError("No Chronos test windows were generated. Check split dates and data coverage.")

    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_true)

    out_df = pd.DataFrame({
        "timestamp": all_timestamps,
        "actual": y_true,
        "prediction": y_pred,
    })

    result_dir = cfg.output_root / "chronos"
    result_dir.mkdir(parents=True, exist_ok=True)

    metrics = metric_bundle(y_true, y_pred)
    save_csv(out_df, result_dir / "predictions.csv")
    save_json(metrics, result_dir / "metrics.json")

    return {"result_dir": str(result_dir)}