from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset


@dataclass
class SplitSpec:
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    val_start: pd.Timestamp
    val_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


@dataclass
class WindowSpec:
    lookback: int
    horizon: int
    stride: int


def build_split_spec(cfg_splits: dict) -> SplitSpec:
    return SplitSpec(
        train_start=pd.Timestamp(cfg_splits["train_start"]),
        train_end=pd.Timestamp(cfg_splits["train_end"]),
        val_start=pd.Timestamp(cfg_splits["val_start"]),
        val_end=pd.Timestamp(cfg_splits["val_end"]),
        test_start=pd.Timestamp(cfg_splits["test_start"]),
        test_end=pd.Timestamp(cfg_splits["test_end"]),
    )


def build_window_spec(cfg_window: dict) -> WindowSpec:
    steps_per_hour = int(cfg_window["steps_per_hour"])
    return WindowSpec(
        lookback=int(cfg_window["lookback_hours"]) * steps_per_hour,
        horizon=int(cfg_window["horizon_hours"]) * steps_per_hour,
        stride=int(cfg_window["stride_hours"]) * steps_per_hour,
    )


def _parse_timestamp(df: pd.DataFrame, timestamp_col: str) -> pd.DataFrame:
    if timestamp_col not in df.columns:
        raise KeyError(f"Expected timestamp column '{timestamp_col}'")

    out = df.copy()
    out[timestamp_col] = pd.to_datetime(out[timestamp_col], errors="coerce")
    bad = out[timestamp_col].isna().sum()
    if bad > 0:
        raise ValueError(f"Found {bad} unparsable timestamps in column '{timestamp_col}'")

    out = out.sort_values(timestamp_col).drop_duplicates(subset=[timestamp_col], keep="first")
    out = out.set_index(timestamp_col)
    return out


def _add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    idx = out.index

    if "hour" not in out.columns:
        out["hour"] = idx.hour
    if "minute" not in out.columns:
        out["minute"] = idx.minute
    if "day_of_week" not in out.columns:
        out["day_of_week"] = idx.dayofweek
    if "day_of_month" not in out.columns:
        out["day_of_month"] = idx.day
    if "month" not in out.columns:
        out["month"] = idx.month
    if "day_of_year" not in out.columns:
        out["day_of_year"] = idx.dayofyear
    if "year" not in out.columns:
        out["year"] = idx.year
    if "is_weekend" not in out.columns:
        out["is_weekend"] = (idx.dayofweek >= 5).astype(int)

    return out


def load_feature_frame(path: str, target_col: str, timestamp_col: str = "timestamp") -> pd.DataFrame:
    df = pd.read_csv(path)
    df = _parse_timestamp(df, timestamp_col)

    if target_col not in df.columns:
        raise KeyError(f"Expected target column '{target_col}' in {path}")

    df = _add_time_features(df)

    # Convert booleans to integers.
    bool_cols = df.select_dtypes(include=["bool"]).columns.tolist()
    for c in bool_cols:
        df[c] = df[c].astype(int)

    # Forward-fill features, keep target rows only where target is available.
    df = df[~df[target_col].isna()].copy()
    non_target_cols = [c for c in df.columns if c != target_col]
    if non_target_cols:
        df[non_target_cols] = df[non_target_cols].ffill().bfill()

    # Fill any remaining numeric NaNs with 0.
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if num_cols:
        df[num_cols] = df[num_cols].fillna(0.0)

    return df


def load_baseline_series(
    actual_file: str,
    baseline_file: str,
    actual_target_col: str,
    baseline_target_col: str,
    timestamp_col: str = "timestamp",
) -> pd.DataFrame:
    act = pd.read_csv(actual_file)
    act = _parse_timestamp(act, timestamp_col)

    fcst = pd.read_csv(baseline_file)
    fcst = _parse_timestamp(fcst, timestamp_col)

    if actual_target_col not in act.columns:
        raise KeyError(f"Expected actual target column '{actual_target_col}' in {actual_file}")
    if baseline_target_col not in fcst.columns:
        raise KeyError(f"Expected baseline target column '{baseline_target_col}' in {baseline_file}")

    out = act[[actual_target_col]].rename(columns={actual_target_col: "actual"}).join(
        fcst[[baseline_target_col]].rename(columns={baseline_target_col: "baseline"}),
        how="inner",
    )
    out = out.dropna(subset=["actual", "baseline"])
    return out


def prepare_supervised_data(
    feature_df: pd.DataFrame,
    split_spec: SplitSpec,
    target_col: str,
) -> Tuple[pd.DataFrame, list[str], int, list[str], StandardScaler]:
    feature_cols = feature_df.columns.tolist()
    if target_col not in feature_cols:
        raise KeyError(f"Target column '{target_col}' not found in features")

    nunique = feature_df.nunique(dropna=False)
    binary_cols = [c for c in feature_cols if nunique[c] <= 2]
    continuous_cols = [c for c in feature_cols if c not in binary_cols]

    train_mask = (feature_df.index >= split_spec.train_start) & (feature_df.index <= split_spec.train_end)
    if not train_mask.any():
        raise ValueError("Training split contains no rows")

    scaler = StandardScaler()
    scaler.fit(feature_df.loc[train_mask, continuous_cols].astype(float).values)

    scaled_df = feature_df.copy()
    scaled_df[continuous_cols] = scaler.transform(feature_df[continuous_cols].astype(float).values)

    target_idx = feature_cols.index(target_col)
    return scaled_df, feature_cols, target_idx, continuous_cols, scaler


class TimeSeriesWindowDataset(Dataset):
    def __init__(
        self,
        data: np.ndarray,
        timestamps: pd.DatetimeIndex,
        lookback: int,
        horizon: int,
        split_start: pd.Timestamp,
        split_end: pd.Timestamp,
        target_idx: int,
        baseline_scaled: Optional[np.ndarray],
        task: str,
        stride: int = 1,
    ) -> None:
        super().__init__()
        self.data = data
        self.timestamps = timestamps
        self.lookback = int(lookback)
        self.horizon = int(horizon)
        self.target_idx = int(target_idx)
        self.baseline_scaled = baseline_scaled
        self.task = task
        self.stride = int(stride)

        self.indices: list[int] = []

        max_start = len(timestamps) - lookback - horizon
        for i in range(0, max_start + 1, self.stride):
            forecast_start = timestamps[i + lookback]
            forecast_end = timestamps[i + lookback + horizon - 1]
            if forecast_start >= split_start and forecast_end <= split_end:
                self.indices.append(i)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        i = self.indices[idx]
        x = self.data[i : i + self.lookback, :]
        y_scaled = self.data[i + self.lookback : i + self.lookback + self.horizon, self.target_idx]

        if self.task == "residual":
            if self.baseline_scaled is None:
                raise ValueError("baseline_scaled must be provided for residual task")
            baseline_scaled = self.baseline_scaled[i + self.lookback : i + self.lookback + self.horizon]
            residual_scaled = y_scaled - baseline_scaled
            return (
                x.astype(np.float32),
                baseline_scaled.astype(np.float32),
                residual_scaled.astype(np.float32),
            )

        return x.astype(np.float32), y_scaled.astype(np.float32)

    def get_forecast_timestamps(self, idx: int) -> list[pd.Timestamp]:
        i = self.indices[idx]
        return list(self.timestamps[i + self.lookback : i + self.lookback + self.horizon])


def make_datasets(
    feature_df: pd.DataFrame,
    baseline_df: pd.DataFrame,
    split_spec: SplitSpec,
    window_spec: WindowSpec,
    target_col: str,
    task: str,
):
    scaled_df, feature_cols, target_idx, continuous_cols, scaler = prepare_supervised_data(
        feature_df, split_spec, target_col
    )

    target_cont_idx = continuous_cols.index(target_col)
    target_mean = float(scaler.mean_[target_cont_idx])
    target_std = float(np.sqrt(scaler.var_[target_cont_idx]))

    baseline_series_scaled = None
    baseline_series_raw = None

    if task == "residual":
        # Align feature timestamps with baseline timestamps.
        aligned = feature_df[[target_col]].join(baseline_df[["baseline"]], how="inner").dropna()
        if aligned.empty:
            raise ValueError("No overlapping timestamps between features and baseline for residual task")

        feature_df = feature_df.loc[aligned.index].copy()
        scaled_df = scaled_df.loc[aligned.index].copy()

        baseline_series_raw = aligned["baseline"].to_numpy(dtype=np.float32)
        baseline_series_scaled = ((baseline_series_raw - target_mean) / target_std).astype(np.float32)
    else:
        feature_df = feature_df.loc[scaled_df.index].copy()

    data = scaled_df[feature_cols].to_numpy(dtype=np.float32)
    timestamps = scaled_df.index

    train_ds = TimeSeriesWindowDataset(
        data=data,
        timestamps=timestamps,
        lookback=window_spec.lookback,
        horizon=window_spec.horizon,
        split_start=split_spec.train_start,
        split_end=split_spec.train_end,
        target_idx=target_idx,
        baseline_scaled=baseline_series_scaled,
        task=task,
        stride=window_spec.stride,
    )
    val_ds = TimeSeriesWindowDataset(
        data=data,
        timestamps=timestamps,
        lookback=window_spec.lookback,
        horizon=window_spec.horizon,
        split_start=split_spec.val_start,
        split_end=split_spec.val_end,
        target_idx=target_idx,
        baseline_scaled=baseline_series_scaled,
        task=task,
        stride=window_spec.stride,
    )
    test_ds = TimeSeriesWindowDataset(
        data=data,
        timestamps=timestamps,
        lookback=window_spec.lookback,
        horizon=window_spec.horizon,
        split_start=split_spec.test_start,
        split_end=split_spec.test_end,
        target_idx=target_idx,
        baseline_scaled=baseline_series_scaled,
        task=task,
        stride=window_spec.stride,
    )

    meta = {
        "feature_cols": feature_cols,
        "target_idx": target_idx,
        "continuous_cols": continuous_cols,
        "scaler": scaler,
        "scaler_mean": target_mean,
        "scaler_std": target_std,
        "timestamps": timestamps,
        "feature_df": feature_df,
        "baseline_series_scaled": baseline_series_scaled,
        "baseline_series_raw": baseline_series_raw,
    }
    return train_ds, val_ds, test_ds, meta