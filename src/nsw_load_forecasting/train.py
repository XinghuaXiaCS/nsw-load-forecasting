from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .config import Config
from .data import (
    build_split_spec,
    build_window_spec,
    load_baseline_series,
    load_feature_frame,
    make_datasets,
)
from .evaluation import predict_deterministic, predict_diffusion
from .models.baseline import SeasonalNaiveBaseline
from .models.diffusion import DiffusionForecaster
from .models.itransformer import ITransformer
from .models.patchtst import PatchTST
from .utils.io import save_csv, save_json, save_torch
from .utils.metrics import metric_bundle


class EarlyStopper:
    def __init__(self, patience: int) -> None:
        self.patience = patience
        self.best = float("inf")
        self.counter = 0
        self.best_state = None

    def step(self, loss: float, model: nn.Module) -> bool:
        if loss < self.best:
            self.best = loss
            self.counter = 0
            self.best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            return False
        self.counter += 1
        return self.counter >= self.patience


def build_model(model_name: str, cfg: Config, input_dim: int, lookback: int, horizon: int) -> nn.Module:
    mcfg = cfg.models[model_name]

    if model_name == "itransformer":
        return ITransformer(
            input_dim=input_dim,
            seq_len=lookback,
            pred_len=horizon,
            d_model=mcfg["d_model"],
            n_heads=mcfg["num_heads"],
            num_layers=mcfg["num_layers"],
            dropout=mcfg["dropout"],
        )

    if model_name == "patchtst":
        return PatchTST(
            input_dim=input_dim,
            seq_len=lookback,
            pred_len=horizon,
            d_model=mcfg["d_model"],
            num_heads=mcfg["num_heads"],
            num_layers=mcfg["num_layers"],
            patch_len=mcfg["patch_len"],
            stride=mcfg["stride"],
            dropout=mcfg["dropout"],
        )

    if model_name == "diffusion":
        return DiffusionForecaster(
            input_dim=input_dim,
            horizon=horizon,
            hidden_dim=mcfg["hidden_dim"],
            timesteps=mcfg["timesteps"],
            beta_start=mcfg["beta_start"],
            beta_end=mcfg["beta_end"],
        )

    raise ValueError(f"Unknown model: {model_name}")


def train_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, task: str, device: str) -> float:
    model.train()
    losses = []

    for batch in loader:
        optimizer.zero_grad()

        if isinstance(model, DiffusionForecaster):
            x, baseline_scaled, residual_scaled = batch
            x = x.to(device)
            residual_scaled = residual_scaled.to(device)
            loss = model.loss(x, residual_scaled)

        elif task == "residual":
            x, baseline_scaled, residual_scaled = batch
            x = x.to(device)
            residual_scaled = residual_scaled.to(device)
            pred_residual_scaled = model(x)
            loss = torch.nn.functional.mse_loss(pred_residual_scaled, residual_scaled)

        else:
            x, y_scaled = batch
            x = x.to(device)
            y_scaled = y_scaled.to(device)
            pred_scaled = model(x)
            loss = torch.nn.functional.mse_loss(pred_scaled, y_scaled)

        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    return float(np.mean(losses))


@torch.no_grad()
def validate_epoch(model: nn.Module, loader: DataLoader, task: str, device: str) -> float:
    model.eval()
    losses = []

    for batch in loader:
        if isinstance(model, DiffusionForecaster):
            x, baseline_scaled, residual_scaled = batch
            x = x.to(device)
            residual_scaled = residual_scaled.to(device)
            loss = model.loss(x, residual_scaled)

        elif task == "residual":
            x, baseline_scaled, residual_scaled = batch
            x = x.to(device)
            residual_scaled = residual_scaled.to(device)
            pred_residual_scaled = model(x)
            loss = torch.nn.functional.mse_loss(pred_residual_scaled, residual_scaled)

        else:
            x, y_scaled = batch
            x = x.to(device)
            y_scaled = y_scaled.to(device)
            pred_scaled = model(x)
            loss = torch.nn.functional.mse_loss(pred_scaled, y_scaled)

        losses.append(loss.item())

    return float(np.mean(losses))


def inverse_target_scale(arr: np.ndarray, mean: float, std: float) -> np.ndarray:
    return arr * std + mean


def train_model(cfg: Config, model_name: str, task: str) -> Dict[str, str]:
    data_cfg = cfg.data
    split_spec = build_split_spec(cfg.splits)
    window_spec = build_window_spec(cfg.window)

    feature_df = load_feature_frame(
        data_cfg["feature_file"],
        data_cfg["target_col"],
        timestamp_col=data_cfg["timestamp_col"],
    )
    baseline_df = load_baseline_series(
        data_cfg["actual_file"],
        data_cfg["baseline_file"],
        data_cfg["actual_target_col"],
        data_cfg["baseline_target_col"],
        timestamp_col=data_cfg["timestamp_col"],
    )

    train_ds, val_ds, test_ds, meta = make_datasets(
        feature_df=feature_df,
        baseline_df=baseline_df,
        split_spec=split_spec,
        window_spec=window_spec,
        target_col=data_cfg["target_col"],
        task=task,
    )

    result_dir = cfg.output_root / f"{model_name}_{task}"
    result_dir.mkdir(parents=True, exist_ok=True)

    # Baseline model is already evaluated in raw MW space.
    if model_name == "baseline":
        lags = [int(h * cfg.window["steps_per_hour"]) for h in cfg.models["baseline"]["seasonal_lag_hours"]]
        model = SeasonalNaiveBaseline(
            target_col=data_cfg["target_col"],
            seasonal_lag_steps=lags,
        )
        model.fit(feature_df.loc[: split_spec.train_end])

        preds = []
        trues = []
        timestamps = []

        for idx in range(len(test_ds)):
            start = test_ds.indices[idx]
            context = feature_df.iloc[start : start + window_spec.lookback]
            pred = model.predict_window(context, window_spec.horizon)
            true = feature_df.iloc[
                start + window_spec.lookback : start + window_spec.lookback + window_spec.horizon
            ][data_cfg["target_col"]].to_numpy()

            preds.append(pred)
            trues.append(true)
            timestamps.extend(test_ds.get_forecast_timestamps(idx))

        y_pred = np.concatenate(preds)
        y_true = np.concatenate(trues)

        pred_df = pd.DataFrame({
            "timestamp": timestamps,
            "actual": y_true,
            "prediction": y_pred,
        })

        metrics = metric_bundle(y_true, y_pred)
        save_csv(pred_df, result_dir / "predictions.csv")
        save_json(metrics, result_dir / "metrics.json")
        return {"result_dir": str(result_dir)}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(
        model_name=model_name,
        cfg=cfg,
        input_dim=len(meta["feature_cols"]),
        lookback=window_spec.lookback,
        horizon=window_spec.horizon,
    ).to(device)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.training["batch_size"],
        shuffle=True,
        num_workers=cfg.training["num_workers"],
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.training["batch_size"],
        shuffle=False,
        num_workers=cfg.training["num_workers"],
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.training["batch_size"],
        shuffle=False,
        num_workers=cfg.training["num_workers"],
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.training["lr"],
        weight_decay=cfg.training["weight_decay"],
    )
    stopper = EarlyStopper(patience=cfg.training["patience"])

    history = []
    for epoch in range(cfg.training["epochs"]):
        train_loss = train_epoch(model, train_loader, optimizer, task, device)
        val_loss = validate_epoch(model, val_loader, task, device)

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
        })

        if stopper.step(val_loss, model):
            break

    if stopper.best_state is not None:
        model.load_state_dict(stopper.best_state)

    if isinstance(model, DiffusionForecaster):
        outputs = predict_diffusion(
            model,
            test_loader,
            device=device,
            n_samples=cfg.models["diffusion"]["samples"],
        )
    else:
        outputs = predict_deterministic(
            model,
            test_loader,
            task=task,
            device=device,
        )

    timestamps = []
    for idx in range(len(test_ds)):
        timestamps.extend(test_ds.get_forecast_timestamps(idx))

    scaler_mean = meta["scaler_mean"]
    scaler_std = meta["scaler_std"]

    y_true_scaled = outputs["y_true"].reshape(-1)
    y_pred_scaled = outputs["y_pred"].reshape(-1)

    y_true_raw = inverse_target_scale(y_true_scaled, scaler_mean, scaler_std)
    y_pred_raw = inverse_target_scale(y_pred_scaled, scaler_mean, scaler_std)

    pred_df = pd.DataFrame({
        "timestamp": timestamps,
        "actual": y_true_raw,
        "prediction": y_pred_raw,
    })

    if task == "residual" and "baseline" in outputs:
        baseline_scaled = outputs["baseline"].reshape(-1)
        baseline_raw = inverse_target_scale(baseline_scaled, scaler_mean, scaler_std)
        pred_df["baseline"] = baseline_raw

    if "p10" in outputs:
        pred_df["p10"] = inverse_target_scale(outputs["p10"].reshape(-1), scaler_mean, scaler_std)
        pred_df["p50"] = inverse_target_scale(outputs["p50"].reshape(-1), scaler_mean, scaler_std)
        pred_df["p90"] = inverse_target_scale(outputs["p90"].reshape(-1), scaler_mean, scaler_std)

    metrics = metric_bundle(pred_df["actual"].to_numpy(), pred_df["prediction"].to_numpy())

    save_csv(pred_df, result_dir / "predictions.csv")
    save_csv(pd.DataFrame(history), result_dir / "history.csv")
    save_json(metrics, result_dir / "metrics.json")
    save_torch(model.state_dict(), result_dir / "model.pt")
    save_json(
        {"task": task, "model": model_name, "device": device},
        result_dir / "run_info.json",
    )

    return {"result_dir": str(result_dir)}