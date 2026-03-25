from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd


@dataclass
class SeasonalNaiveBaseline:
    target_col: str
    seasonal_lag_steps: List[int]

    def fit(self, df: pd.DataFrame) -> None:
        self.train_df = df.copy()

    def predict_window(self, context: pd.DataFrame, horizon: int) -> np.ndarray:
        values = context[self.target_col].to_numpy()
        preds = []
        for h in range(horizon):
            candidates = []
            for lag in self.seasonal_lag_steps:
                idx = len(values) - lag + h
                if 0 <= idx < len(values):
                    candidates.append(values[idx])
            if not candidates:
                candidates.append(values[-1])
            preds.append(float(np.mean(candidates)))
        return np.asarray(preds, dtype=np.float32)
