from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import yaml


@dataclass
class Config:
    raw: Dict[str, Any]

    @property
    def data(self) -> Dict[str, Any]:
        return self.raw["data"]

    @property
    def splits(self) -> Dict[str, Any]:
        return self.raw["splits"]

    @property
    def window(self) -> Dict[str, Any]:
        return self.raw["window"]

    @property
    def training(self) -> Dict[str, Any]:
        return self.raw["training"]

    @property
    def models(self) -> Dict[str, Any]:
        return self.raw["models"]

    @property
    def output(self) -> Dict[str, Any]:
        return self.raw["output"]

    @property
    def seed(self) -> int:
        return int(self.raw.get("seed", 42))

    @property
    def output_root(self) -> Path:
        path = Path(self.output["root_dir"])
        path.mkdir(parents=True, exist_ok=True)
        return path


def load_config(path: str | Path) -> Config:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return Config(raw=raw)
