from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


FORECAST_HORIZON = 8
WEEKLY_FREQ = "W-SUN"
VALIDATION_WEEKS = 8
PRIMARY_METRIC = "smape"


@dataclass(frozen=True)
class Paths:
    artifact_dir: Path

    @property
    def models_dir(self) -> Path:
        return self.artifact_dir / "models"

    @property
    def registry_path(self) -> Path:
        return self.artifact_dir / "model_registry.json"

    @property
    def forecasts_path(self) -> Path:
        return self.artifact_dir / "forecasts.csv"

    @property
    def metrics_path(self) -> Path:
        return self.artifact_dir / "metrics.csv"

    @property
    def metadata_path(self) -> Path:
        return self.artifact_dir / "metadata.json"

    def ensure(self) -> None:
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)

