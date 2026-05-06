from __future__ import annotations

from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .config import FORECAST_HORIZON
from .training import load_artifacts


class PredictRequest(BaseModel):
    state: Optional[str] = Field(default=None, description="State name. Null returns all states.")
    horizon: int = Field(default=FORECAST_HORIZON, ge=1, le=FORECAST_HORIZON)
    model: str = Field(default="best", pattern="^best$")


def create_app(artifact_dir: str | Path = "artifacts") -> FastAPI:
    app = FastAPI(title="MicroGCC Sales Forecasting API", version="0.1.0")
    registry, forecasts, metrics = load_artifacts(artifact_dir)

    @app.get("/health")
    def health() -> dict:
        return {
            "status": "ok",
            "created_at": registry["created_at"],
            "last_observed_date": registry["last_observed_date"],
            "forecast_horizon_weeks": registry["forecast_horizon_weeks"],
        }

    @app.get("/states")
    def states() -> list[dict]:
        selected = registry["selected_models"]
        return [{"state": state, "selected_model": selected[state], "last_observed_date": registry["last_observed_date"]} for state in sorted(selected)]

    @app.get("/metrics")
    def leaderboard() -> list[dict]:
        return metrics.sort_values(["state", "smape"]).to_dict("records")

    @app.post("/predict")
    def predict(payload: PredictRequest) -> dict:
        out = forecasts.copy()
        if payload.state is not None:
            if payload.state not in set(out["state"]):
                raise HTTPException(status_code=404, detail=f"Unknown state: {payload.state}")
            out = out[out["state"] == payload.state]
        out = out.sort_values(["state", "ds"]).groupby("state").head(payload.horizon)
        records = [
            {"state": row.state, "date": row.ds.date().isoformat(), "forecast": float(row.yhat), "model": row.model}
            for row in out.itertuples(index=False)
        ]
        return {"horizon": payload.horizon, "count": len(records), "predictions": records}

    return app

