from __future__ import annotations

from pathlib import Path

import pandas as pd

from .data import load_sales
from .training import load_artifacts


def generate_figures(data_path: str | Path, artifact_dir: str | Path, output_dir: str | Path) -> list[Path]:
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("matplotlib is required to generate figures") from exc

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = load_sales(data_path)
    registry, forecasts, metrics = load_artifacts(artifact_dir)
    written: list[Path] = []

    total = df.groupby("ds", as_index=False)["y"].sum()
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(total["ds"], total["y"] / 1e9, color="#1f6f68", linewidth=2)
    ax.set_title("Historical Weekly Sales Across All States")
    ax.set_ylabel("Sales, billions")
    ax.set_xlabel("Week")
    ax.grid(alpha=0.25)
    path = out / "historical_total_sales.png"
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    written.append(path)

    example_state = "California" if "California" in set(df["state"]) else sorted(df["state"].unique())[0]
    hist = df[df["state"] == example_state]
    fut = forecasts[forecasts["state"] == example_state]
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(hist["ds"], hist["y"] / 1e6, label="History", color="#2b2d42")
    ax.plot(fut["ds"], fut["yhat"] / 1e6, label="8-week forecast", color="#d1495b", marker="o")
    ax.set_title(f"{example_state}: Historical Sales and Forecast")
    ax.set_ylabel("Sales, millions")
    ax.legend()
    ax.grid(alpha=0.25)
    path = out / "example_state_forecast.png"
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    written.append(path)

    best = metrics.loc[metrics.groupby("state")["smape"].idxmin()].sort_values("smape")
    states_to_show = []
    for model in ["transformer_with_pe", "lstm", "xgboost", "prophet", "sarima", "ets_holt_winters"]:
        candidates = best[best["model"] == model]["state"].tolist()
        if candidates:
            states_to_show.append(candidates[0])
    states_to_show = list(dict.fromkeys(states_to_show))[:6]

    fig, axes = plt.subplots(3, 2, figsize=(13, 10), sharex=False)
    axes = axes.ravel()
    for ax, state in zip(axes, states_to_show):
        hist = df[df["state"] == state].tail(80)
        fut = forecasts[forecasts["state"] == state]
        selected_model = registry["selected_models"][state]
        ax.plot(hist["ds"], hist["y"] / 1e6, color="#293241", linewidth=1.8)
        ax.plot(fut["ds"], fut["yhat"] / 1e6, color="#ee6c4d", marker="o", linewidth=2)
        ax.axvline(df["ds"].max(), color="#8d99ae", linestyle="--", linewidth=1)
        ax.set_title(f"{state}: {selected_model}", fontsize=10)
        ax.set_ylabel("Sales, M")
        ax.grid(alpha=0.2)
    for ax in axes[len(states_to_show):]:
        ax.axis("off")
    fig.suptitle("Different States Prefer Different Forecasting Biases", fontsize=15, y=0.995)
    path = out / "forecast_family_small_multiples.png"
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    written.append(path)

    model_scores = metrics.groupby("model", as_index=False)["smape"].mean().sort_values("smape")
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(model_scores["model"], model_scores["smape"], color="#3a86ff")
    ax.set_title("Average Validation sMAPE by Model")
    ax.set_ylabel("sMAPE, lower is better")
    ax.tick_params(axis="x", rotation=20)
    path = out / "model_leaderboard.png"
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    written.append(path)

    dist = best["model"].value_counts()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(dist.index, dist.values, color="#f4a261")
    ax.set_title("Best Model Distribution Across States")
    ax.set_ylabel("State count")
    ax.tick_params(axis="x", rotation=20)
    path = out / "best_model_distribution.png"
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    written.append(path)

    ordered_states = best.sort_values("smape")["state"].tolist()
    heatmap = (
        metrics.pivot(index="model", columns="state", values="smape")
        .reindex(model_scores["model"])
        .reindex(columns=ordered_states)
    )
    fig, ax = plt.subplots(figsize=(15, 6))
    image = ax.imshow(np.clip(heatmap.to_numpy(), 0, 20), aspect="auto", cmap="viridis_r")
    ax.set_title("Validation Error Landscape: Model sMAPE by State")
    ax.set_xlabel("States ordered from easiest to hardest")
    ax.set_ylabel("Model")
    ax.set_yticks(range(len(heatmap.index)))
    ax.set_yticklabels(heatmap.index)
    ax.set_xticks(range(len(ordered_states)))
    ax.set_xticklabels(ordered_states, rotation=75, ha="right", fontsize=7)
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("sMAPE, clipped at 20")
    path = out / "state_model_error_heatmap.png"
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    written.append(path)

    fig, ax = plt.subplots(figsize=(13, 7))
    colors = {
        "ets_holt_winters": "#2a9d8f",
        "lstm": "#264653",
        "sarima": "#e9c46a",
        "xgboost": "#f4a261",
        "prophet": "#e76f51",
        "transformer_with_pe": "#3a86ff",
        "transformer_without_pe": "#8338ec",
    }
    bar_colors = [colors.get(model, "#8d99ae") for model in best["model"]]
    ax.bar(best["state"], best["smape"], color=bar_colors)
    ax.set_title("Best Achievable Validation Error by State")
    ax.set_ylabel("Best model sMAPE")
    ax.tick_params(axis="x", rotation=75, labelsize=7)
    handles = [
        plt.Line2D([0], [0], marker="s", color="w", markerfacecolor=color, markersize=9, label=model)
        for model, color in colors.items()
        if model in set(best["model"])
    ]
    ax.legend(handles=handles, loc="upper left", ncols=3, fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    path = out / "best_state_error_bars.png"
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    written.append(path)

    transformer_scores = (
        metrics[metrics["model"].str.contains("transformer")]
        .groupby("model", as_index=False)["smape"]
        .mean()
        .sort_values("smape")
    )
    if not transformer_scores.empty:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.bar(transformer_scores["model"], transformer_scores["smape"], color=["#3a86ff", "#8338ec"][: len(transformer_scores)])
        ax.set_title("Transformer Ablation: Position Matters")
        ax.set_ylabel("Average validation sMAPE")
        ax.tick_params(axis="x", rotation=12)
        for idx, row in transformer_scores.reset_index(drop=True).iterrows():
            ax.text(idx, row["smape"] + 0.4, f"{row['smape']:.1f}", ha="center")
        path = out / "transformer_pe_ablation.png"
        fig.tight_layout()
        fig.savefig(path, dpi=180)
        plt.close(fig)
        written.append(path)

    return written
