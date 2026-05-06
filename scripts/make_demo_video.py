from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import textwrap
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont


WIDTH = 1920
HEIGHT = 1080
BG = "#f7f1e8"
INK = "#1d1d1f"
MUTED = "#5f6368"
ACCENT = "#c2410c"
BLUE = "#1f6f8b"


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def draw_wrapped(draw: ImageDraw.ImageDraw, text: str, xy: tuple[int, int], font_obj, fill: str, width: int, line_gap: int = 10) -> int:
    x, y = xy
    avg_char = max(font_obj.getlength("abcdefghijklmnopqrstuvwxyz") / 26, 1)
    chars = max(int(width / avg_char), 20)
    for paragraph in text.split("\n"):
        if not paragraph.strip():
            y += font_obj.size
            continue
        for line in textwrap.wrap(paragraph, width=chars):
            draw.text((x, y), line, font=font_obj, fill=fill)
            y += font_obj.size + line_gap
    return y


def cover(title: str, subtitle: str) -> Image.Image:
    img = Image.new("RGB", (WIDTH, HEIGHT), BG)
    draw = ImageDraw.Draw(img)
    draw.rectangle((0, 0, WIDTH, 28), fill=ACCENT)
    draw.text((110, 220), title, font=font(82, True), fill=INK)
    draw_wrapped(draw, subtitle, (115, 345), font(38), MUTED, 1280, 16)
    draw.text((115, 850), "MicroGCC Forecasting System", font=font(34, True), fill=BLUE)
    draw.text((115, 900), "43 state series · 11 models · 344 production forecasts", font=font(30), fill=MUTED)
    return img


def text_slide(title: str, bullets: list[str], footer: str | None = None) -> Image.Image:
    img = Image.new("RGB", (WIDTH, HEIGHT), BG)
    draw = ImageDraw.Draw(img)
    draw.text((95, 85), title, font=font(62, True), fill=INK)
    y = 230
    for bullet in bullets:
        draw.ellipse((105, y + 14, 123, y + 32), fill=ACCENT)
        y = draw_wrapped(draw, bullet, (150, y), font(38), INK, 1500, 14) + 26
    if footer:
        draw.text((95, 980), footer, font=font(28), fill=MUTED)
    return img


def figure_slide(title: str, caption: str, figure_path: Path) -> Image.Image:
    img = Image.new("RGB", (WIDTH, HEIGHT), BG)
    draw = ImageDraw.Draw(img)
    draw.text((80, 55), title, font=font(54, True), fill=INK)
    draw_wrapped(draw, caption, (82, 130), font(28), MUTED, 1700, 8)

    fig = Image.open(figure_path).convert("RGB")
    max_w, max_h = 1660, 780
    fig.thumbnail((max_w, max_h), Image.Resampling.LANCZOS)
    x = (WIDTH - fig.width) // 2
    y = 260 + (max_h - fig.height) // 2
    draw.rounded_rectangle((x - 16, y - 16, x + fig.width + 16, y + fig.height + 16), radius=24, fill="#ffffff")
    img.paste(fig, (x, y))
    return img


def metrics_slide(metrics: pd.DataFrame, registry: dict) -> Image.Image:
    img = Image.new("RGB", (WIDTH, HEIGHT), BG)
    draw = ImageDraw.Draw(img)
    draw.text((80, 55), "What won?", font=font(60, True), fill=INK)
    draw.text((82, 135), "Average validation sMAPE by model. Lower is better.", font=font(30), fill=MUTED)

    scores = metrics.groupby("model")["smape"].mean().sort_values().head(8)
    max_score = float(scores.max())
    y = 245
    for model, score in scores.items():
        draw.text((110, y), model, font=font(31, True), fill=INK)
        bar_w = int((score / max_score) * 830)
        draw.rounded_rectangle((590, y + 3, 590 + bar_w, y + 36), radius=12, fill=BLUE if model != "ets_holt_winters" else ACCENT)
        draw.text((1460, y), f"{score:.2f}", font=font(31), fill=INK)
        y += 74

    selected = pd.Series(registry["selected_models"]).value_counts()
    draw.text((110, 865), "Selected model distribution:", font=font(31, True), fill=INK)
    draw.text((570, 865), " · ".join(f"{k}: {v}" for k, v in selected.items()), font=font(29), fill=MUTED)
    return img


def api_slide() -> Image.Image:
    img = Image.new("RGB", (WIDTH, HEIGHT), "#111827")
    draw = ImageDraw.Draw(img)
    draw.text((90, 70), "Serving the forecasts", font=font(58, True), fill="#ffffff")
    draw.text((92, 145), "Training is offline. Prediction serving is artifact-first and fast.", font=font(31), fill="#d1d5db")
    code = """python -m microgcc serve --artifacts artifacts --host 0.0.0.0 --port 8000

curl -X POST http://localhost:8000/predict \\
  -H "Content-Type: application/json" \\
  -d '{"state":"California","horizon":8,"model":"best"}'

{
  "state": "California",
  "date": "2023-05-14",
  "forecast": 481234567.0,
  "model": "transformer_with_pe"
}"""
    draw.rounded_rectangle((90, 250, 1830, 890), radius=26, fill="#020617")
    draw_wrapped(draw, code, (135, 295), font(34), "#e5e7eb", 1650, 12)
    return img


def save_slides(slides: list[Image.Image], work_dir: Path) -> list[Path]:
    work_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for idx, slide in enumerate(slides, start=1):
        path = work_dir / f"slide_{idx:03d}.png"
        slide.save(path)
        paths.append(path)
    return paths


def render_video(slide_paths: list[Path], output: Path, seconds_per_slide: float) -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required to render the video")
    concat = output.parent / "slides.ffconcat"
    with concat.open("w", encoding="utf-8") as f:
        f.write("ffconcat version 1.0\n")
        for path in slide_paths:
            f.write(f"file '{path.resolve()}'\n")
            f.write(f"duration {seconds_per_slide}\n")
        f.write(f"file '{slide_paths[-1].resolve()}'\n")

    output.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(concat),
        "-vf",
        "fps=30,format=yuv420p",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(output),
    ]
    subprocess.run(cmd, check=True)


def build_video(artifact_dir: Path, figures_dir: Path, output: Path, seconds_per_slide: float) -> None:
    registry = json.loads((artifact_dir / "model_registry.json").read_text(encoding="utf-8"))
    metrics = pd.read_csv(artifact_dir / "metrics.csv")
    forecasts = pd.read_csv(artifact_dir / "forecasts.csv")

    slides = [
        cover(
            "Forecasting weekly sales, end to end",
            "A production-style walkthrough: clean the panel, run a model tournament, select winners per state, and serve 344 forecasts through an API.",
        ),
        text_slide(
            "The data tells us the first trick",
            [
                "The date strings are mixed, but day-first parsing gives a perfect weekly Sunday panel.",
                "That gives us 43 aligned state series from 2019-10-06 through 2023-05-07.",
                "The target is simple: forecast the next 8 Sundays for every state.",
            ],
            "A rectangular panel lets local time-series models and global ML models compete fairly.",
        ),
        figure_slide(
            "The aggregate series has structure",
            "Trend and seasonality are visible before we train anything. This is why classical seasonal models are serious competitors.",
            figures_dir / "historical_total_sales.png",
        ),
        text_slide(
            "The tournament",
            [
                "11 candidate models: ETS, SARIMA/ARIMA, Prophet, XGBoost, Random Forest, HistGradientBoosting, Ridge, LSTM, Transformer with PE, Transformer without PE, and seasonal naive.",
                "Validation is leakage-safe: train through 2023-03-12, validate on the next 8 weeks, then forecast 2023-05-14 through 2023-07-02.",
                "Selection is per state using sMAPE, not one global winner forced everywhere.",
            ],
        ),
        metrics_slide(metrics, registry),
        figure_slide(
            "The best model is not one model",
            "ETS dominates, but LSTM, SARIMA, Prophet, XGBoost, and even the Transformer win states. The system serves the winner behind one API contract.",
            figures_dir / "best_model_distribution.png",
        ),
        figure_slide(
            "State-level behavior matters",
            "The heatmap shows why the tournament is useful. Model quality changes by state; the winner is a local decision.",
            figures_dir / "state_model_error_heatmap.png",
        ),
        figure_slide(
            "Ablation: attention needs position",
            "The Transformer with sinusoidal positional encoding beats the no-position Transformer and wins California, even though simpler models win on average.",
            figures_dir / "transformer_pe_ablation.png",
        ),
        figure_slide(
            "Final forecasts",
            f"The forecast artifact contains {len(forecasts)} rows: 43 states times 8 weeks, from {registry['forecast_start']} to {registry['forecast_end']}.",
            figures_dir / "forecast_family_small_multiples.png",
        ),
        api_slide(),
        text_slide(
            "What ships",
            [
                "artifacts/model_registry.json records model attempts, winners, failures, and forecast windows.",
                "artifacts/metrics.csv stores the full validation leaderboard: 43 states times 11 models.",
                "artifacts/forecasts.csv stores the production response table: 344 future forecasts.",
                "FastAPI loads those artifacts and serves /health, /states, /metrics, and /predict.",
            ],
            "The lesson: forecasting quality comes from the comparison harness, not from worshipping one model family.",
        ),
    ]

    slide_paths = save_slides(slides, output.parent / "slides")
    render_video(slide_paths, output, seconds_per_slide)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate an MP4 demo video for the forecasting project.")
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument("--figures", type=Path, default=Path("reports/figures"))
    parser.add_argument("--output", type=Path, default=Path("reports/video/microgcc_demo.mp4"))
    parser.add_argument("--seconds-per-slide", type=float, default=5.0)
    args = parser.parse_args()
    build_video(args.artifacts, args.figures, args.output, args.seconds_per_slide)
    print(args.output)


if __name__ == "__main__":
    main()

