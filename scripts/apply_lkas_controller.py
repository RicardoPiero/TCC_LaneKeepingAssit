#!/usr/bin/env python3
"""Aplica o controlador contínuo aos datasets CSV do LKAS.

Este script não executa novamente o SegFormer. Ele usa as posições de lane e o
erro lateral já exportados para simular e calibrar o comando de direção.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from segformer_runtime.controller import ProportionalSteeringController


REQUIRED_COLUMNS = {
    "frame",
    "target_available",
    "error_px",
    "left_x_base",
    "right_x_base",
}

ADDED_FIELDS = [
    "lane_width_px",
    "error_normalized",
    "steering_raw",
    "steering_filtered",
    "steering_delta",
    "steering_direction",
    "steering_held",
    "steering_saturated",
]

SUMMARY_FIELDS = [
    "dataset",
    "source_file",
    "output_file",
    "frames",
    "target_available_pct",
    "kp",
    "deadband_px",
    "smoothing_alpha",
    "max_step_per_frame",
    "max_output",
    "hold_missing_frames",
    "steering_mean",
    "steering_mean_abs",
    "steering_abs_p95",
    "steering_abs_max",
    "steering_delta_mean_abs",
    "steering_delta_abs_p95",
    "steering_delta_abs_max",
    "steering_left_pct",
    "steering_straight_pct",
    "steering_right_pct",
    "steering_held_pct",
    "steering_saturated_pct",
]


def _float(value: object) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        result = float(text)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _bool(value: object) -> bool:
    number = _float(value)
    return bool(number is not None and number != 0)


def _round(value: Optional[float], digits: int = 6):
    return "" if value is None else round(float(value), digits)


def _pct(count: int, total: int) -> float:
    return 100.0 * count / total if total else 0.0


def _percentile(values: list[float], q: float) -> float:
    return float(np.percentile(values, q)) if values else 0.0


def _direction(steering: float, threshold: float) -> str:
    if steering > threshold:
        return "ESQUERDA"
    if steering < -threshold:
        return "DIREITA"
    return "RETO"


def process_dataset(path: Path, output_dir: Path, args) -> tuple[dict, dict]:
    if not path.exists():
        raise FileNotFoundError(f"Dataset não encontrado: {path}")

    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        missing = sorted(REQUIRED_COLUMNS.difference(fieldnames))
        if missing:
            raise ValueError(
                f"{path}: colunas ausentes: {', '.join(missing)}"
            )
        rows = list(reader)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{path.stem}_control.csv"

    controller = ProportionalSteeringController(
        kp=args.kp,
        deadband_px=args.deadband_px,
        smoothing_alpha=args.smoothing_alpha,
        max_step_per_frame=args.max_step,
        max_output=args.max_output,
        hold_missing_frames=args.hold_missing_frames,
    )

    steering_values: list[float] = []
    steering_deltas: list[float] = []
    frames: list[float] = []
    errors: list[float] = []
    normalized_errors: list[float] = []
    lane_widths: list[float] = []
    raw_values: list[float] = []

    available_count = 0
    held_count = 0
    saturated_count = 0
    left_count = 0
    straight_count = 0
    right_count = 0
    previous_steering = 0.0

    output_fields = fieldnames + [
        field for field in ADDED_FIELDS if field not in fieldnames
    ]

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=output_fields)
        writer.writeheader()

        for index, row in enumerate(rows):
            left_x = _float(row.get("left_x_base"))
            right_x = _float(row.get("right_x_base"))
            error_px = _float(row.get("error_px"))
            target_available = _bool(row.get("target_available"))

            lane_width = None
            if left_x is not None and right_x is not None:
                candidate = right_x - left_x
                if math.isfinite(candidate) and candidate > 1.0:
                    lane_width = float(candidate)

            result = controller.update(
                error_px=error_px,
                lane_width_px=lane_width,
                target_available=target_available,
            )

            steering = result.steering_filtered
            delta = steering - previous_steering
            previous_steering = steering
            direction = _direction(steering, args.straight_threshold)

            if result.target_available:
                available_count += 1
            if result.held:
                held_count += 1
            if result.saturated:
                saturated_count += 1

            if direction == "ESQUERDA":
                left_count += 1
            elif direction == "DIREITA":
                right_count += 1
            else:
                straight_count += 1

            frame = _float(row.get("frame"))
            frames.append(float(index if frame is None else frame))
            steering_values.append(float(steering))
            steering_deltas.append(float(delta))
            errors.append(float("nan") if error_px is None else error_px)
            normalized_errors.append(
                float("nan")
                if result.error_normalized is None
                else result.error_normalized
            )
            lane_widths.append(
                float("nan") if result.lane_width_px is None else result.lane_width_px
            )
            raw_values.append(
                float("nan") if result.steering_raw is None else result.steering_raw
            )

            output_row = dict(row)
            output_row.update(
                {
                    "lane_width_px": _round(result.lane_width_px),
                    "error_normalized": _round(result.error_normalized),
                    "steering_raw": _round(result.steering_raw),
                    "steering_filtered": _round(result.steering_filtered),
                    "steering_delta": _round(delta),
                    "steering_direction": direction,
                    "steering_held": int(result.held),
                    "steering_saturated": int(result.saturated),
                }
            )
            writer.writerow(output_row)

    total = len(rows)
    steering_abs = np.abs(np.asarray(steering_values, dtype=np.float64))
    delta_abs = np.abs(np.asarray(steering_deltas, dtype=np.float64))

    summary = {
        "dataset": path.stem,
        "source_file": str(path),
        "output_file": str(output_path),
        "frames": total,
        "target_available_pct": round(_pct(available_count, total), 6),
        "kp": args.kp,
        "deadband_px": args.deadband_px,
        "smoothing_alpha": args.smoothing_alpha,
        "max_step_per_frame": args.max_step,
        "max_output": args.max_output,
        "hold_missing_frames": args.hold_missing_frames,
        "steering_mean": round(float(np.mean(steering_values)) if total else 0.0, 6),
        "steering_mean_abs": round(float(np.mean(steering_abs)) if total else 0.0, 6),
        "steering_abs_p95": round(_percentile(steering_abs.tolist(), 95), 6),
        "steering_abs_max": round(float(np.max(steering_abs)) if total else 0.0, 6),
        "steering_delta_mean_abs": round(float(np.mean(delta_abs)) if total else 0.0, 6),
        "steering_delta_abs_p95": round(_percentile(delta_abs.tolist(), 95), 6),
        "steering_delta_abs_max": round(float(np.max(delta_abs)) if total else 0.0, 6),
        "steering_left_pct": round(_pct(left_count, total), 6),
        "steering_straight_pct": round(_pct(straight_count, total), 6),
        "steering_right_pct": round(_pct(right_count, total), 6),
        "steering_held_pct": round(_pct(held_count, total), 6),
        "steering_saturated_pct": round(_pct(saturated_count, total), 6),
    }

    series = {
        "frame": np.asarray(frames, dtype=np.float64),
        "error_px": np.asarray(errors, dtype=np.float64),
        "error_normalized": np.asarray(normalized_errors, dtype=np.float64),
        "lane_width_px": np.asarray(lane_widths, dtype=np.float64),
        "steering_raw": np.asarray(raw_values, dtype=np.float64),
        "steering_filtered": np.asarray(steering_values, dtype=np.float64),
        "steering_delta": np.asarray(steering_deltas, dtype=np.float64),
    }
    return summary, series


def save_plots(name: str, series: dict, output_dir: Path) -> None:
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    frame = series["frame"]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(frame, series["error_normalized"], label="Erro normalizado")
    ax.axhline(0.0, linewidth=1)
    ax.set_title(f"Erro normalizado - {name}")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Erro normalizado")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_dir / f"{name}_normalized_error.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(frame, series["steering_raw"], label="Steering bruto")
    ax.plot(frame, series["steering_filtered"], label="Steering filtrado")
    ax.axhline(0.0, linewidth=1)
    ax.set_title(f"Comando contínuo de direção - {name}")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Steering [-1, +1]")
    ax.set_ylim(-1.05, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_dir / f"{name}_steering.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(frame, series["steering_delta"], label="Delta steering")
    ax.axhline(0.0, linewidth=1)
    ax.set_title(f"Variação do comando por frame - {name}")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Delta steering")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_dir / f"{name}_steering_delta.png", dpi=160)
    plt.close(fig)


def save_summaries(summaries: list[dict], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "lkas_controller_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(summaries)

    json_path = output_dir / "lkas_controller_summary.json"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(summaries, handle, ensure_ascii=False, indent=2)


def print_summary(summaries: list[dict]) -> None:
    print("\nResumo do controlador LKAS")
    print(
        f"{'Dataset':<24} {'Frames':>7} {'|Steer|':>9} "
        f"{'P95':>8} {'Delta':>8} {'Sat%':>8}"
    )
    for item in summaries:
        print(
            f"{item['dataset']:<24} {item['frames']:>7} "
            f"{item['steering_mean_abs']:>9.3f} "
            f"{item['steering_abs_p95']:>8.3f} "
            f"{item['steering_delta_mean_abs']:>8.4f} "
            f"{item['steering_saturated_pct']:>7.2f}%"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Aplica controlador proporcional aos CSVs do LKAS"
    )
    parser.add_argument("inputs", nargs="+", help="Datasets CSV do LKAS")
    parser.add_argument(
        "--output-dir",
        default="experiments/lkas_control",
        help="Diretório dos CSVs de controle, resumo e gráficos",
    )
    parser.add_argument("--kp", type=float, default=0.8)
    parser.add_argument("--deadband-px", type=float, default=3.0)
    parser.add_argument("--smoothing-alpha", type=float, default=0.25)
    parser.add_argument("--max-step", type=float, default=0.08)
    parser.add_argument("--max-output", type=float, default=1.0)
    parser.add_argument("--hold-missing-frames", type=int, default=10)
    parser.add_argument(
        "--straight-threshold",
        type=float,
        default=0.05,
        help="Faixa de steering considerada RETO somente para o resumo",
    )
    parser.add_argument("--no-plots", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)

    summaries: list[dict] = []
    series_collection: list[tuple[str, dict]] = []

    for input_name in args.inputs:
        path = Path(input_name)
        summary, series = process_dataset(path, output_dir, args)
        summaries.append(summary)
        series_collection.append((summary["dataset"], series))

    save_summaries(summaries, output_dir)

    if not args.no_plots:
        for name, series in series_collection:
            save_plots(name, series, output_dir)

    print_summary(summaries)
    print(f"\nSaída: {output_dir}")
    print(f"Resumo: {output_dir / 'lkas_controller_summary.csv'}")


if __name__ == "__main__":
    main()
