#!/usr/bin/env python3
"""Analisa os CSVs de execucao do LKAS.

O script nao executa novamente o SegFormer. Ele usa as saidas ja exportadas
pelo runtime para medir disponibilidade, erro lateral, estabilidade temporal,
geometria da pista e desempenho computacional.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Iterable, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REQUIRED_COLUMNS = {
    "frame",
    "target",
    "target_available",
    "vehicle_center_x",
    "target_x",
    "error_px",
    "error_threshold_px",
    "command",
    "raw_left_detected",
    "raw_right_detected",
    "left_detected",
    "right_detected",
    "left_x_base",
    "right_x_base",
    "left_confidence",
    "right_confidence",
    "left_curvature",
    "right_curvature",
    "processing_ms",
}


SUMMARY_FIELDS = [
    "dataset",
    "source_file",
    "frames",
    "target",
    "target_available_pct",
    "sem_target_frames",
    "max_sem_target_streak",
    "raw_left_detected_pct",
    "raw_right_detected_pct",
    "raw_both_detected_pct",
    "left_detected_pct",
    "right_detected_pct",
    "both_detected_pct",
    "error_threshold_px",
    "error_mean_px",
    "error_mae_px",
    "error_rmse_px",
    "error_std_px",
    "error_median_px",
    "error_abs_p95_px",
    "error_abs_max_px",
    "within_threshold_pct",
    "command_esquerda_pct",
    "command_reto_pct",
    "command_direita_pct",
    "command_sem_target_pct",
    "target_x_mean_px",
    "target_x_std_px",
    "jitter_pairs",
    "jitter_mean_abs_px",
    "jitter_std_px",
    "jitter_abs_p95_px",
    "jitter_abs_max_px",
    "jitter_gt_5_px_pct",
    "jitter_gt_10_px_pct",
    "jitter_gt_20_px_pct",
    "lane_width_mean_px",
    "lane_width_std_px",
    "lane_width_p05_px",
    "lane_width_p95_px",
    "lane_width_min_px",
    "lane_width_max_px",
    "left_confidence_mean",
    "right_confidence_mean",
    "left_curvature_median",
    "right_curvature_median",
    "processing_mean_ms",
    "processing_median_ms",
    "processing_p95_ms",
    "processing_max_ms",
    "processing_mean_fps",
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


def _int(value: object) -> Optional[int]:
    number = _float(value)
    return None if number is None else int(round(number))


def _bool(value: object) -> bool:
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "sim", "ok"}


def _percent(count: int, total: int) -> float:
    return 0.0 if total <= 0 else 100.0 * float(count) / float(total)


def _round(value: Optional[float], digits: int = 6):
    if value is None or not math.isfinite(float(value)):
        return None
    return round(float(value), digits)


def _array(values: Iterable[Optional[float]]) -> np.ndarray:
    return np.asarray([value for value in values if value is not None], dtype=np.float64)


def _percentile(values: np.ndarray, percentile: float) -> Optional[float]:
    if values.size == 0:
        return None
    return float(np.percentile(values, percentile))


def _mean(values: np.ndarray) -> Optional[float]:
    return None if values.size == 0 else float(np.mean(values))


def _median(values: np.ndarray) -> Optional[float]:
    return None if values.size == 0 else float(np.median(values))


def _std(values: np.ndarray) -> Optional[float]:
    return None if values.size == 0 else float(np.std(values))


def _max(values: np.ndarray) -> Optional[float]:
    return None if values.size == 0 else float(np.max(values))


def _min(values: np.ndarray) -> Optional[float]:
    return None if values.size == 0 else float(np.min(values))


def _safe_name(path: Path) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", path.stem)


def load_dataset(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"CSV nao encontrado: {path}")

    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])
        missing = sorted(REQUIRED_COLUMNS - fieldnames)
        if missing:
            raise ValueError(
                f"{path} nao possui as colunas necessarias: {', '.join(missing)}"
            )
        rows = list(reader)

    if not rows:
        raise ValueError(f"CSV vazio: {path}")
    return rows


def max_false_streak(values: Iterable[bool]) -> int:
    longest = 0
    current = 0
    for value in values:
        if value:
            current = 0
        else:
            current += 1
            longest = max(longest, current)
    return longest


def consecutive_differences(
    frames: list[Optional[int]], values: list[Optional[float]]
) -> np.ndarray:
    differences: list[float] = []
    previous_frame: Optional[int] = None
    previous_value: Optional[float] = None

    for frame, value in zip(frames, values):
        if (
            frame is not None
            and value is not None
            and previous_frame is not None
            and previous_value is not None
            and frame == previous_frame + 1
        ):
            differences.append(value - previous_value)

        if frame is None or value is None:
            previous_frame = None
            previous_value = None
        else:
            previous_frame = frame
            previous_value = value

    return np.asarray(differences, dtype=np.float64)


def analyze_dataset(path: Path) -> tuple[dict, dict[str, np.ndarray]]:
    rows = load_dataset(path)
    total = len(rows)

    frames = [_int(row.get("frame")) for row in rows]
    timestamps = [_float(row.get("timestamp_s")) for row in rows]
    target_values = [row.get("target", "") for row in rows]
    target_available = [_bool(row.get("target_available")) for row in rows]

    raw_left = [_bool(row.get("raw_left_detected")) for row in rows]
    raw_right = [_bool(row.get("raw_right_detected")) for row in rows]
    left = [_bool(row.get("left_detected")) for row in rows]
    right = [_bool(row.get("right_detected")) for row in rows]

    vehicle_center = [_float(row.get("vehicle_center_x")) for row in rows]
    target_x = [_float(row.get("target_x")) for row in rows]
    errors = [_float(row.get("error_px")) for row in rows]
    thresholds = _array(_float(row.get("error_threshold_px")) for row in rows)

    left_x = [_float(row.get("left_x_base")) for row in rows]
    right_x = [_float(row.get("right_x_base")) for row in rows]
    processing = _array(_float(row.get("processing_ms")) for row in rows)
    left_confidence = _array(_float(row.get("left_confidence")) for row in rows)
    right_confidence = _array(_float(row.get("right_confidence")) for row in rows)
    left_curvature = _array(_float(row.get("left_curvature")) for row in rows)
    right_curvature = _array(_float(row.get("right_curvature")) for row in rows)

    valid_errors = _array(
        error if available else None
        for error, available in zip(errors, target_available)
    )
    valid_target_x = _array(
        value if available else None
        for value, available in zip(target_x, target_available)
    )

    lane_width_values = _array(
        (rx - lx) if lx is not None and rx is not None else None
        for lx, rx in zip(left_x, right_x)
    )
    jitter = consecutive_differences(frames, target_x)
    jitter_abs = np.abs(jitter)

    threshold = float(np.median(thresholds)) if thresholds.size else 20.0
    command_counts = Counter(row.get("command", "") for row in rows)
    target_counter = Counter(value for value in target_values if value)
    target_name = target_counter.most_common(1)[0][0] if target_counter else ""

    error_abs = np.abs(valid_errors)
    error_rmse = (
        float(np.sqrt(np.mean(np.square(valid_errors))))
        if valid_errors.size
        else None
    )
    within_threshold = (
        _percent(int(np.sum(error_abs <= threshold)), int(error_abs.size))
        if error_abs.size
        else 0.0
    )

    processing_fps = 1000.0 / processing[processing > 0] if processing.size else np.array([])

    summary = {
        "dataset": _safe_name(path),
        "source_file": str(path),
        "frames": total,
        "target": target_name,
        "target_available_pct": _percent(sum(target_available), total),
        "sem_target_frames": total - sum(target_available),
        "max_sem_target_streak": max_false_streak(target_available),
        "raw_left_detected_pct": _percent(sum(raw_left), total),
        "raw_right_detected_pct": _percent(sum(raw_right), total),
        "raw_both_detected_pct": _percent(
            sum(l and r for l, r in zip(raw_left, raw_right)), total
        ),
        "left_detected_pct": _percent(sum(left), total),
        "right_detected_pct": _percent(sum(right), total),
        "both_detected_pct": _percent(
            sum(l and r for l, r in zip(left, right)), total
        ),
        "error_threshold_px": threshold,
        "error_mean_px": _mean(valid_errors),
        "error_mae_px": _mean(error_abs),
        "error_rmse_px": error_rmse,
        "error_std_px": _std(valid_errors),
        "error_median_px": _median(valid_errors),
        "error_abs_p95_px": _percentile(error_abs, 95),
        "error_abs_max_px": _max(error_abs),
        "within_threshold_pct": within_threshold,
        "command_esquerda_pct": _percent(command_counts["ESQUERDA"], total),
        "command_reto_pct": _percent(command_counts["RETO"], total),
        "command_direita_pct": _percent(command_counts["DIREITA"], total),
        "command_sem_target_pct": _percent(command_counts["SEM TARGET"], total),
        "target_x_mean_px": _mean(valid_target_x),
        "target_x_std_px": _std(valid_target_x),
        "jitter_pairs": int(jitter.size),
        "jitter_mean_abs_px": _mean(jitter_abs),
        "jitter_std_px": _std(jitter),
        "jitter_abs_p95_px": _percentile(jitter_abs, 95),
        "jitter_abs_max_px": _max(jitter_abs),
        "jitter_gt_5_px_pct": _percent(int(np.sum(jitter_abs > 5.0)), int(jitter_abs.size)),
        "jitter_gt_10_px_pct": _percent(int(np.sum(jitter_abs > 10.0)), int(jitter_abs.size)),
        "jitter_gt_20_px_pct": _percent(int(np.sum(jitter_abs > 20.0)), int(jitter_abs.size)),
        "lane_width_mean_px": _mean(lane_width_values),
        "lane_width_std_px": _std(lane_width_values),
        "lane_width_p05_px": _percentile(lane_width_values, 5),
        "lane_width_p95_px": _percentile(lane_width_values, 95),
        "lane_width_min_px": _min(lane_width_values),
        "lane_width_max_px": _max(lane_width_values),
        "left_confidence_mean": _mean(left_confidence),
        "right_confidence_mean": _mean(right_confidence),
        "left_curvature_median": _median(left_curvature),
        "right_curvature_median": _median(right_curvature),
        "processing_mean_ms": _mean(processing),
        "processing_median_ms": _median(processing),
        "processing_p95_ms": _percentile(processing, 95),
        "processing_max_ms": _max(processing),
        "processing_mean_fps": _mean(processing_fps),
    }

    summary = {
        key: _round(value) if isinstance(value, (float, np.floating)) else value
        for key, value in summary.items()
    }

    x_axis = np.asarray(
        [
            timestamp if timestamp is not None else float(frame or index)
            for index, (timestamp, frame) in enumerate(zip(timestamps, frames))
        ],
        dtype=np.float64,
    )
    series = {
        "x": x_axis,
        "error": np.asarray(
            [np.nan if value is None else value for value in errors], dtype=np.float64
        ),
        "target_x": np.asarray(
            [np.nan if value is None else value for value in target_x], dtype=np.float64
        ),
        "vehicle_center": np.asarray(
            [np.nan if value is None else value for value in vehicle_center], dtype=np.float64
        ),
        "lane_width": np.asarray(
            [
                np.nan if lx is None or rx is None else rx - lx
                for lx, rx in zip(left_x, right_x)
            ],
            dtype=np.float64,
        ),
        "processing_ms": np.asarray(
            [
                np.nan
                if _float(row.get("processing_ms")) is None
                else _float(row.get("processing_ms"))
                for row in rows
            ],
            dtype=np.float64,
        ),
        "jitter": np.concatenate(([np.nan], np.diff(np.asarray(
            [np.nan if value is None else value for value in target_x], dtype=np.float64
        )))),
        "threshold": np.asarray([threshold] * total, dtype=np.float64),
    }
    return summary, series


def save_summary_csv(summaries: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for summary in summaries:
            writer.writerow({field: summary.get(field) for field in SUMMARY_FIELDS})


def save_summary_json(summaries: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(summaries, handle, ensure_ascii=False, indent=2)


def _save_line_plot(
    x: np.ndarray,
    series: list[tuple[np.ndarray, str]],
    title: str,
    ylabel: str,
    output: Path,
    horizontal: Optional[list[tuple[float, str]]] = None,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    for values, label in series:
        ax.plot(x, values, linewidth=1.0, label=label)
    for value, label in horizontal or []:
        ax.axhline(value, linestyle="--", linewidth=1.0, label=label)
    ax.set_title(title)
    ax.set_xlabel("Tempo do video (s)")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    if len(series) + len(horizontal or []) > 1:
        ax.legend()
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)


def save_dataset_plots(name: str, series: dict[str, np.ndarray], output_dir: Path) -> None:
    plot_dir = output_dir / "plots"
    threshold = float(series["threshold"][0]) if series["threshold"].size else 20.0

    _save_line_plot(
        series["x"],
        [(series["error"], "Erro lateral")],
        f"Erro lateral - {name}",
        "Erro (px)",
        plot_dir / f"{name}_error.png",
        horizontal=[(threshold, "+limite"), (-threshold, "-limite"), (0.0, "zero")],
    )
    _save_line_plot(
        series["x"],
        [
            (series["target_x"], "Setpoint"),
            (series["vehicle_center"], "Centro do veiculo"),
        ],
        f"Setpoint e centro do veiculo - {name}",
        "Posicao X no BEV (px)",
        plot_dir / f"{name}_target.png",
    )
    _save_line_plot(
        series["x"],
        [(series["jitter"], "Delta do setpoint")],
        f"Jitter do setpoint - {name}",
        "Delta X entre frames (px)",
        plot_dir / f"{name}_jitter.png",
        horizontal=[(0.0, "zero")],
    )
    _save_line_plot(
        series["x"],
        [(series["lane_width"], "Largura estimada")],
        f"Largura da pista na base do BEV - {name}",
        "Largura (px)",
        plot_dir / f"{name}_lane_width.png",
    )
    _save_line_plot(
        series["x"],
        [(series["processing_ms"], "Processamento")],
        f"Tempo de processamento - {name}",
        "Tempo (ms/frame)",
        plot_dir / f"{name}_processing.png",
    )


def save_comparison_plot(summaries: list[dict], output_dir: Path) -> None:
    if len(summaries) < 2:
        return

    labels = [summary["dataset"] for summary in summaries]
    metrics = [
        ("target_available_pct", "Target disponivel (%)"),
        ("within_threshold_pct", "Dentro da zona morta (%)"),
        ("error_mae_px", "MAE do erro (px)"),
        ("jitter_mean_abs_px", "Jitter medio absoluto (px)"),
        ("processing_mean_ms", "Processamento medio (ms)"),
    ]

    for key, ylabel in metrics:
        values = [float(summary.get(key) or 0.0) for summary in summaries]
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(labels, values)
        ax.set_title(ylabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        output = output_dir / "plots" / f"comparison_{key}.png"
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=160)
        plt.close(fig)


def print_terminal_summary(summaries: list[dict]) -> None:
    print("\nResumo da analise LKAS")
    print(
        f"{'Dataset':<25} {'Frames':>7} {'Target%':>9} "
        f"{'MAE(px)':>9} {'P95(px)':>9} {'Jitter':>9} {'ms/frame':>10}"
    )
    for summary in summaries:
        print(
            f"{summary['dataset']:<25} "
            f"{summary['frames']:>7} "
            f"{float(summary['target_available_pct'] or 0):>8.2f}% "
            f"{float(summary['error_mae_px'] or 0):>9.2f} "
            f"{float(summary['error_abs_p95_px'] or 0):>9.2f} "
            f"{float(summary['jitter_mean_abs_px'] or 0):>9.3f} "
            f"{float(summary['processing_mean_ms'] or 0):>10.2f}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analisa um ou mais datasets CSV exportados pelo runtime LKAS"
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="Arquivos CSV, por exemplo experiments/lkas_video_01.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="experiments/lkas_analysis",
        help="Diretorio dos resumos e graficos",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Gera somente os resumos CSV e JSON",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)

    summaries: list[dict] = []
    all_series: list[tuple[str, dict[str, np.ndarray]]] = []

    for input_name in args.inputs:
        path = Path(input_name)
        summary, series = analyze_dataset(path)
        summaries.append(summary)
        all_series.append((summary["dataset"], series))

    save_summary_csv(summaries, output_dir / "lkas_analysis_summary.csv")
    save_summary_json(summaries, output_dir / "lkas_analysis_summary.json")

    if not args.no_plots:
        for name, series in all_series:
            save_dataset_plots(name, series, output_dir)
        save_comparison_plot(summaries, output_dir)

    print_terminal_summary(summaries)
    print(f"\nResumo CSV:  {output_dir / 'lkas_analysis_summary.csv'}")
    print(f"Resumo JSON: {output_dir / 'lkas_analysis_summary.json'}")
    if not args.no_plots:
        print(f"Graficos:    {output_dir / 'plots'}")


if __name__ == "__main__":
    main()
