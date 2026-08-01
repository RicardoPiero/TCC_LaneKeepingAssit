from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from . import pipeline as base
from .controller import ProportionalSteeringController
from .geometry import LaneTemporalTracker, detect_lane_pixels
from .model import infer_mask, load_checkpoint
from .overlay import draw_overlay
from .setpoint import build_setpoint_coefficients, calculate_target_guidance


WINDOW_NAME = "LKAS - SegFormer-B0 + Controle Continuo"

CONTROL_FIELDS = [
    "lane_width_px",
    "error_normalized",
    "steering_raw",
    "steering_filtered",
    "steering_delta",
    "steering_direction",
    "steering_held",
    "steering_saturated",
    "controller_kp",
    "controller_deadband_px",
    "controller_smoothing_alpha",
    "controller_max_step",
    "controller_max_output",
    "controller_hold_missing_frames",
]
DATASET_FIELDS = list(base.DATASET_FIELDS) + CONTROL_FIELDS


def _round_or_none(value, digits=6):
    if value is None:
        return None
    return round(float(value), digits)


def _open_dataset_writer(path):
    if not path:
        return None, None
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    handle = output.open("w", newline="", encoding="utf-8")
    writer = csv.DictWriter(handle, fieldnames=DATASET_FIELDS)
    writer.writeheader()
    return handle, writer


def _lane_width(left, right):
    if left is None or right is None:
        return None
    width = float(right.x_base - left.x_base)
    if not np.isfinite(width) or width <= 1.0:
        return None
    return width


def _steering_direction(value, threshold):
    if value > threshold:
        return "ESQUERDA"
    if value < -threshold:
        return "DIREITA"
    return "RETO"


def run(args):
    video_path, config_path = base.resolve_video(
        args.video, args.video_path, args.config
    )

    if not Path(video_path).exists():
        raise FileNotFoundError(f"Video nao encontrado: {video_path}")
    if not Path(config_path).exists():
        raise FileNotFoundError(f"Configuracao nao encontrada: {config_path}")
    if not Path(args.checkpoint).exists():
        raise FileNotFoundError(f"Checkpoint nao encontrado: {args.checkpoint}")

    cfg = base.apply_roi_overrides(base.load_runtime_config(config_path), args)
    pseudo_left_offset, pseudo_right_offset = base.resolve_pseudo_offsets(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, image_mean, image_std = load_checkpoint(args.checkpoint, device)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Nao foi possivel abrir: {video_path}")

    source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    base.validate_roi(cfg, frame_width, frame_height)
    eval_size = (args.eval_width, args.eval_height)

    controller = ProportionalSteeringController(
        kp=args.kp,
        deadband_px=args.deadband_px,
        smoothing_alpha=args.smoothing_alpha,
        max_step_per_frame=args.max_step,
        max_output=args.max_output,
        hold_missing_frames=args.steering_hold_frames,
    )

    tracker = None
    video_writer = None
    dataset_file = None
    dataset_writer = None
    live_fps = None
    paused = False
    processed_frames = 0
    previous_steering = 0.0
    mosaic = np.zeros((720, 1280, 3), dtype=np.uint8)

    if args.output_video:
        output = Path(args.output_video)
        output.parent.mkdir(parents=True, exist_ok=True)
        video_writer = cv2.VideoWriter(
            str(output),
            cv2.VideoWriter_fourcc(*"mp4v"),
            source_fps if source_fps > 0 else 20.0,
            (1280, 720),
        )

    dataset_file, dataset_writer = _open_dataset_writer(args.export_dataset)

    print(f"Video: {video_path}")
    print(f"Config base: {config_path}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Device: {device}")
    print(
        "ROI efetiva: "
        f"TL={cfg.roi_top_left} TR={cfg.roi_top_right} "
        f"BL={cfg.roi_bottom_left} BR={cfg.roi_bottom_right}"
    )
    print(
        f"Pseudo-lanes: esquerda=+{pseudo_left_offset:.1f}px | "
        f"direita=-{pseudo_right_offset:.1f}px"
    )
    print(f"Target LKAS: {args.target}")
    print(
        "Controlador: "
        f"Kp={args.kp:.3f} deadband={args.deadband_px:.1f}px "
        f"alpha={args.smoothing_alpha:.3f} max-step={args.max_step:.3f}"
    )
    if args.export_dataset:
        print(f"Dataset CSV: {args.export_dataset}")
    print("Controles: Q/ESC=sair | ESPACO=pausar/continuar")

    try:
        while True:
            if not paused:
                ok, frame = cap.read()
                if not ok:
                    break

                frame_index = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
                timestamp_ms = float(cap.get(cv2.CAP_PROP_POS_MSEC) or 0.0)
                if timestamp_ms > 0:
                    timestamp_s = timestamp_ms / 1000.0
                elif source_fps > 0:
                    timestamp_s = frame_index / source_fps
                else:
                    timestamp_s = 0.0

                t0 = time.perf_counter()

                source_panel = base.draw_roi(frame, cfg)
                bev = base.apply_bev(frame, cfg)
                bev_clahe = base.apply_clahe(bev, cfg)

                if tracker is None:
                    tracker = LaneTemporalTracker(
                        bev_clahe.shape,
                        history_size=args.history,
                        max_missing=args.hold_frames,
                    )

                prior_left, prior_right = tracker.current()
                mask_small = infer_mask(
                    model,
                    bev_clahe,
                    eval_size,
                    device,
                    image_mean,
                    image_std,
                )
                mask = cv2.resize(
                    mask_small,
                    (bev_clahe.shape[1], bev_clahe.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )

                raw_left, raw_right = detect_lane_pixels(
                    mask,
                    min_rows=args.min_rows,
                    min_pixels_per_row=args.min_pixels_per_row,
                    prior_left=prior_left,
                    prior_right=prior_right,
                )
                left, right = tracker.update(raw_left, raw_right)

                error, target_x, legacy_command = calculate_target_guidance(
                    left,
                    right,
                    bev_clahe.shape[1],
                    target=args.target,
                    pseudo_left_offset_px=pseudo_left_offset,
                    pseudo_right_offset_px=pseudo_right_offset,
                    error_threshold_px=args.error_threshold,
                )

                lane_width = _lane_width(left, right)
                steering_result = controller.update(
                    error_px=None if target_x is None else error,
                    lane_width_px=lane_width,
                    target_available=target_x is not None,
                )
                steering = float(steering_result.steering_filtered)
                steering_delta = steering - previous_steering
                previous_steering = steering
                steering_direction = _steering_direction(
                    steering, abs(args.straight_threshold)
                )

                setpoints = build_setpoint_coefficients(
                    left,
                    right,
                    pseudo_left_offset_px=pseudo_left_offset,
                    pseudo_right_offset_px=pseudo_right_offset,
                )

                result_panel = draw_overlay(
                    bev_clahe,
                    mask,
                    left,
                    right,
                    pseudo_left_offset_px=pseudo_left_offset,
                    pseudo_right_offset_px=pseudo_right_offset,
                    target=args.target,
                )

                elapsed = time.perf_counter() - t0
                processing_ms = elapsed * 1000.0
                inst_fps = 1.0 / elapsed if elapsed > 0 else 0.0
                live_fps = (
                    inst_fps
                    if live_fps is None
                    else 0.9 * live_fps + 0.1 * inst_fps
                )
                processed_frames += 1

                left_status = "OK" if left is not None else "--"
                right_status = "OK" if right is not None else "--"
                target_x_text = "--" if target_x is None else f"{target_x:.1f}"
                normalized_text = (
                    "--"
                    if steering_result.error_normalized is None
                    else f"{steering_result.error_normalized:+.2f}"
                )
                hold_text = " HOLD" if steering_result.held else ""

                source_panel = base.fit_panel(source_panel)
                result_panel = base.fit_panel(result_panel)
                base.put_text(
                    source_panel,
                    "VIDEO ORIGINAL + ROI",
                    f"frame {frame_index}/{max(0, total_frames - 1)} | "
                    f"fonte {source_fps:.1f} FPS | runtime {live_fps:.2f} FPS",
                )
                base.put_text(
                    result_panel,
                    "SEGFORMER-B0 - LKAS CONTINUO",
                    f"{args.target.upper()} x={target_x_text} "
                    f"err={error:+.1f}px norm={normalized_text} "
                    f"steer={steering:+.3f} {steering_direction}{hold_text} "
                    f"L={left_status} R={right_status}",
                )
                mosaic = np.hstack((source_panel, result_panel))

                if video_writer is not None:
                    video_writer.write(mosaic)

                if dataset_writer is not None:
                    dataset_writer.writerow(
                        {
                            "video": (
                                args.video
                                if not args.video_path
                                else Path(video_path).stem
                            ),
                            "video_path": video_path,
                            "config_path": config_path,
                            "frame": frame_index,
                            "timestamp_s": round(timestamp_s, 6),
                            "source_fps": round(source_fps, 6),
                            "bev_width": bev_clahe.shape[1],
                            "bev_height": bev_clahe.shape[0],
                            "target": args.target,
                            "target_available": int(target_x is not None),
                            "vehicle_center_x": round(
                                bev_clahe.shape[1] / 2.0, 6
                            ),
                            "target_x": _round_or_none(target_x),
                            "error_px": (
                                None if target_x is None else round(error, 6)
                            ),
                            "error_threshold_px": round(
                                abs(args.error_threshold), 6
                            ),
                            "command": legacy_command,
                            "pseudo_left_offset_px": round(
                                pseudo_left_offset, 6
                            ),
                            "pseudo_right_offset_px": round(
                                pseudo_right_offset, 6
                            ),
                            "raw_left_detected": int(raw_left is not None),
                            "raw_right_detected": int(raw_right is not None),
                            "left_detected": int(left is not None),
                            "right_detected": int(right is not None),
                            "left_x_base": base._lane_value(left, "x_base"),
                            "right_x_base": base._lane_value(right, "x_base"),
                            "pseudo_left_x_base": base._coefficient_x_base(
                                setpoints["left"], bev_clahe.shape[0]
                            ),
                            "center_x_base": base._coefficient_x_base(
                                setpoints["center"], bev_clahe.shape[0]
                            ),
                            "pseudo_right_x_base": base._coefficient_x_base(
                                setpoints["right"], bev_clahe.shape[0]
                            ),
                            "left_confidence": base._lane_value(
                                left, "confidence"
                            ),
                            "right_confidence": base._lane_value(
                                right, "confidence"
                            ),
                            "left_curvature": base._lane_value(
                                left, "curvature"
                            ),
                            "right_curvature": base._lane_value(
                                right, "curvature"
                            ),
                            "processing_ms": round(processing_ms, 6),
                            "runtime_fps": round(inst_fps, 6),
                            "lane_width_px": _round_or_none(
                                steering_result.lane_width_px
                            ),
                            "error_normalized": _round_or_none(
                                steering_result.error_normalized
                            ),
                            "steering_raw": _round_or_none(
                                steering_result.steering_raw
                            ),
                            "steering_filtered": round(steering, 6),
                            "steering_delta": round(steering_delta, 6),
                            "steering_direction": steering_direction,
                            "steering_held": int(steering_result.held),
                            "steering_saturated": int(
                                steering_result.saturated
                            ),
                            "controller_kp": round(args.kp, 6),
                            "controller_deadband_px": round(
                                abs(args.deadband_px), 6
                            ),
                            "controller_smoothing_alpha": round(
                                args.smoothing_alpha, 6
                            ),
                            "controller_max_step": round(args.max_step, 6),
                            "controller_max_output": round(
                                args.max_output, 6
                            ),
                            "controller_hold_missing_frames": (
                                args.steering_hold_frames
                            ),
                        }
                    )
                    if processed_frames % 100 == 0:
                        dataset_file.flush()

            cv2.imshow(WINDOW_NAME, mosaic)
            key = cv2.waitKeyEx(args.display_delay if not paused else 30)
            if key in (27, ord("q"), ord("Q")):
                break
            if key == 32:
                paused = not paused
    finally:
        cap.release()
        if video_writer is not None:
            video_writer.release()
        if dataset_file is not None:
            dataset_file.flush()
            dataset_file.close()
            print(
                f"Dataset salvo: {args.export_dataset} "
                f"({processed_frames} frames)"
            )
        cv2.destroyAllWindows()


def build_parser() -> argparse.ArgumentParser:
    parser = base.build_parser()
    parser.description = (
        "Runtime LKAS SegFormer-B0 com controlador continuo integrado"
    )
    parser.add_argument("--kp", type=float, default=0.8)
    parser.add_argument("--deadband-px", type=float, default=3.0)
    parser.add_argument("--smoothing-alpha", type=float, default=0.25)
    parser.add_argument("--max-step", type=float, default=0.08)
    parser.add_argument("--max-output", type=float, default=1.0)
    parser.add_argument("--steering-hold-frames", type=int, default=10)
    parser.add_argument(
        "--straight-threshold",
        type=float,
        default=0.05,
        help="Modulo de steering considerado RETO na exibicao",
    )
    return parser


def main():
    run(build_parser().parse_args())
