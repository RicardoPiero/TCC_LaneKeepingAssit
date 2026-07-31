"""Runtime enxuto do LKA usando apenas SegFormer-B0."""

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch

from .geometry import LaneTemporalTracker, detect_lane_pixels
from .model import infer_mask, load_checkpoint
from .overlay import draw_overlay
from .setpoint import calculate_target_guidance


WINDOW_NAME = "LKA - SegFormer-B0"


@dataclass
class RuntimeConfig:
    roi_top_left: tuple
    roi_top_right: tuple
    roi_bottom_left: tuple
    roi_bottom_right: tuple
    bev_width: int
    bev_height: int
    clahe_clip_limit: float
    clahe_grid_size: int


def load_runtime_config(path):
    """Carrega apenas ROI, BEV e CLAHE do JSON usado no projeto."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return RuntimeConfig(
        roi_top_left=tuple(data["roi"]["top_left"]),
        roi_top_right=tuple(data["roi"]["top_right"]),
        roi_bottom_left=tuple(data["roi"]["bottom_left"]),
        roi_bottom_right=tuple(data["roi"]["bottom_right"]),
        bev_width=int(data["bev"]["width"]),
        bev_height=int(data["bev"]["height"]),
        clahe_clip_limit=float(data["clahe"]["clip_limit"]),
        clahe_grid_size=int(data["clahe"]["grid_size"]),
    )


def apply_roi_overrides(cfg, args):
    """Sobrescreve somente os cantos de ROI informados na linha de comando."""
    if args.roi_top_left is not None:
        cfg.roi_top_left = tuple(args.roi_top_left)
    if args.roi_top_right is not None:
        cfg.roi_top_right = tuple(args.roi_top_right)
    if args.roi_bottom_left is not None:
        cfg.roi_bottom_left = tuple(args.roi_bottom_left)
    if args.roi_bottom_right is not None:
        cfg.roi_bottom_right = tuple(args.roi_bottom_right)
    return cfg


def validate_roi(cfg, frame_width, frame_height):
    """Valida a ROI aceitando pontos exatamente nas bordas do frame."""
    points = {
        "top-left": cfg.roi_top_left,
        "top-right": cfg.roi_top_right,
        "bottom-left": cfg.roi_bottom_left,
        "bottom-right": cfg.roi_bottom_right,
    }
    for name, point in points.items():
        x, y = point
        if not (0 <= x <= frame_width and 0 <= y <= frame_height):
            raise ValueError(
                f"ROI {name} fora do frame: ({x}, {y}); "
                f"resolucao={frame_width}x{frame_height}"
            )


def resolve_pseudo_offsets(args):
    """Resolve offsets independentes, mantendo compatibilidade com --pseudo-offset."""
    common = 20.0 if args.pseudo_offset is None else abs(float(args.pseudo_offset))
    left = common if args.pseudo_left_offset is None else abs(float(args.pseudo_left_offset))
    right = common if args.pseudo_right_offset is None else abs(float(args.pseudo_right_offset))
    return left, right


def draw_roi(frame, cfg):
    out = frame.copy()
    pts = np.asarray([
        cfg.roi_top_left,
        cfg.roi_top_right,
        cfg.roi_bottom_right,
        cfg.roi_bottom_left,
    ], dtype=np.int32)
    cv2.polylines(out, [pts], True, (0, 255, 0), 2)
    for label, point in zip(("TL", "TR", "BR", "BL"), pts):
        cv2.circle(out, tuple(point), 5, (0, 255, 0), -1)
        cv2.putText(out, label, tuple(point), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return out


def apply_bev(frame, cfg):
    src = np.float32([
        cfg.roi_top_left,
        cfg.roi_top_right,
        cfg.roi_bottom_right,
        cfg.roi_bottom_left,
    ])
    dst = np.float32([
        [0, 0],
        [cfg.bev_width, 0],
        [cfg.bev_width, cfg.bev_height],
        [0, cfg.bev_height],
    ])
    matrix = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(frame, matrix, (cfg.bev_width, cfg.bev_height))


def apply_clahe(image, cfg):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(
        clipLimit=cfg.clahe_clip_limit,
        tileGridSize=(cfg.clahe_grid_size, cfg.clahe_grid_size),
    )
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


def fit_panel(image, width=640, height=720):
    h, w = image.shape[:2]
    scale = min(width / max(1, w), height / max(1, h))
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    x0 = (width - nw) // 2
    y0 = (height - nh) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = resized
    return canvas


def put_text(panel, title, details):
    cv2.rectangle(panel, (0, 0), (panel.shape[1], 68), (0, 0, 0), -1)
    cv2.putText(panel, title, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(panel, details, (12, 53), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (220, 220, 220), 1, cv2.LINE_AA)


def resolve_video(video_name, video_path, config_path):
    presets = {
        "video_01": ("data/raw/test_video.mp4", "config/test_video_params.json"),
        "test_video": ("data/raw/test_video.mp4", "config/test_video_params.json"),
        "video_02": ("data/raw/video_02.mp4", "config/video_02_params.json"),
    }
    if video_path:
        if not config_path:
            raise ValueError("Ao usar --video-path, informe tambem --config.")
        return video_path, config_path
    if video_name not in presets:
        raise ValueError(f"Video desconhecido: {video_name}")
    default_video, default_config = presets[video_name]
    return default_video, config_path or default_config


def run(args):
    video_path, config_path = resolve_video(args.video, args.video_path, args.config)

    if not Path(video_path).exists():
        raise FileNotFoundError(f"Video nao encontrado: {video_path}")
    if not Path(config_path).exists():
        raise FileNotFoundError(f"Configuracao nao encontrada: {config_path}")
    if not Path(args.checkpoint).exists():
        raise FileNotFoundError(f"Checkpoint nao encontrado: {args.checkpoint}")

    cfg = apply_roi_overrides(load_runtime_config(config_path), args)
    pseudo_left_offset, pseudo_right_offset = resolve_pseudo_offsets(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, image_mean, image_std = load_checkpoint(args.checkpoint, device)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Nao foi possivel abrir: {video_path}")

    source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    validate_roi(cfg, frame_width, frame_height)
    eval_size = (args.eval_width, args.eval_height)

    tracker = None
    writer = None
    live_fps = None
    paused = False
    mosaic = np.zeros((720, 1280, 3), dtype=np.uint8)

    if args.output_video:
        output = Path(args.output_video)
        output.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(output),
            cv2.VideoWriter_fourcc(*"mp4v"),
            source_fps if source_fps > 0 else 20.0,
            (1280, 720),
        )

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
    print("Controles: Q/ESC=sair | ESPACO=pausar/continuar")

    try:
        while True:
            if not paused:
                ok, frame = cap.read()
                if not ok:
                    break
                frame_index = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
                t0 = time.perf_counter()

                source_panel = draw_roi(frame, cfg)
                bev = apply_bev(frame, cfg)
                bev_clahe = apply_clahe(bev, cfg)

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

                error, target_x, command = calculate_target_guidance(
                    left,
                    right,
                    bev_clahe.shape[1],
                    target=args.target,
                    pseudo_left_offset_px=pseudo_left_offset,
                    pseudo_right_offset_px=pseudo_right_offset,
                    error_threshold_px=args.error_threshold,
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
                inst_fps = 1.0 / elapsed if elapsed > 0 else 0.0
                live_fps = inst_fps if live_fps is None else 0.9 * live_fps + 0.1 * inst_fps

                left_status = "OK" if left is not None else "--"
                right_status = "OK" if right is not None else "--"
                raw_left_status = "OK" if raw_left is not None else "--"
                raw_right_status = "OK" if raw_right is not None else "--"
                target_x_text = "--" if target_x is None else f"{target_x:.1f}"

                source_panel = fit_panel(source_panel)
                result_panel = fit_panel(result_panel)
                put_text(
                    source_panel,
                    "VIDEO ORIGINAL + ROI",
                    f"frame {frame_index}/{max(0, total_frames - 1)} | fonte {source_fps:.1f} FPS | runtime {live_fps:.2f} FPS",
                )
                put_text(
                    result_panel,
                    "SEGFORMER-B0 - LKAS",
                    f"target={args.target.upper()} x={target_x_text} | err={error:+.1f}px | {command} | L={left_status} R={right_status}",
                )
                mosaic = np.hstack((source_panel, result_panel))

                if writer is not None:
                    writer.write(mosaic)

            cv2.imshow(WINDOW_NAME, mosaic)
            key = cv2.waitKeyEx(args.display_delay if not paused else 30)
            if key in (27, ord("q"), ord("Q")):
                break
            if key == 32:
                paused = not paused
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()


def build_parser():
    parser = argparse.ArgumentParser(description="Runtime LKAS usando somente SegFormer-B0")
    parser.add_argument("--video", default="video_02", choices=["video_01", "test_video", "video_02"])
    parser.add_argument("--video-path", default=None, help="Caminho customizado; exige --config")
    parser.add_argument("--config", default=None, help="JSON base de ROI/BEV/CLAHE")
    parser.add_argument("--checkpoint", default="experiments/segformer_b0_final/best.pt")

    # ROI: cada argumento recebe X Y e sobrescreve apenas aquele canto do JSON.
    parser.add_argument("--roi-top-left", nargs=2, type=int, metavar=("X", "Y"), default=None)
    parser.add_argument("--roi-top-right", nargs=2, type=int, metavar=("X", "Y"), default=None)
    parser.add_argument("--roi-bottom-left", nargs=2, type=int, metavar=("X", "Y"), default=None)
    parser.add_argument("--roi-bottom-right", nargs=2, type=int, metavar=("X", "Y"), default=None)

    parser.add_argument("--eval-width", type=int, default=384)
    parser.add_argument("--eval-height", type=int, default=256)
    parser.add_argument("--history", type=int, default=8)
    parser.add_argument("--hold-frames", type=int, default=10)
    parser.add_argument("--min-rows", type=int, default=30)
    parser.add_argument("--min-pixels-per-row", type=int, default=2)

    # Setpoint do LKAS.
    parser.add_argument(
        "--target",
        choices=["left", "center", "right"],
        default="center",
        help="Trajetoria seguida pelo LKAS: pseudo esquerda, centro ou pseudo direita",
    )
    parser.add_argument(
        "--error-threshold",
        type=float,
        default=20.0,
        help="Zona morta do erro lateral em pixels para comando ESQUERDA/RETO/DIREITA",
    )

    # --pseudo-offset continua funcionando e define os dois lados.
    # Os argumentos individuais, quando presentes, tem prioridade.
    parser.add_argument("--pseudo-offset", type=float, default=None, help="Distancia comum das duas pseudo-lanes")
    parser.add_argument("--pseudo-left-offset", type=float, default=None, help="Distancia da lane esquerda para dentro (+X)")
    parser.add_argument("--pseudo-right-offset", type=float, default=None, help="Distancia da lane direita para dentro (-X)")

    parser.add_argument("--display-delay", type=int, default=1)
    parser.add_argument("--output-video", default=None)
    return parser


def main():
    run(build_parser().parse_args())
