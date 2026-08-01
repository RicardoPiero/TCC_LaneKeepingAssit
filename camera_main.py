from __future__ import annotations

import argparse
import os
from pathlib import Path as RealPath
import sys
import time

import cv2

from segformer_runtime import pipeline_control as runtime


class _CameraPath:
    def __init__(self, value: str) -> None:
        self.value = value
        index = value.split("://", 1)[1]
        self._stem = f"camera_{index}"

    def exists(self) -> bool:
        return True

    @property
    def stem(self) -> str:
        return self._stem

    def __str__(self) -> str:
        return self.value


class _CameraCapture:
    def __init__(
        self,
        original_video_capture,
        index: int,
        width: int | None,
        height: int | None,
        fps: float | None,
    ) -> None:
        backend = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY
        self._cap = original_video_capture(index, backend)
        self._frame_count = 0
        self._started_at = time.perf_counter()

        if width is not None:
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(width))
        if height is not None:
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(height))
        if fps is not None:
            self._cap.set(cv2.CAP_PROP_FPS, float(fps))

        # Reduz a fila de frames quando o backend oferece suporte.
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    def isOpened(self) -> bool:
        return bool(self._cap.isOpened())

    def read(self):
        ok, frame = self._cap.read()
        if ok:
            self._frame_count += 1
        return ok, frame

    def get(self, prop_id: int) -> float:
        if prop_id == cv2.CAP_PROP_POS_FRAMES:
            return float(self._frame_count)
        if prop_id == cv2.CAP_PROP_POS_MSEC:
            return (time.perf_counter() - self._started_at) * 1000.0
        if prop_id == cv2.CAP_PROP_FRAME_COUNT:
            return 0.0
        return float(self._cap.get(prop_id))

    def set(self, prop_id: int, value: float) -> bool:
        return bool(self._cap.set(prop_id, value))

    def release(self) -> None:
        self._cap.release()


def _build_camera_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--camera-width", type=int, default=None)
    parser.add_argument("--camera-height", type=int, default=None)
    parser.add_argument("--camera-fps", type=float, default=None)
    return parser


def main() -> None:
    camera_parser = _build_camera_parser()
    camera_args, remaining = camera_parser.parse_known_args()

    if "--video-path" in remaining:
        raise ValueError(
            "Nao use --video-path com camera_main.py; use --camera-index."
        )
    if "--config" not in remaining:
        raise ValueError(
            "A camera exige --config com ROI/BEV/CLAHE calibrados para o enquadramento."
        )

    sentinel = f"camera://{camera_args.camera_index}"
    original_video_capture = runtime.cv2.VideoCapture

    def camera_aware_path(value):
        text = str(value)
        if text.startswith("camera://"):
            return _CameraPath(text)
        return RealPath(value)

    def camera_aware_capture(source):
        if isinstance(source, str) and source.startswith("camera://"):
            index = int(source.split("://", 1)[1])
            return _CameraCapture(
                original_video_capture,
                index=index,
                width=camera_args.camera_width,
                height=camera_args.camera_height,
                fps=camera_args.camera_fps,
            )
        return original_video_capture(source)

    runtime.Path = camera_aware_path
    runtime.cv2.VideoCapture = camera_aware_capture

    sys.argv = [sys.argv[0], *remaining, "--video-path", sentinel]
    runtime.main()


if __name__ == "__main__":
    main()
