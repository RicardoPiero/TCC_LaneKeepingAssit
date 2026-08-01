"""Controlador contínuo de direção para o LKAS.

Convenção de sinais do projeto:
- steering > 0: correção para a esquerda;
- steering < 0: correção para a direita;
- steering = 0: manter direção.

O erro lateral recebido segue a mesma convenção:
    error_px = vehicle_center_x - target_x
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class SteeringResult:
    lane_width_px: Optional[float]
    error_normalized: Optional[float]
    steering_raw: Optional[float]
    steering_filtered: float
    target_available: bool
    held: bool
    saturated: bool


class ProportionalSteeringController:
    """Controlador P normalizado, suavizado e limitado por frame.

    A normalização usa metade da largura estimada da pista:
        error_normalized = error_px / (lane_width_px / 2)

    Depois são aplicados:
    1. zona morta em pixels;
    2. ganho proporcional ``kp``;
    3. saturação em ``[-max_output, +max_output]``;
    4. filtro exponencial;
    5. limite de variação por frame.
    """

    def __init__(
        self,
        kp: float = 0.8,
        deadband_px: float = 3.0,
        smoothing_alpha: float = 0.25,
        max_step_per_frame: float = 0.08,
        max_output: float = 1.0,
        hold_missing_frames: int = 10,
    ) -> None:
        if kp < 0:
            raise ValueError("kp deve ser >= 0")
        if not 0 < smoothing_alpha <= 1:
            raise ValueError("smoothing_alpha deve estar em (0, 1]")
        if max_step_per_frame <= 0:
            raise ValueError("max_step_per_frame deve ser > 0")
        if max_output <= 0:
            raise ValueError("max_output deve ser > 0")
        if hold_missing_frames < 0:
            raise ValueError("hold_missing_frames deve ser >= 0")

        self.kp = float(kp)
        self.deadband_px = abs(float(deadband_px))
        self.smoothing_alpha = float(smoothing_alpha)
        self.max_step_per_frame = float(max_step_per_frame)
        self.max_output = float(max_output)
        self.hold_missing_frames = int(hold_missing_frames)

        self._steering = 0.0
        self._missing_count = 0

    @property
    def steering(self) -> float:
        return float(self._steering)

    def reset(self, steering: float = 0.0) -> None:
        self._steering = float(
            np.clip(steering, -self.max_output, self.max_output)
        )
        self._missing_count = 0

    def _rate_limit(self, desired: float) -> float:
        delta = float(desired) - self._steering
        delta = float(
            np.clip(delta, -self.max_step_per_frame, self.max_step_per_frame)
        )
        self._steering = float(
            np.clip(
                self._steering + delta,
                -self.max_output,
                self.max_output,
            )
        )
        return self._steering

    def update(
        self,
        error_px: Optional[float],
        lane_width_px: Optional[float],
        target_available: bool = True,
    ) -> SteeringResult:
        valid_width = (
            lane_width_px is not None
            and np.isfinite(lane_width_px)
            and float(lane_width_px) > 1.0
        )
        valid_error = error_px is not None and np.isfinite(error_px)
        available = bool(target_available and valid_width and valid_error)

        if not available:
            self._missing_count += 1
            held = self._missing_count <= self.hold_missing_frames

            if held:
                filtered = self._steering
            else:
                filtered = self._rate_limit(0.0)

            return SteeringResult(
                lane_width_px=None if not valid_width else float(lane_width_px),
                error_normalized=None,
                steering_raw=None,
                steering_filtered=float(filtered),
                target_available=False,
                held=held,
                saturated=False,
            )

        self._missing_count = 0
        lane_width = float(lane_width_px)
        error = float(error_px)
        half_width = max(lane_width / 2.0, 1.0)

        if abs(error) <= self.deadband_px:
            normalized = 0.0
        else:
            normalized = float(np.clip(error / half_width, -1.0, 1.0))

        proportional = self.kp * normalized
        raw = float(np.clip(proportional, -self.max_output, self.max_output))
        saturated = abs(proportional) > self.max_output

        smoothed = (
            self.smoothing_alpha * raw
            + (1.0 - self.smoothing_alpha) * self._steering
        )
        filtered = self._rate_limit(smoothed)

        return SteeringResult(
            lane_width_px=lane_width,
            error_normalized=normalized,
            steering_raw=raw,
            steering_filtered=float(filtered),
            target_available=True,
            held=False,
            saturated=saturated,
        )
