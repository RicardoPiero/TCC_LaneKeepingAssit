from collections import deque
from dataclasses import dataclass
import time
from typing import Optional, Tuple

import cv2
import numpy as np


@dataclass
class LaneDetection:
    coefficients: np.ndarray
    x_base: float
    x_top: float
    curvature: float
    confidence: float
    timestamp: float


def calculate_curvature(coeffs: np.ndarray, y_eval: float) -> float:
    a, b, _ = coeffs
    dx_dy = 2.0 * a * y_eval + b
    d2x_dy2 = 2.0 * a
    if abs(d2x_dy2) < 1e-6:
        return 999999.0
    return float(((1.0 + dx_dy ** 2) ** 1.5) / abs(d2x_dy2))


def _make_lane(coeffs, height, confidence):
    coeffs = np.asarray(coeffs, dtype=np.float64)
    return LaneDetection(
        coefficients=coeffs,
        x_base=float(np.polyval(coeffs, height - 1)),
        x_top=float(np.polyval(coeffs, 0)),
        curvature=calculate_curvature(coeffs, height - 1),
        confidence=float(confidence),
        timestamp=time.time(),
    )


def _smooth_histogram(values: np.ndarray, kernel_size: int = 11) -> np.ndarray:
    kernel_size = max(1, int(kernel_size))
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = np.ones(kernel_size, dtype=np.float64) / kernel_size
    return np.convolve(values.astype(np.float64), kernel, mode="same")


def _find_seed_x(binary: np.ndarray, x_min: int, x_max: int) -> Optional[float]:
    height, width = binary.shape
    x_min = max(0, int(x_min))
    x_max = min(width, int(x_max))
    if x_max <= x_min:
        return None

    for fraction in (0.35, 0.55, 0.80, 1.00):
        y0 = int(round(height * (1.0 - fraction)))
        histogram = binary[y0:, x_min:x_max].sum(axis=0)
        if histogram.size == 0 or float(histogram.max()) <= 0:
            continue
        smooth = _smooth_histogram(histogram, kernel_size=11)
        return float(x_min + int(np.argmax(smooth)))
    return None


def _nearest_run_center(xs: np.ndarray, expected_x: float, max_gap: int = 3) -> Optional[float]:
    if xs.size == 0:
        return None
    xs = np.asarray(xs, dtype=np.float64)
    if xs.size == 1:
        return float(xs[0])

    breaks = np.flatnonzero(np.diff(xs) > max_gap) + 1
    groups = np.split(xs, breaks)
    centers = np.asarray([np.median(group) for group in groups if group.size], dtype=np.float64)
    if centers.size == 0:
        return None
    return float(centers[np.argmin(np.abs(centers - expected_x))])


def _robust_polyfit(y_arr: np.ndarray, x_arr: np.ndarray, min_rows: int):
    keep = np.ones(len(y_arr), dtype=bool)
    coeffs = None

    for _ in range(3):
        if int(keep.sum()) < min_rows:
            break
        try:
            coeffs = np.polyfit(y_arr[keep], x_arr[keep], 2)
        except np.linalg.LinAlgError:
            return None, keep

        residuals = np.abs(x_arr - np.polyval(coeffs, y_arr))
        active = residuals[keep]
        median_residual = float(np.median(active))
        mad = float(np.median(np.abs(active - median_residual)))
        robust_sigma = 1.4826 * mad
        residual_limit = max(3.0, median_residual + 3.0 * robust_sigma)
        new_keep = residuals <= residual_limit
        if int(new_keep.sum()) < min_rows or np.array_equal(new_keep, keep):
            break
        keep = new_keep

    return coeffs, keep


def _trace_side_with_windows(
    binary: np.ndarray,
    x_min: int,
    x_max: int,
    min_rows: int,
    min_pixels_per_row: int,
    prior_lane: Optional[LaneDetection] = None,
    n_windows: int = 14,
) -> Optional[LaneDetection]:
    height, width = binary.shape
    x_min = max(0, int(x_min))
    x_max = min(width, int(x_max))
    if x_max <= x_min:
        return None

    if prior_lane is not None:
        seed_x = float(np.polyval(prior_lane.coefficients, height - 1))
        if not (x_min <= seed_x < x_max):
            seed_x = _find_seed_x(binary, x_min, x_max)
    else:
        seed_x = _find_seed_x(binary, x_min, x_max)

    if seed_x is None:
        return None

    n_windows = max(5, int(n_windows))
    window_height = max(1, int(np.ceil(height / n_windows)))
    half_width = max(24, int(round(width * 0.085)))
    current_x = float(seed_x)
    prior_offset = 0.0
    collected_y, collected_x = [], []

    for window in range(n_windows):
        y_high = height - window * window_height
        y_low = max(0, y_high - window_height)
        if y_high <= y_low:
            continue

        local_y, local_x, local_residuals = [], [], []
        for y in range(y_low, y_high):
            if prior_lane is not None:
                expected_x = float(np.polyval(prior_lane.coefficients, y) + prior_offset)
            else:
                expected_x = current_x

            search_left = max(x_min, int(round(expected_x - half_width)))
            search_right = min(x_max, int(round(expected_x + half_width + 1)))
            if search_right <= search_left:
                continue

            row_x_local = np.flatnonzero(binary[y, search_left:search_right])
            if row_x_local.size < min_pixels_per_row:
                continue

            row_x = row_x_local.astype(np.float64) + search_left
            x_center = _nearest_run_center(row_x, expected_x)
            if x_center is None or abs(x_center - expected_x) > half_width * 0.85:
                continue

            local_y.append(float(y))
            local_x.append(float(x_center))
            local_residuals.append(float(x_center - expected_x))

        if local_y:
            collected_y.extend(local_y)
            collected_x.extend(local_x)
            min_recent_rows = max(3, int(round(window_height * 0.10)))
            if len(local_x) >= min_recent_rows:
                if prior_lane is not None:
                    residual_step = float(np.median(local_residuals))
                    max_residual_step = max(4.0, width * 0.025)
                    prior_offset += float(np.clip(residual_step, -max_residual_step, max_residual_step))
                    prior_offset = float(np.clip(prior_offset, -half_width * 0.65, half_width * 0.65))
                else:
                    next_x = float(np.median(local_x))
                    max_step = half_width * 0.55
                    current_x += float(np.clip(next_x - current_x, -max_step, max_step))

    if len(collected_y) < min_rows:
        return None

    y_arr = np.asarray(collected_y, dtype=np.float64)
    x_arr = np.asarray(collected_x, dtype=np.float64)
    coeffs, keep = _robust_polyfit(y_arr, x_arr, min_rows)
    if coeffs is None or int(keep.sum()) < min_rows:
        return None

    used_y = y_arr[keep]
    used_x = x_arr[keep]
    support_span = float(np.max(used_y) - np.min(used_y))
    if support_span < height * 0.18:
        return None

    sample_y = np.linspace(float(np.min(used_y)), float(np.max(used_y)), 7)
    sample_x = np.polyval(coeffs, sample_y)
    tolerance = max(12.0, width * 0.07)
    if np.any(sample_x < x_min - tolerance) or np.any(sample_x > x_max + tolerance):
        return None

    fit_residual = np.abs(used_x - np.polyval(coeffs, used_y))
    if float(np.median(fit_residual)) > max(8.0, width * 0.025):
        return None

    used_rows = int(keep.sum())
    confidence_rows = used_rows / max(float(min_rows), height * 0.60)
    confidence_span = support_span / max(1.0, height * 0.65)
    confidence = min(1.0, 0.65 * confidence_rows + 0.35 * confidence_span)
    return _make_lane(coeffs, height, confidence)


def _pair_is_plausible(left, right, height, width) -> bool:
    if left is None or right is None:
        return True
    sample_y = np.linspace(height * 0.35, height - 1, 8)
    left_x = np.polyval(left.coefficients, sample_y)
    right_x = np.polyval(right.coefficients, sample_y)
    lane_width = right_x - left_x
    min_lane_width = max(20.0, width * 0.08)
    max_lane_width = width * 0.98
    if np.any(lane_width <= min_lane_width) or np.any(lane_width >= max_lane_width):
        return False
    median_width = float(np.median(lane_width))
    if median_width <= 0:
        return False
    return float(np.ptp(lane_width)) <= max(width * 0.42, median_width * 0.85)


def detect_lane_pixels(
    mask: np.ndarray,
    min_rows: int = 30,
    min_pixels_per_row: int = 2,
    prior_left: Optional[LaneDetection] = None,
    prior_right: Optional[LaneDetection] = None,
) -> Tuple[Optional[LaneDetection], Optional[LaneDetection]]:
    if mask.ndim == 3:
        mask = mask[..., 0]
    binary = np.asarray(mask) > 0
    height, width = binary.shape
    split = width // 2

    left = _trace_side_with_windows(binary, 0, max(1, split), min_rows, min_pixels_per_row, prior_left)
    right = _trace_side_with_windows(binary, min(width - 1, split), width, min_rows, min_pixels_per_row, prior_right)

    if not _pair_is_plausible(left, right, height, width):
        left_conf = left.confidence if left is not None else -1.0
        right_conf = right.confidence if right is not None else -1.0
        if left_conf >= right_conf:
            right = None
        else:
            left = None
    return left, right


class LaneTemporalTracker:
    def __init__(self, image_shape, history_size: int = 8, max_missing: int = 10):
        self.height, self.width = image_shape[:2]
        self.history_size = max(1, int(history_size))
        self.max_missing = max(0, int(max_missing))
        self.left_history = deque(maxlen=self.history_size)
        self.right_history = deque(maxlen=self.history_size)
        self.left_missing = 0
        self.right_missing = 0

    def _smooth(self, history):
        if not history:
            return None
        lanes = list(history)
        weights = np.exp(np.linspace(-2.0, 0.0, len(lanes)))
        weights /= weights.sum()
        coeffs = np.average(np.stack([lane.coefficients for lane in lanes]), axis=0, weights=weights)
        confidence = float(np.average([lane.confidence for lane in lanes], weights=weights))
        return _make_lane(coeffs, self.height, confidence)

    def _jump_is_plausible(self, new_lane, history):
        if new_lane is None or not history:
            return True
        previous = self._smooth(history)
        sample_y = np.asarray([self.height * 0.40, self.height * 0.70, self.height - 1.0])
        prev_x = np.polyval(previous.coefficients, sample_y)
        new_x = np.polyval(new_lane.coefficients, sample_y)
        max_jump = max(28.0, self.width * 0.15)
        return float(np.max(np.abs(new_x - prev_x))) <= max_jump

    def update(self, left, right):
        if not self._jump_is_plausible(left, self.left_history):
            left = None
        if not self._jump_is_plausible(right, self.right_history):
            right = None
        if not _pair_is_plausible(left, right, self.height, self.width):
            left_conf = left.confidence if left is not None else -1.0
            right_conf = right.confidence if right is not None else -1.0
            if left_conf >= right_conf:
                right = None
            else:
                left = None

        if left is not None:
            self.left_history.append(left)
            self.left_missing = 0
        else:
            self.left_missing += 1
            if self.left_missing > self.max_missing:
                self.left_history.clear()

        if right is not None:
            self.right_history.append(right)
            self.right_missing = 0
        else:
            self.right_missing += 1
            if self.right_missing > self.max_missing:
                self.right_history.clear()
        return self.current()

    def current(self):
        left = self._smooth(self.left_history)
        right = self._smooth(self.right_history)
        if not _pair_is_plausible(left, right, self.height, self.width):
            left_conf = left.confidence if left is not None else -1.0
            right_conf = right.confidence if right is not None else -1.0
            if left_conf >= right_conf:
                right = None
            else:
                left = None
        return left, right


def derive_pseudo_lanes(left, right, height, inner_offset_px=20.0):
    inner_left = center = inner_right = None
    if left is not None:
        coeffs = left.coefficients.copy()
        coeffs[2] += float(inner_offset_px)
        inner_left = _make_lane(coeffs, height, left.confidence)
    if right is not None:
        coeffs = right.coefficients.copy()
        coeffs[2] -= float(inner_offset_px)
        inner_right = _make_lane(coeffs, height, right.confidence)
    if left is not None and right is not None:
        coeffs = (left.coefficients + right.coefficients) / 2.0
        center = _make_lane(coeffs, height, min(left.confidence, right.confidence))
    return inner_left, center, inner_right


def calculate_guidance(left, right, img_width):
    vehicle_center = img_width / 2.0
    if left is not None and right is not None:
        lane_center = (left.x_base + right.x_base) / 2.0
    elif left is not None:
        lane_center = left.x_base + img_width / 4.0
    elif right is not None:
        lane_center = right.x_base - img_width / 4.0
    else:
        return 0.0, "SEM DETECCAO"

    offset = vehicle_center - lane_center
    if offset > 20:
        command = "ESQUERDA"
    elif offset < -20:
        command = "DIREITA"
    else:
        command = "RETO"
    return float(offset), command


def _draw_lane(img, lane, color, thickness=2):
    if lane is None:
        return
    height, width = img.shape[:2]
    y = np.arange(height, dtype=np.float64)
    x = np.polyval(lane.coefficients, y)
    valid = np.isfinite(x) & (x >= 0) & (x < width)
    points = np.column_stack((x[valid], y[valid])).astype(np.int32)
    if len(points) >= 2:
        cv2.polylines(img, [points], False, color, thickness, cv2.LINE_AA)


def draw_overlay(image, mask, left, right, pseudo_offset_px=20.0):
    """Desenha mascara, lanes reais, area da pista e as 3 pseudo-lanes."""
    overlay = image.copy()
    tint = image.copy()
    tint[mask > 0] = (0, 255, 255)
    overlay = cv2.addWeighted(overlay, 0.78, tint, 0.22, 0)

    if left is not None and right is not None:
        height = image.shape[0]
        y = np.arange(height, dtype=np.float64)
        lx = np.polyval(left.coefficients, y)
        rx = np.polyval(right.coefficients, y)
        polygon = np.vstack((np.column_stack((lx, y)), np.flipud(np.column_stack((rx, y))))).astype(np.int32)
        fill = overlay.copy()
        cv2.fillPoly(fill, [polygon], (0, 255, 0))
        overlay = cv2.addWeighted(fill, 0.20, overlay, 0.80, 0)

    _draw_lane(overlay, left, (255, 0, 0), 3)
    _draw_lane(overlay, right, (0, 0, 255), 3)

    inner_left, center, inner_right = derive_pseudo_lanes(left, right, image.shape[0], pseudo_offset_px)
    _draw_lane(overlay, inner_left, (255, 255, 0), 2)
    _draw_lane(overlay, center, (255, 255, 255), 2)
    _draw_lane(overlay, inner_right, (255, 0, 255), 2)
    return overlay
