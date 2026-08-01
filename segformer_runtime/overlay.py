import cv2
import numpy as np

from .setpoint import build_setpoint_coefficients


def _draw_coefficients(img, coefficients, color, thickness=2):
    if coefficients is None:
        return
    height, width = img.shape[:2]
    y = np.arange(height, dtype=np.float64)
    x = np.polyval(coefficients, y)
    valid = np.isfinite(x) & (x >= 0) & (x < width)
    points = np.column_stack((x[valid], y[valid])).astype(np.int32)
    if len(points) >= 2:
        cv2.polylines(img, [points], False, color, thickness, cv2.LINE_AA)


def draw_overlay(
    image,
    mask,
    left,
    right,
    pseudo_left_offset_px=20.0,
    pseudo_right_offset_px=20.0,
    target="center",
):
    """Desenha mascara, limites reais e as tres trajetorias de setpoint.

    Cores:
    - azul: lane esquerda real;
    - vermelho: lane direita real;
    - ciano: setpoint left;
    - branco: setpoint center;
    - magenta: setpoint right.

    O target selecionado e redesenhado com maior espessura.
    """
    overlay = image.copy()
    tint = image.copy()
    tint[mask > 0] = (0, 255, 255)
    overlay = cv2.addWeighted(overlay, 0.78, tint, 0.22, 0)

    if left is not None and right is not None:
        height = image.shape[0]
        y = np.arange(height, dtype=np.float64)
        lx = np.polyval(left.coefficients, y)
        rx = np.polyval(right.coefficients, y)
        polygon = np.vstack(
            (
                np.column_stack((lx, y)),
                np.flipud(np.column_stack((rx, y))),
            )
        ).astype(np.int32)
        fill = overlay.copy()
        cv2.fillPoly(fill, [polygon], (0, 255, 0))
        overlay = cv2.addWeighted(fill, 0.20, overlay, 0.80, 0)

    # Limites reais detectados.
    _draw_coefficients(overlay, None if left is None else left.coefficients, (255, 0, 0), 3)
    _draw_coefficients(overlay, None if right is None else right.coefficients, (0, 0, 255), 3)

    setpoints = build_setpoint_coefficients(
        left,
        right,
        pseudo_left_offset_px=pseudo_left_offset_px,
        pseudo_right_offset_px=pseudo_right_offset_px,
    )
    colors = {
        "left": (255, 255, 0),
        "center": (255, 255, 255),
        "right": (255, 0, 255),
    }

    for name in ("left", "center", "right"):
        thickness = 4 if name == target else 2
        _draw_coefficients(overlay, setpoints[name], colors[name], thickness)

    return overlay
