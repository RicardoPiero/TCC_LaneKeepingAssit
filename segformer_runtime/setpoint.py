import numpy as np


VALID_TARGETS = ("left", "center", "right")


def build_setpoint_coefficients(
    left,
    right,
    pseudo_left_offset_px=20.0,
    pseudo_right_offset_px=20.0,
):
    """Retorna os coeficientes das trajetorias left, center e right."""
    pseudo_left = None
    center = None
    pseudo_right = None

    if left is not None:
        pseudo_left = np.asarray(left.coefficients, dtype=np.float64).copy()
        pseudo_left[2] += abs(float(pseudo_left_offset_px))

    if left is not None and right is not None:
        center = (
            np.asarray(left.coefficients, dtype=np.float64)
            + np.asarray(right.coefficients, dtype=np.float64)
        ) / 2.0

    if right is not None:
        pseudo_right = np.asarray(right.coefficients, dtype=np.float64).copy()
        pseudo_right[2] -= abs(float(pseudo_right_offset_px))

    return {
        "left": pseudo_left,
        "center": center,
        "right": pseudo_right,
    }


def calculate_target_guidance(
    left,
    right,
    img_width,
    target="center",
    pseudo_left_offset_px=20.0,
    pseudo_right_offset_px=20.0,
    error_threshold_px=20.0,
):
    """Calcula erro lateral e comando em relacao ao setpoint selecionado.

    O erro segue a convencao historica do projeto:
        erro = centro_do_veiculo - x_do_setpoint

    Retorna:
        (erro_px, target_x, comando)

    target_x e None quando a trajetoria solicitada nao pode ser calculada.
    """
    if target not in VALID_TARGETS:
        raise ValueError(f"Target invalido: {target}. Use: {', '.join(VALID_TARGETS)}")

    if target == "left":
        if left is None:
            return 0.0, None, "SEM TARGET"
        target_x = float(left.x_base + abs(float(pseudo_left_offset_px)))

    elif target == "center":
        if left is None or right is None:
            return 0.0, None, "SEM TARGET"
        target_x = float((left.x_base + right.x_base) / 2.0)

    else:  # right
        if right is None:
            return 0.0, None, "SEM TARGET"
        target_x = float(right.x_base - abs(float(pseudo_right_offset_px)))

    vehicle_center = float(img_width) / 2.0
    error = vehicle_center - target_x
    threshold = abs(float(error_threshold_px))

    if error > threshold:
        command = "ESQUERDA"
    elif error < -threshold:
        command = "DIREITA"
    else:
        command = "RETO"

    return float(error), target_x, command
