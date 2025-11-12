import math
from typing import Tuple

def find_rotation_coeffs(
    le: Tuple[float, float],
    re: Tuple[float, float],
    nose: Tuple[float, float]
) -> Tuple[float, float]:
    """
    sin_b — мера поворота (0 = фронтально)
    cos_minor — направление в плоскости (1 = нос вниз)
    Работает с нормализованными [0,1] или пикселями.
    """
    eye_vec = (re[0] - le[0], re[1] - le[1])
    eye_len = math.hypot(*eye_vec)
    if eye_len == 0:
        return 0.0, 1.0

    eye_norm = (eye_vec[0] / eye_len, eye_vec[1] / eye_len)
    mid_eye = ((le[0] + re[0]) / 2, (le[1] + re[1]) / 2)
    nose_vec = (nose[0] - mid_eye[0], nose[1] - mid_eye[1])
    nose_len = math.hypot(*nose_vec)
    if nose_len == 0:
        return 0.0, 1.0

    dot = eye_norm[0] * nose_vec[0] + eye_norm[1] * nose_vec[1]
    perp_vec = (nose_vec[0] - dot * eye_norm[0], nose_vec[1] - dot * eye_norm[1])
    perp_len = math.hypot(*perp_vec)
    if perp_len == 0:
        return 0.0, 1.0

    sin_b = abs(dot) / nose_len
    cos_minor = perp_vec[1] / perp_len
    return sin_b, cos_minor