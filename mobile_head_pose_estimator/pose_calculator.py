import math
import numpy as np
import cv2
from utils import find_rotation_coeffs
from typing import Dict, Tuple, Optional

class HeadPoseCalculator:
    def calculate(self, data: Dict, mode: str = 'pnp') -> Optional[Dict]:
        lm = data['landmarks']
        w, h = data['image_size']

        if mode == 'coeffs':
            sin_b, cos_minor = find_rotation_coeffs(lm['left_eye'], lm['right_eye'], lm['nose'])
            return {'sin_b': sin_b, 'cos_minor': cos_minor}

        if mode == 'pnp':
            return self._pnp(lm, w, h)

        # geometric — безопасный
        return self._geometric_safe(lm)

    def _pnp(self, lm: Dict, w: float, h: float) -> Dict:
        model = np.array([
            [0.0, 0.0, 0.0],
            [-0.1, 0.15, -0.05],
            [0.1, 0.15, -0.05],
            [0.0, -0.12, -0.08]
        ], dtype=np.float32)

        pts = np.array([
            lm['nose'], lm['left_eye'], lm['right_eye'], lm['mouth']
        ], dtype=np.float32)

        # Камера: fx=w, fy=h, cx=w/2, cy=h/2
        K = np.array([[w, 0, w/2], [0, h, h/2], [0, 0, 1]], dtype=np.float32)
        dist = np.zeros((4,1))

        success, rvec, tvec = cv2.solvePnP(model, pts, K, dist)
        if not success:
            return {'error': 'solvePnP failed'}

        R, _ = cv2.Rodrigues(rvec)
        return self._to_euler(R)

    def _geometric_safe(self, lm: Dict) -> Dict:
        le, re, nose, mouth = lm['left_eye'], lm['right_eye'], lm['nose'], lm['mouth']
        mid_x = (le[0] + re[0]) / 2
        mid_y = (le[1] + re[1]) / 2

        # Yaw: по смещению носа
        dx = nose[0] - mid_x
        yaw = math.degrees(math.atan2(dx, 0.1))

        # Pitch: по рту, но с защитой
        dy = mouth[1] - mid_y
        pitch_val = max(-1.0, min(1.0, dy / 0.2))  # ограничение
        pitch = math.degrees(math.asin(pitch_val))

        # Roll: по глазам
        dx_eye = re[0] - le[0]
        dy_eye = re[1] - le[1]
        roll = math.degrees(math.atan2(dy_eye, dx_eye))

        return {'yaw': yaw, 'pitch': pitch, 'roll': roll}

    def _to_euler(self, R: np.ndarray) -> Dict:
        sy = math.sqrt(R[0,0]**2 + R[1,0]**2)
        singular = sy < 1e-6
        if not singular:
            yaw = math.atan2(R[1,0], R[0,0])
            pitch = math.atan2(-R[2,0], sy)
            roll = math.atan2(R[2,1], R[2,2])
        else:
            yaw = math.atan2(-R[1,2], R[1,1])
            pitch = math.atan2(-R[2,0], sy)
            roll = 0
        return {
            'yaw': math.degrees(yaw),
            'pitch': math.degrees(pitch),
            'roll': math.degrees(roll)
        }