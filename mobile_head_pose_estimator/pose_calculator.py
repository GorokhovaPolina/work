import cv2
import numpy as np
import math
from utils import find_rotation_coeffs

class GeometricPoseCalculator:
    def calculate_pose(self, data, mode='pnp'):
        lm = data['landmarks']
        w, h = data['image_size']

        if mode == 'coeffs':
            sin_b, cos_minor = find_rotation_coeffs(lm['left_eye'], lm['right_eye'], lm['nose'])
            if sin_b == -8.0:
                return {'error': 'invalid geometry'}
            return {'sin_b': sin_b, 'cos_minor': cos_minor}

        # === PnP ===
        model = np.array([
            [0.0, 0.0, 0.0],
            [-0.065, 0.035, -0.03],
            [0.065, 0.035, -0.03],
            [0.0, -0.065, -0.04]
        ], dtype=np.float32)

        pts = np.array([
            lm['nose'], lm['left_eye'], lm['right_eye'], lm['mouth']
        ], dtype=np.float32)

        K = np.array([[w, 0, w/2], [0, h, h/2], [0, 0, 1]], dtype=np.float32)
        dist = np.zeros((4,1))

        success, rvec, tvec = cv2.solvePnP(model, pts, K, dist, flags=cv2.SOLVEPNP_EPNP)
        if not success:
            return {'error': 'solvePnP failed'}

        R, _ = cv2.Rodrigues(rvec)
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
            'roll': math.degrees(roll),
            'rvec': rvec,
            'tvec': tvec,
            'K': K,
            'dist': dist
        }
