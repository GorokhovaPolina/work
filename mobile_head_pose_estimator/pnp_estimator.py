import cv2
import numpy as np
import math
from head_pose_estimator import HeadPoseEstimator

class PnPEstimator(HeadPoseEstimator):
    def estimate(self, landmarks, image_size):
        w, h = image_size

        # 3D модель (в метрах)
        model = np.array([
            [0.0, 0.0, 0.0],           # нос
            [-0.065, 0.035, -0.03],    # левый глаз
            [0.065, 0.035, -0.03],     # правый глаз
            [0.0, -0.065, -0.04]       # рот
        ], dtype=np.float32)

        pts = np.array([
            landmarks['nose'],
            landmarks['left_eye'],
            landmarks['right_eye'],
            landmarks['mouth']
        ], dtype=np.float32)

        K = np.array([[w, 0, w/2], [0, h, h/2], [0, 0, 1]], dtype=np.float32)
        dist = np.zeros((4,1))

        success, rvec, tvec = cv2.solvePnP(model, pts, K, dist, flags=cv2.SOLVEPNP_EPNP)
        if not success:
            return None

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