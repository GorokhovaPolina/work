import cv2
import numpy as np
import math

def find_rotation_coeffs(left_eye, right_eye, nose):
    le = np.array(left_eye, dtype=float)
    re = np.array(right_eye, dtype=float)
    no = np.array(nose, dtype=float)

    mid = (le + re) / 2.0
    v = no - mid
    norm = np.linalg.norm(v)
    if norm < 1e-6:
        return -8.0, -8.0
    sin_b = float(v[1] / norm)
    sin_b = max(-1.0, min(1.0, sin_b))
    cos_minor = float(math.sqrt(max(0.0, 1.0 - sin_b * sin_b)))
    return sin_b, cos_minor

def rotation_matrix_to_euler(R):
    sy = math.sqrt(R[0,0] * R[0,0] + R[1,0] * R[1,0])
    singular = sy < 1e-6
    if not singular:
        x = math.atan2(R[2,1], R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else:
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0.0
    return np.array([x, y, z], dtype=float)

class GeometricPoseCalculator:
    def __init__(self):
        self.model_points = np.array([
            (-3.2,  0.0,  0.0),
            ( 3.2,  0.0,  0.0),
            ( 0.0, -2.0,  6.0)
        ], dtype=np.float64)

    def _approx_pose_from_2d(self, left_eye, right_eye, nose):
        le = np.array(left_eye, dtype=float)
        re = np.array(right_eye, dtype=float)
        no = np.array(nose, dtype=float)
        dx = re[0] - le[0]
        dy = re[1] - le[1]
        roll = math.degrees(math.atan2(dy, dx))
        mid = (le + re) / 2.0
        half_ipd = max(1.0, dx / 2.0)
        yaw = (no[0] - mid[0]) / half_ipd * 20.0
        pitch = - (no[1] - mid[1]) / half_ipd * 20.0
        yaw = max(-180.0, min(180.0, yaw))
        pitch = max(-90.0, min(90.0, pitch))
        roll = (roll + 180.0) % 360.0 - 180.0
        return {'yaw': round(yaw, 2), 'pitch': round(pitch, 2), 'roll': round(roll, 2), 'method': 'approx'}

    def calculate_pose(self, data, mode='pnp'):
        lm = data['landmarks']
        w, h = data['image_size']

        left_eye = tuple(lm['left_eye'])
        right_eye = tuple(lm['right_eye'])
        nose = tuple(lm['nose'])

        if mode == 'coeffs':
            sin_b, cos_minor = find_rotation_coeffs(left_eye, right_eye, nose)
            if sin_b == -8.0:
                return {'error': 'invalid geometry'}
            return {'sin_b': sin_b, 'cos_minor': cos_minor}

        image_points = np.array([left_eye, right_eye, nose], dtype=np.float64)
        model_points = self.model_points.astype(np.float64)
        focal_length = float(w)
        center = (w / 2.0, h / 2.0)
        K = np.array([[focal_length, 0.0, center[0]],[0.0, focal_length, center[1]],[0.0, 0.0, 1.0]], dtype=np.float64)
        dist = np.zeros((4,1), dtype=np.float64)

        try:
            success, rvec, tvec = cv2.solvePnP(model_points, image_points, K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
            if not success:
                raise RuntimeError('solvePnP returned False')
            R, _ = cv2.Rodrigues(rvec)
            angles_rad = rotation_matrix_to_euler(R)
            pitch = math.degrees(angles_rad[0])
            yaw   = math.degrees(angles_rad[1])
            roll  = math.degrees(angles_rad[2])
            def norm(a):
                a = (a + 180.0) % 360.0 - 180.0
                if abs(a) < 1e-6:
                    a = 0.0
                return a
            return {
                'yaw': norm(yaw),
                'pitch': norm(pitch),
                'roll': norm(roll),
                'rvec': rvec,
                'tvec': tvec,
                'K': K,
                'dist': dist,
                'method': 'pnp'
            }
        except Exception as e:
            approx = self._approx_pose_from_2d(left_eye, right_eye, nose)
            approx['error'] = f'solvePnP failed: {e}'
            return approx