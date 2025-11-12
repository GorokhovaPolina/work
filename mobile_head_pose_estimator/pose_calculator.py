import cv2
import numpy as np
import math
from utils import find_rotation_coeffs

class GeometricPoseCalculator:
    def calculate_pose(self, data, mode='pnp'):
        """
        Unified calculate_pose keeping original signature.
        mode:
        - 'coeffs' -> returns {'sin_b','cos_minor'} (old behaviour)
        - 'pnp'    -> tries solvePnP, returns yaw/pitch/roll + rvec/tvec/K/dist if success.
                        On failure falls back to pure geometric estimate (no exception).
        - 'geom'   -> pure 2D-geometric estimate (yaw,pitch,roll,sin_b,cos_minor,method='geom')
        """
        import numpy as np
        import math
        import cv2

        lm = data.get('landmarks', {})
        w, h = data.get('image_size', (0,0))

        # Extract points with fallback to None
        left_eye = tuple(lm.get('left_eye')) if 'left_eye' in lm else None
        right_eye = tuple(lm.get('right_eye')) if 'right_eye' in lm else None
        nose = tuple(lm.get('nose')) if 'nose' in lm else None

        # Safety checks
        if left_eye is None or right_eye is None or nose is None:
            return {'error': 'missing landmarks'}

        # Helper: geometric coeffs
        def _find_rotation_coeffs(le, re, no):
            le = np.array(le, dtype=float)
            re = np.array(re, dtype=float)
            no = np.array(no, dtype=float)
            mid = (le + re) / 2.0
            v = no - mid
            norm = np.linalg.norm(v)
            if norm < 1e-6:
                return -8.0, 0.0
            sin_b = float(v[1] / norm)      # image y grows down => nose below eyes => positive
            sin_b = max(-1.0, min(1.0, sin_b))
            cos_minor = float(math.sqrt(max(0.0, 1.0 - sin_b * sin_b)))
            return sin_b, cos_minor

        # Helper: pure geometric estimation (2D)
        def _geom_estimate(le, re, no):
            le = np.array(le, dtype=float)
            re = np.array(re, dtype=float)
            no = np.array(no, dtype=float)

            # roll from eye line
            eye_vec = re - le
            dx = eye_vec[0]
            dy = eye_vec[1]
            roll = math.degrees(math.atan2(dy, dx))

            mid = (le + re) / 2.0
            nose_vec = no - mid
            ipd = max(1.0, np.linalg.norm(eye_vec))  # interocular distance (pixels)

            # heuristics: scale to degrees — tune multiplier (here 40 deg per ipd)
            yaw = (nose_vec[0] / ipd) * 40.0    # nose right -> negative/positive depending on your convention
            pitch = - (nose_vec[1] / ipd) * 40.0  # nose down -> negative pitch; invert as you need

            # clamp
            yaw = max(-180.0, min(180.0, yaw))
            pitch = max(-90.0, min(90.0, pitch))
            roll = (roll + 180.0) % 360.0 - 180.0

            sin_b, cos_minor = _find_rotation_coeffs(le, re, no)

            return {
                'yaw': float(yaw),
                'pitch': float(pitch),
                'roll': float(roll),
                'sin_b': float(sin_b),
                'cos_minor': float(cos_minor),
                'method': 'geom'
            }

        # If user explicitly asks for 'coeffs' -> return sin_b, cos_minor (C++ behaviour)
        if mode == 'coeffs':
            sin_b, cos_minor = _find_rotation_coeffs(left_eye, right_eye, nose)
            if sin_b == -8.0:
                return {'error': 'invalid geometry'}
            return {'sin_b': sin_b, 'cos_minor': cos_minor}

        # If user asked explicit geometric estimation
        if mode == 'geom':
            return _geom_estimate(left_eye, right_eye, nose)

        # Default: try solvePnP (but be robust and fallback to geom on any failure)
        # prepare image points and camera matrix
        image_points = np.array([left_eye, right_eye, nose], dtype=np.float64)
        model_points = getattr(self, 'model_points', None)
        if model_points is None:
            # if no 3D model defined, fallback to geom
            return _geom_estimate(left_eye, right_eye, nose)

        focal_length = float(w) if w else 1.0
        center = (w / 2.0 if w else 0.0, h / 2.0 if h else 0.0)
        K = np.array([[focal_length, 0.0, center[0]],
                    [0.0, focal_length, center[1]],
                    [0.0, 0.0, 1.0]], dtype=np.float64)
        dist = np.zeros((4,1), dtype=np.float64)

        try:
            # Try solvePnP. In some environments OpenCV expects different shapes/dtypes, so use the common (N,3) model and (N,2) image points
            success, rvec, tvec = cv2.solvePnP(model_points.astype(np.float64),
                                                image_points.astype(np.float64),
                                                K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
            if not success:
                raise RuntimeError("solvePnP returned False")

            # convert to rotation matrix and Euler
            R, _ = cv2.Rodrigues(rvec)

            # rotation_matrix_to_euler must exist in module (implement as you had)
            angles_rad = rotation_matrix_to_euler(R)  # returns [pitch_rad, yaw_rad, roll_rad] per your convention
            pitch = math.degrees(angles_rad[0])
            yaw = math.degrees(angles_rad[1])
            roll = math.degrees(angles_rad[2])

            # normalize to [-180,180]
            def _norm(a):
                a = (a + 180.0) % 360.0 - 180.0
                if abs(a) < 1e-6:
                    a = 0.0
                return float(a)

            return {
                'yaw': _norm(yaw),
                'pitch': _norm(pitch),
                'roll': _norm(roll),
                'rvec': rvec,
                'tvec': tvec,
                'K': K,
                'dist': dist,
                'method': 'pnp'
            }

        except Exception as e:
            # On any failure, fallback to geometric estimate — do not raise
            geom = _geom_estimate(left_eye, right_eye, nose)
            geom['error'] = f'solvePnP failed: {e}'
            return geom

