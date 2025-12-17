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
                raise ValueError('invalid geom: nose coincides with eye mid point')
                # return -8.0, 0.0
            sin_b = float(v[1] / norm)      # image y grows down => nose below eyes => positive
            sin_b = max(-1.0, min(1.0, sin_b))
            cos_minor = float(math.sqrt(max(0.0, 1.0 - sin_b * sin_b)))
            return sin_b, cos_minor

        # Helper: pure geometric estimation (2D)
        def _geom_estimate1(le, re, no):
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
            ipd = max(1.0, np.linalg.norm(eye_vec))
            
            # Нормализуем вертикальное смещение
            vertical_ratio = nose_vec[1] / ipd
            
            # Эмпирическая калибровка: 
            # vertical_ratio ~ 0.1-0.3 для прямого взгляда (зависит от анатомии)
            # Вычитаем базовое смещение для прямого взгляда
            baseline_vertical = 0.45  # variable!
            calibrated_vertical = vertical_ratio - baseline_vertical
            
            pitch = -calibrated_vertical * 80.0  # инвертируем и масштабируем
            yaw = (nose_vec[0] / ipd) * 60.0

            # clamp angles
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

        def _geom_estimate(le, re, no, lm):
            le = np.array(le, dtype=float)
            re = np.array(re, dtype=float)
            no = np.array(no, dtype=float)
    
            # roll из линии глаз
            eye_vec = re - le
            dx = eye_vec[0]
            dy = eye_vec[1]
            roll = math.degrees(math.atan2(dy, dx))
    
            mid = (le + re) / 2.0
            nose_vec = no - mid
            ipd = max(1.0, np.linalg.norm(eye_vec))





            mouth_left1 = tuple(lm.get('kp_mouth_left')) if 'kp_mouth_left' in lm else None
            mouth_right1 = tuple(lm.get('kp_mouth_right')) if 'kp_mouth_right' in lm else None
            # НОВЫЙ МЕТОД: используем пропорцию "нос:рот = 1:3"
            if True:
                mouth_left = np.array(mouth_left1, dtype=float)
                mouth_right = np.array(mouth_right1, dtype=float)
                mouth_center = (mouth_left + mouth_right) / 2.0
        
                # Длина носа (от средней точки глаз до кончика носа)
                nose_length = np.linalg.norm(no - mid)
        
                # Расстояние от носа до центра рта
                nose_to_mouth = np.linalg.norm(mouth_center - no)
        
                if nose_to_mouth < 0:
                    # Идеальное соотношение: нос = 3 × (нос->рот)
                    ideal_ratio = 3.0
            
                    # Текущее соотношение
                    current_ratio = nose_length / nose_to_mouth
            
                    # Отклонение от идеальной пропорции
                    # Если нос слишком длинный относительно расстояния до рта -> лицо задирается (pitch > 0)
                    # Если слишком короткий -> опускается (pitch < 0)
                    ratio_deviation = current_ratio - ideal_ratio
            
                    # Преобразуем отклонение в угол
                    # Эмпирический коэффициент: отклонение на 0.1 дает примерно 10 градусов
                    pitch = -ratio_deviation * 100.0  
            
                    # Также учитываем абсолютное вертикальное смещение как дополнительный фактор
                    # Нормализуем вертикальное смещение носа относительно IPD для тонкой настройки
                    vertical_ratio = nose_vec[1] / ipd
                    baseline_vertical = 0.45
                    pitch_adjustment = -(vertical_ratio - baseline_vertical) * 30.0
            
                    pitch = pitch * 0.7 + pitch_adjustment * 0.3  # взвешенное среднее
            
                else:
                    # fallback
                    vertical_ratio = nose_vec[1] / ipd
                    baseline_vertical = 0.6
                    calibrated_vertical = vertical_ratio - baseline_vertical
                    pitch = -calibrated_vertical * 80.0
            else:
                # fallback если нет точек рта
                vertical_ratio = nose_vec[1] / ipd
                baseline_vertical = 0.6
                calibrated_vertical = vertical_ratio - baseline_vertical
                pitch = -calibrated_vertical * 80.0
    
            # Горизонтальный yaw остается прежним
            yaw = (nose_vec[0] / ipd) * 60.0
    
            # clamp angles
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
                'method': 'geom',
                'nose_length': float(nose_length) if 'nose_length' in locals() else 0.0,
                'nose_to_mouth': float(nose_to_mouth) if 'nose_to_mouth' in locals() else 0.0,
                'ratio': float(current_ratio) if 'current_ratio' in locals() else 0.0
            }
        # If user explicitly asks for 'coeffs' -> return sin_b, cos_minor
        if mode == 'coeffs':
            try: 
                sin_b, cos_minor = _find_rotation_coeffs(left_eye, right_eye, nose)
                return {'sin_b': sin_b, 'cos_minor': cos_minor}
            except ValueError as e: 
                return {'error': str(e)}

        # If user asked explicit geometric estimation
        if mode == 'geom':
            return _geom_estimate(left_eye, right_eye, nose, lm)

        image_points = np.array([left_eye, right_eye, nose], dtype=np.float64)
        model_points = getattr(self, 'model_points', None)
        if model_points is None:
            # if no 3D model defined, fallback to geom
            return _geom_estimate(left_eye, right_eye, nose, lm)

        focal_length = float(w) if w else 1.0
        center = (w / 2.0 if w else 0.0, h / 2.0 if h else 0.0)
        K = np.array([[focal_length, 0.0, center[0]],
                    [0.0, focal_length, center[1]],
                    [0.0, 0.0, 1.0]], dtype=np.float64)
        dist = np.zeros((4,1), dtype=np.float64)

        try:
            success, rvec, tvec = cv2.solvePnP(model_points.astype(np.float64),
                                                image_points.astype(np.float64),
                                                K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
            if not success:
                raise RuntimeError("solvePnP returned False")

            R, _ = cv2.Rodrigues(rvec)

            angles_rad = rotation_matrix_to_euler(R)  # returns [pitch_rad, yaw_rad, roll_rad] per convention
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
            geom = _geom_estimate(left_eye, right_eye, nose, lm)
            geom['error'] = f'solvePnP failed: {e}'
            return geom

