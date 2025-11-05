import math
import numpy as np
import cv2
from utils import find_rotation_coeffs

class GeometricPoseCalculator:
    """
    Расчет углов поворота головы на основе геометрии 4 точек
    """
    def calculate_pose(self, landmarks, mode='geometric'):
        """
        Расчет pitch, yaw, roll. Mode: 'geometric' (default), 'coeffs' (как C++), 'pnp' (точнее с 3D моделью)
        """
        if landmarks is None or len(landmarks) != 4:
            return None
        
        if mode == 'coeffs':
            left_eye = landmarks['left_eye']
            right_eye = landmarks['right_eye']
            nose = landmarks['nose']
            coeffs = find_rotation_coeffs(left_eye, right_eye, nose)
            return {'sin_b': coeffs[0], 'cos_minor': coeffs[1]}  # Для RotateCheck
        
        if mode == 'pnp':
            # Лучше и точнее: solvePnP с 3D моделью лица (assume standard face model)
            # 3D точки (generic face model in mm)
            model_points = np.array([
                (0.0, 0.0, 0.0),  # Nose
                (-225.0, 170.0, -135.0),  # Left eye
                (225.0, 170.0, -135.0),   # Right eye
                (0.0, -150.0, -125.0)     # Mouth
            ], dtype=np.float32)
            
            # 2D точки из landmarks
            image_points = np.array([
                landmarks['nose'],
                landmarks['left_eye'],
                landmarks['right_eye'],
                landmarks['mouth']
            ], dtype=np.float32)
            
            # Camera matrix (assume focal=1, center= (0,0) for norm; in real — calibrate)
            size = (100, 100)  # Placeholder, adjust to image size
            camera_matrix = np.array(
                [[size[0], 0, size[0]/2],
                 [0, size[0], size[1]/2],
                 [0, 0, 1]], dtype=np.float32
            )
            dist_coeffs = np.zeros((4,1))  # No distortion
            
            # solvePnP
            success, rvec, tvec = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs)
            if not success:
                return None
            
            # Rotation vector to Euler
            rot_mat, _ = cv2.Rodrigues(rvec)
            proj_mat = np.hstack((rot_mat, tvec))
            _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(proj_mat)
            yaw, pitch, roll = euler_angles.flatten()[:3]
            return {'yaw': yaw, 'pitch': pitch, 'roll': roll}
        
        # Default: geometric как раньше
        normalized_coords = self._normalize_coordinates(landmarks)
        
        yaw = self._calculate_yaw(normalized_coords)
        pitch = self._calculate_pitch(normalized_coords) 
        roll = self._calculate_roll(normalized_coords)
        
        return {'yaw': yaw, 'pitch': pitch, 'roll': roll}
    
    def _normalize_coordinates(self, landmarks):
        """
        Нормализация координат относительно расстояния между глазами
        """
        eye_distance = math.dist(landmarks['right_eye'], landmarks['left_eye'])
        if eye_distance == 0:
            eye_distance = 1  # Избежать деления на 0
        
        normalized = {}
        nose = landmarks['nose']
        for key, (x, y) in landmarks.items():
            # Относительно носа
            nx, ny = (x - nose[0]) / eye_distance, (y - nose[1]) / eye_distance
            normalized[key] = (nx, ny)
            
        return normalized
    
    def _calculate_yaw(self, coords):
        """Yaw: горизонтальный поворот (по глазам и рту)"""
        eye_mid = ((coords['left_eye'][0] + coords['right_eye'][0]) / 2, (coords['left_eye'][1] + coords['right_eye'][1]) / 2)
        mouth_vec = (coords['mouth'][0] - eye_mid[0], coords['mouth'][1] - eye_mid[1])
        return math.degrees(math.atan2(mouth_vec[0], mouth_vec[1]))  # Примерная оценка
    
    def _calculate_pitch(self, coords):
        """Pitch: вертикальный наклон (по расстоянию глаз-рот)"""
        eye_mid_y = (coords['left_eye'][1] + coords['right_eye'][1]) / 2
        mouth_y = coords['mouth'][1]
        return math.degrees(math.asin((mouth_y - eye_mid_y)))  # Упрощено
    
    def _calculate_roll(self, coords):
        """Roll: наклон головы (угол между глазами)"""
        dx = coords['right_eye'][0] - coords['left_eye'][0]
        dy = coords['right_eye'][1] - coords['left_eye'][1]
        return math.degrees(math.atan2(dy, dx))