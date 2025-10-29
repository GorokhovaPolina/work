import cv2
import numpy as np
from sklearn.linear_model import RANSACRegressor
from utils import kalman_filter_update  # Из utils.py

class AffineMotionTracker:
    """
    Трекер на основе аффинного преобразования
    """
    def __init__(self):
        self.affine_matrix = None
        self.tracking_points = None
        self.confidence = 1.0
        self.prev_frame = None
        self.kalman_states = {}  # Для каждого угла: yaw, pitch, roll
        
    def initialize(self, initial_landmarks):
        """
        Инициализация трекера на основе начальной позы
        """
        self.tracking_points = np.array([list(pt) for pt in initial_landmarks.values()], dtype=np.float32)
        self.affine_matrix = np.eye(3)  # Identity
        self.confidence = 1.0
        # Инициализация Kalman для углов (пока placeholder, после calculate_pose)
        self.kalman_states = {'yaw': np.array([0, 0]), 'pitch': np.array([0, 0]), 'roll': np.array([0, 0])}
    
    def track(self, frame):
        """
        Трекинг позы в новом кадре
        """
        if self.tracking_points is None or self.prev_frame is None:
            return None
        
        gray_prev = cv2.cvtColor(self.prev_frame, cv2.COLOR_BGR2GRAY)
        gray_curr = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Расчет оптического потока (LK method)
        curr_points, status, err = cv2.calcOpticalFlowPyrLK(gray_prev, gray_curr, self.tracking_points, None)
        
        # Фильтр хороших точек
        good_points = status.flatten() == 1
        if np.sum(good_points) < 3:
            self.confidence = 0.0
            return None
        
        prev_good = self.tracking_points[good_points]
        curr_good = curr_points[good_points]
        
        # Расчет аффинного преобразования с RANSAC
        affine_matrix, inliers = self._calculate_affine_transform(prev_good, curr_good)
        self.affine_matrix = affine_matrix
        
        # Обновление точек
        self.tracking_points = curr_points
        
        # Обновление confidence
        self.confidence *= len(inliers) / len(prev_good) if len(prev_good) > 0 else 0.5
        
        self.prev_frame = frame
        
        # Возврат transformed landmarks (применяем affine к исходным)
        transformed_points = cv2.transform(np.array([self.tracking_points]), affine_matrix)[0]
        keys = ['nose', 'left_eye', 'right_eye', 'mouth']
        return {keys[i]: tuple(transformed_points[i]) for i in range(len(keys))}
    
    def _calculate_affine_transform(self, prev_points, curr_points):
        """
        Расчет аффинного преобразования между наборами точек
        """
        # Использование cv2 с RANSAC
        affine_matrix, inliers = cv2.estimateAffinePartial2D(prev_points, curr_points, method=cv2.RANSAC)
        return affine_matrix, inliers