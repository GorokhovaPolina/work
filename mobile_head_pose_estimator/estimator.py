import numpy as np
from detector import FastLandmarkDetector
from tracker import AffineMotionTracker
from pose_calculator import GeometricPoseCalculator
from config import MobileConfig
from utils import kalman_filter_update, euler_to_rot_matrix

class MobileHeadPoseEstimator:
    """
    Основной класс для оценки позы головы на мобильных устройствах
    """
    def __init__(self):
        self.detector = FastLandmarkDetector()
        self.tracker = AffineMotionTracker()
        self.pose_calculator = GeometricPoseCalculator()
        self.config = MobileConfig()
        self.frame_count = 0
        self.prev_pose = None
        self.kalman_states = {'yaw': np.array([0, 0]), 'pitch': np.array([0, 0]), 'roll': np.array([0, 0])}
        
    def needs_redetection(self):
        return (self.frame_count % self.config.detection_interval == 0) or \
               (self.tracker.confidence < self.config.min_tracking_confidence)
    
    def process_frame(self, frame):
        self.frame_count += 1
        landmarks = None
        
        if self.needs_redetection():
            landmarks = self.detector.detect(frame)
            if landmarks:
                self.tracker.initialize(landmarks)
                self.tracker.prev_frame = frame
        else:
            landmarks = self.tracker.track(frame)
        
        if landmarks is None:
            euler = {'yaw': 0, 'pitch': 0, 'roll': 0}
            matrix = euler_to_rot_matrix(0, 0, 0)
            return {'euler': euler, 'matrix': matrix}
        
        base_pose = self.pose_calculator.calculate_pose(landmarks, mode='geometric')  # Или 'pnp' для точнее, 'coeffs' для C++
        if 'sin_b' in base_pose:  # Для coeffs mode — нет refine/matrix
            return {'coeffs': base_pose}
        
        refined_pose = self.refine_pose(base_pose)
        self.prev_pose = refined_pose
        
        matrix = euler_to_rot_matrix(refined_pose['yaw'], refined_pose['pitch'], refined_pose['roll'])
        return {'euler': refined_pose, 'matrix': matrix}
    
    def refine_pose(self, base_pose):
        for angle in ['yaw', 'pitch', 'roll']:
            self.kalman_states[angle] = kalman_filter_update(
                self.kalman_states[angle], 
                base_pose[angle], 
                Q=self.config.kalman_filter_q, 
                R=self.config.kalman_filter_r
            )
            base_pose[angle] = self.kalman_states[angle][0] * self.config.pose_smoothing_factor + \
                               base_pose[angle] * (1 - self.config.pose_smoothing_factor)
        return base_pose