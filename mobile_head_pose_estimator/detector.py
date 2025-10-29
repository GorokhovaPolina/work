import cv2
import mediapipe as mp
import numpy as np

class FastLandmarkDetector:
    """
    Оптимизированный детектор 4 ключевых точек с MediaPipe
    """
    def __init__(self):
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(min_detection_confidence=0.7)

    def detect(self, image):
        """
        Детектирует 4 ключевые точки: нос, левый глаз, правый глаз, рот
        """
        if image is None or image.size == 0:
            return None
            
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(image_rgb)
        
        if not results.detections:
            return None
        
        # Первое обнаруженное лицо
        detection = results.detections[0]
        h, w, _ = image.shape
        keypoints = detection.location_data.relative_keypoints
        
        return {
            'nose': (int(keypoints[2].x * w), int(keypoints[2].y * h)),
            'left_eye': (int(keypoints[1].x * w), int(keypoints[1].y * h)),
            'right_eye': (int(keypoints[0].x * w), int(keypoints[0].y * h)),
            'mouth': (int(keypoints[3].x * w), int(keypoints[3].y * h))
        }
    
    def _preprocess_image(self, image):
        """MediaPipe не требует сильной предобработки 🥰"""
        return image