from pose_calculator import GeometricPoseCalculator
from json_parser import load_keypoints_from_json

class MobileHeadPoseEstimator:
    def __init__(self):
        self.pose_calc = GeometricPoseCalculator()

    def process_json(self, json_path: str):
        """Принимает путь к JSON → возвращает pose или None"""
        landmarks = load_keypoints_from_json(json_path)
        if landmarks is None:
            return None
        return self.pose_calc.calculate_pose(landmarks)
