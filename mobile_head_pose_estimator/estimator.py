from pose_calculator import GeometricPoseCalculator
from json_parser import load_keypoints_from_json

class MobileHeadPoseEstimator:
    def __init__(self, mode='pnp'):
        self.calculator = GeometricPoseCalculator()
        self.mode = mode

    def process_json(self, json_path):
        data = load_keypoints_from_json(json_path)
        if not data:
            return None
        return self.calculator.calculate_pose(data, mode=self.mode)
