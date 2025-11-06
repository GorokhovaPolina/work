from pose_calculator import HeadPoseCalculator
from json_parser import load_keypoints

class Estimator:
    def __init__(self, mode: str = 'pnp'):
        self.calc = HeadPoseCalculator()
        self.mode = mode

    def process(self, path: str):
        data = load_keypoints(path)
        if not data:
            return None
        return self.calc.calculate(data, mode=self.mode)