import json
import numpy as np
import cv2
from json_parser import load_keypoints_from_json
from pnp_estimator import PnPEstimator
from visualizer import visualize

def test_cone_visualization():
    data = load_keypoints_from_json("tests/mock_data.json")
    estimator = PnPEstimator()
    result = estimator.estimate(data['landmarks'], data['image_size'])
    
    img = np.zeros((181, 147, 3), dtype=np.uint8)  # чёрный фон
    nose = tuple(map(int, data['landmarks']['nose']))
    
    visualize(img, nose, result)
    cv2.imwrite("tests/cone_test.jpg", img)
    print("Конус сохранён: tests/cone_test.jpg")

if __name__ == "__main__":
    test_cone_visualization()