import json
import numpy as np
import cv2
from json_parser import load_keypoints_from_json
from pnp_estimator import PnPEstimator
from visualizer1 import visualize

def test_cone_visualization():
    # === ЗАГРУЗКА ===
    data = load_keypoints_from_json("tests/mock_data.json")
    if data is None:
        print("ОШИБКА: не удалось загрузить mock_data.json")
        return

    # === ОЦЕНКА ===
    estimator = PnPEstimator()
    result = estimator.estimate(data['landmarks'], data['image_size'])
    if result is None:
        print("ОШИБКА: estimator вернул None")
        return

    # === ВИЗУАЛИЗАЦИЯ ===
    h, w = data['image_size']
    img = np.zeros((h, w, 3), dtype=np.uint8)  # чёрный фон
    nose = tuple(map(int, data['landmarks']['nose']))
    
    visualize(img, nose, result)
    
    # === СОХРАНЕНИЕ ===
    output_path = "tests/cone_test.jpg"
    cv2.imwrite(output_path, img)
    print(f"КОНУС СОХРАНЁН: {output_path}")

if __name__ == "__main__":
    test_cone_visualization()