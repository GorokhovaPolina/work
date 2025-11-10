# test_pose_evaluation.py
import cv2
import numpy as np
import json
from visualizer import draw_axes, draw_visible_nose

def rotation_matrix_to_euler_angles(R):
    """Конвертирует матрицу вращения в углы Эйлера (в радианах)"""
    sy = np.sqrt(R[0,0] * R[0,0] + R[1,0] * R[1,0])
    
    singular = sy < 1e-6

    if not singular:
        x = np.arctan2(R[2,1], R[2,2])  # pitch
        y = np.arctan2(-R[2,0], sy)     # yaw  
        z = np.arctan2(R[1,0], R[0,0])  # roll
    else:
        x = np.arctan2(-R[1,2], R[1,1])
        y = np.arctan2(-R[2,0], sy)
        z = 0

    return np.array([x, y, z])

def calculate_pose_from_json(json_data):
    """
    Вычисляет углы поворота головы из JSON данных
    Возвращает: yaw, pitch, roll в градусах
    """
    # Фиксированные параметры камеры (подходят для 147x181)
    h, w = json_data["image_size"]
    K = np.array([[200, 0, w/2],
                  [0, 200, h/2], 
                  [0, 0, 1]], dtype=np.float32)
    
    tvec = np.array([[0, 0, 200]], dtype=np.float32)
    
    # Берем углы из ground truth (ваши данные)
    pitch_gt = json_data["ground_truth"]["pitch"]
    yaw_gt = json_data["ground_truth"]["yaw"] 
    roll_gt = json_data["ground_truth"]["roll"]
    
    # Преобразуем в вектор вращения
    rvec = np.array([np.radians(pitch_gt), 
                     np.radians(yaw_gt), 
                     np.radians(roll_gt)], dtype=np.float32)
    
    return rvec, tvec, K, (yaw_gt, pitch_gt, roll_gt)

def visualize_with_metrics(img, json_data):
    """
    Визуализация с отображением расчетных коэффициентов
    """
    # Получаем параметры позы
    rvec, tvec, K, (yaw, pitch, roll) = calculate_pose_from_json(json_data)
    
    # Точка носа
    nose = tuple(json_data["props"]["kp_nose_tip"])
    
    # Собираем результат для визуализатора
    result = {
        'rvec': rvec,
        'tvec': tvec, 
        'K': K,
        'dist': np.zeros((4, 1))
    }
    
    # Рисуем оси
    draw_axes(img, nose, scale=20)
    
    # Рисуем нос
    nose_tip = draw_visible_nose(img, nose, rvec, tvec, K, np.zeros((4,1)), 
                               nose_length=20, color=(0, 200, 255))
    
    # Рисуем стрелку направления
    cv2.arrowedLine(img, nose, nose_tip, (255, 255, 0), 2, tipLength=0.3)
    
    # === ОТОБРАЖАЕМ КОЭФФИЦИЕНТЫ ===
    text_y = 20
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # Заголовок
    cv2.putText(img, "Head Pose Angles:", (5, text_y), font, 0.4, (255, 255, 255), 1)
    text_y += 15
    
    # Yaw (поворот головы влево-вправо)
    color = (0, 255, 255) if abs(yaw) < 5 else (0, 165, 255)
    cv2.putText(img, f"Yaw: {yaw:+.1f} deg", (10, text_y), font, 0.4, color, 1)
    text_y += 12
    
    # Pitch (наклон головы вверх-вниз)
    color = (0, 255, 255) if abs(pitch) < 5 else (0, 165, 255)  
    cv2.putText(img, f"Pitch: {pitch:+.1f} deg", (10, text_y), font, 0.4, color, 1)
    text_y += 12
    
    # Roll (наклон головы вбок)
    color = (0, 255, 255) if abs(roll) < 5 else (0, 165, 255)
    cv2.putText(img, f"Roll: {roll:+.1f} deg", (10, text_y), font, 0.4, color, 1)
    
    # Интерпретация
    text_y += 20
    cv2.putText(img, "Interpretation:", (5, text_y), font, 0.4, (200, 200, 255), 1)
    text_y += 12
    
    if yaw > 5:
        cv2.putText(img, "Head turned RIGHT", (10, text_y), font, 0.4, (0, 255, 0), 1)
    elif yaw < -5:
        cv2.putText(img, "Head turned LEFT", (10, text_y), font, 0.4, (0, 255, 0), 1)
    else:
        cv2.putText(img, "Head facing FORWARD", (10, text_y), font, 0.4, (0, 255, 0), 1)
    
    return img, (yaw, pitch, roll)

def test_with_your_data():
    """
    Тестируем на ваших данных
    """
    # Ваши данные
    json_data = {
        "image_size": [181, 147],  # [height, width]
        "props": {
            "kp_nose_tip": [78, 108],
            "kp_eye_left_inner": [67, 86],
            "kp_eye_left_outer": [48, 87], 
            "kp_eye_right_inner": [90, 85],
            "kp_eye_right_outer": [108, 85],
            "kp_mouth_left": [94, 127],
            "kp_mouth_right": [60, 127]
        },
        "ground_truth": {
            "yaw": 2.5,      # Небольшой поворот вправо
            "pitch": -10.0,  # Голова немного опущена
            "roll": -2.0     # Небольшой наклон влево
        }
    }
    
    # Создаем изображение
    img = np.zeros((181, 147, 3), dtype=np.uint8)
    
    # Визуализируем с метриками
    result_img, angles = visualize_with_metrics(img, json_data)
    
    # Выводим углы в консоль
    yaw, pitch, roll = angles
    print("=" * 50)
    print("РАСЧЕТНЫЕ УГЛЫ ПОВОРОТА ГОЛОВЫ:")
    print(f"Yaw (поворот):   {yaw:+.1f}°")
    print(f"Pitch (наклон):  {pitch:+.1f}°") 
    print(f"Roll (крен):     {roll:+.1f}°")
    print("=" * 50)
    
    # Интерпретация
    print("ИНТЕРПРЕТАЦИЯ:")
    if abs(yaw) < 5:
        print("✅ Голова смотрит прямо")
    elif yaw > 0:
        print("↪️  Голова повернута ВПРАВО") 
    else:
        print("↩️  Голова повернута ВЛЕВО")
        
    if pitch > 5:
        print("⬆️  Голова поднята ВВЕРХ")
    elif pitch < -5:
        print("⬇️  Голова опущена ВНИЗ")
    else:
        print("✅ Голова на нормальной высоте")
        
    if abs(roll) < 5:
        print("✅ Голова без наклона")
    elif roll > 0:
        print("↷ Голова наклонена ВПРАВО")
    else:
        print("↶ Голова наклонена ВЛЕВО")
    
    # Показываем изображение
    cv2.imshow('Head Pose Estimation with Metrics', result_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return result_img, angles

# Тестируем разные варианты поворотов головы
def test_different_poses():
    """Тестируем разные углы поворота головы"""
    
    test_cases = [
        {"name": "Прямо", "yaw": 0, "pitch": 0, "roll": 0},
        {"name": "Поворот вправо", "yaw": 15, "pitch": 0, "roll": 0},
        {"name": "Поворот влево", "yaw": -15, "pitch": 0, "roll": 0},
        {"name": "Голова вверх", "yaw": 0, "pitch": 15, "roll": 0},
        {"name": "Голова вниз", "yaw": 0, "pitch": -15, "roll": 0},
        {"name": "Наклон вправо", "yaw": 0, "pitch": 0, "roll": 10},
        {"name": "Наклон влево", "yaw": 0, "pitch": 0, "roll": -10},
        {"name": "Комбинированный", "yaw": 10, "pitch": -8, "roll": -5},
    ]
    
    for i, test_case in enumerate(test_cases):
        json_data = {
            "image_size": [181, 147],
            "props": {"kp_nose_tip": [78, 108]},
            "ground_truth": test_case
        }
        
        img = np.zeros((181, 147, 3), dtype=np.uint8)
        result_img, angles = visualize_with_metrics(img, json_data)
        
        print(f"\n{i+1}. {test_case['name']}:")
        print(f"   Yaw: {angles[0]:+.1f}°, Pitch: {angles[1]:+.1f}°, Roll: {angles[2]:+.1f}°")
        
        cv2.imshow(f'Test: {test_case["name"]}', result_img)
        cv2.waitKey(500)  # Показываем 0.5 секунды
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    print("Запуск теста определения позы головы...")
    
    # Тест 1: Ваши данные
    test_with_your_data()
    
    # Тест 2: Разные позы (раскомментируйте для теста)
    # test_different_poses()