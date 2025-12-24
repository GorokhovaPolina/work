import scipy.io
import numpy as np
import cv2
import os
import json
import glob
from pathlib import Path
from tqdm import tqdm

def extract_landmarks_from_mat(mat_path, image_path=None):
    try:
        data = scipy.io.loadmat(mat_path)
        
        # 3D точки (68 точек)
        pt3d = data['pt3d_68']  # [3, 68]
        
        # Углы поворота
        pose = data['Pose_Para'][0]
        if image_path and os.path.exists(image_path):
            img = cv2.imread(image_path)
            h, w = img.shape[:2]
        else:
            w, h = 800, 800
        # Индексы ключевых точек
        indices = {
            'nose_tip': 33,           # Кончик носа
            'left_eye_inner': 39,     # Внутренний угол левого глаза
            'left_eye_outer': 36,     # Внешний угол левого глаза  
            'right_eye_inner': 42,    # Внутренний угол правого глаза
            'right_eye_outer': 45,    # Внешний угол правого глаза
            'mouth_left': 48,         # Левый угол рта
            'mouth_right': 54,        # Правый угол рта
            'chin': 8,                # Подбородок (для масштабирования)
            'left_eyebrow': 17,       # Левый бровь
            'right_eyebrow': 26       # Правый бровь
        }
        
        keypoints_2d = {}
        for name, idx in indices.items():
            # Берем X, Y координаты из 3D точек
            x = float(pt3d[0, idx])
            y = float(pt3d[1, idx])
            
            # Нормализуем к координатам изображения
            # Это приближение - для точности нужна калибровка камеры
            x_img = (x + 1) * w / 2
            y_img = (y + 1) * h / 2
            
            keypoints_2d[name] = [x_img, y_img]

        left_eye = [
            (keypoints_2d['left_eye_inner'][0] + keypoints_2d['left_eye_outer'][0]) / 2,
            (keypoints_2d['left_eye_inner'][1] + keypoints_2d['left_eye_outer'][1]) / 2
        ]
        
        right_eye = [
            (keypoints_2d['right_eye_inner'][0] + keypoints_2d['right_eye_outer'][0]) / 2,
            (keypoints_2d['right_eye_inner'][1] + keypoints_2d['right_eye_outer'][1]) / 2
        ]
        
        mouth = [
            (keypoints_2d['mouth_left'][0] + keypoints_2d['mouth_right'][0]) / 2,
            (keypoints_2d['mouth_left'][1] + keypoints_2d['mouth_right'][1]) / 2
        ]
        
        result = {
            "image_size": [w, h],
            "pose_angles": {
                "yaw": float(pose[0]),
                "pitch": float(pose[1]), 
                "roll": float(pose[2])
            },
            "props": {
                "kp_nose_tip": keypoints_2d['nose_tip'],
                "kp_eye_left_inner": keypoints_2d['left_eye_inner'],
                "kp_eye_left_outer": keypoints_2d['left_eye_outer'],
                "kp_eye_right_inner": keypoints_2d['right_eye_inner'],
                "kp_eye_right_outer": keypoints_2d['right_eye_outer'],
                "kp_mouth_left": keypoints_2d['mouth_left'],
                "kp_mouth_right": keypoints_2d['mouth_right'],
                "kp_chin": keypoints_2d['chin']
            },
            "landmarks": {
                "nose": keypoints_2d['nose_tip'],
                "left_eye": left_eye,
                "right_eye": right_eye,
                "mouth": mouth,
                "chin": keypoints_2d['chin']
            }
        }
        
        return result
        
    except Exception as e:
        print(f"Ошибка при обработке {mat_path}: {e}")
        return None

def convert_aflw2000_to_json(image_dir, mat_dir, output_json_dir, output_image_dir=None):
    """
    Конвертирует весь датасет AFLW2000-3D в ваш формат JSON
    
    Аргументы:
        image_dir: папка с изображениями
        mat_dir: папка с .mat файлами
        output_json_dir: куда сохранять JSON
        output_image_dir: куда сохранять изображения с разметкой (опционально)
    """
    os.makedirs(output_json_dir, exist_ok=True)
    if output_image_dir:
        os.makedirs(output_image_dir, exist_ok=True)
    
    # Получаем список изображений
    image_files = sorted(glob.glob(os.path.join(image_dir, "*.jpg")))
    
    successful = 0
    failed = 0
    
    for img_path in tqdm(image_files, desc="Обработка AFLW2000-3D"):
        img_name = os.path.basename(img_path)
        base_name = os.path.splitext(img_name)[0]
        
        # Соответствующий .mat файл
        mat_path = os.path.join(mat_dir, base_name + ".mat")
        
        if not os.path.exists(mat_path):
            print(f"Нет .mat файла для {img_name}")
            failed += 1
            continue
        
        # Извлекаем данные
        data = extract_landmarks_from_mat(mat_path, img_path)
        
        if data is None:
            failed += 1
            continue
        
        # Сохраняем JSON
        json_path = os.path.join(output_json_dir, base_name + ".json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        # Сохраняем визуализацию (опционально)
        if output_image_dir:
            img = cv2.imread(img_path)
            if img is not None:
                # Рисуем ключевые точки
                colors = {
                    'nose': (0, 0, 255),      # Красный
                    'eyes': (0, 255, 0),      # Зеленый  
                    'mouth': (255, 0, 0),     # Синий
                    'chin': (255, 255, 0)     # Голубой
                }
                
                # Нос
                nose = tuple(map(int, data['props']['kp_nose_tip']))
                cv2.circle(img, nose, 5, colors['nose'], -1)
                
                # Глаза
                for eye in ['left_eye_inner', 'left_eye_outer', 
                           'right_eye_inner', 'right_eye_outer']:
                    point = tuple(map(int, data['props'][f'kp_{eye}']))
                    cv2.circle(img, point, 3, colors['eyes'], -1)
                
                # Рот
                mouth_left = tuple(map(int, data['props']['kp_mouth_left']))
                mouth_right = tuple(map(int, data['props']['kp_mouth_right']))
                cv2.circle(img, mouth_left, 3, colors['mouth'], -1)
                cv2.circle(img, mouth_right, 3, colors['mouth'], -1)
                
                # Подбородок
                chin = tuple(map(int, data['props']['kp_chin']))
                cv2.circle(img, chin, 3, colors['chin'], -1)
                
                # Подписываем углы
                angles = data['pose_angles']
                text = f"Yaw: {angles['yaw']:+.1f}° | Pitch: {angles['pitch']:+.1f}° | Roll: {angles['roll']:+.1f}°"
                cv2.putText(img, text, (10, 30), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                output_path = os.path.join(output_image_dir, img_name)
                cv2.imwrite(output_path, img)
        
        successful += 1
    
    print(f"\nКонвертация завершена:")
    print(f"  Успешно: {successful}")
    print(f"  Неудачно: {failed}")
    print(f"  JSON сохранены в: {output_json_dir}")
    
    return successful, failed

def create_annotation_summary(json_dir, output_file="aflw2000_annotations_summary.json"):
    """
    Создает сводный файл со всеми аннотациями
    """
    json_files = glob.glob(os.path.join(json_dir, "*.json"))
    
    summary = {}
    
    for json_path in tqdm(json_files, desc="Создание сводки"):
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        img_name = os.path.basename(json_path).replace('.json', '.jpg')
        
        summary[img_name] = {
            'image_size': data['image_size'],
            'ground_truth': data['pose_angles'],
            'landmarks': data['landmarks']
        }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"Сводка сохранена в: {output_file}")
    return summary

if __name__ == "__main__":
    IMAGE_DIR = "AFLW2000"
    MAT_DIR = "AFLW2000"
    
    OUTPUT_JSON_DIR = "aflw2000_json"
    OUTPUT_IMAGE_DIR = "aflw2000_visualized"
    
    print("Шаг 1: Конвертация AFLW2000-3D в JSON...")
    successful, failed = convert_aflw2000_to_json(
        IMAGE_DIR, MAT_DIR, 
        OUTPUT_JSON_DIR, OUTPUT_IMAGE_DIR
    )
    print("\nШаг 2: Создание сводной аннотации...")
    summary = create_annotation_summary(OUTPUT_JSON_DIR)