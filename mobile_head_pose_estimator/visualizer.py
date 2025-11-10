# visualizer.py
import cv2
import numpy as np

def draw_minimal_axes(img, nose, rvec, tvec, K, dist, length=25):
    """
    Рисует минималистичные оси, прикрепленные к голове
    """
    # Точки осей в 3D пространстве (относительно носа)
    axis_points = np.float32([
        [length, 0, 0],    # X - красный
        [0, -length, 0],   # Y - зеленый  
        [0, 0, length]     # Z - синий (вперед)
    ])
    
    # Проекция в 2D
    pts, _ = cv2.projectPoints(axis_points, rvec, tvec, K, dist)
    pts = np.int32(pts).reshape(-1, 2)
    
    # Рисуем оси
    cv2.arrowedLine(img, nose, tuple(pts[0]), (0, 0, 255), 2, tipLength=0.2)  # X
    cv2.arrowedLine(img, nose, tuple(pts[1]), (0, 255, 0), 2, tipLength=0.2)  # Y  
    cv2.arrowedLine(img, nose, tuple(pts[2]), (255, 0, 0), 2, tipLength=0.2)  # Z

def draw_elegant_cone(img, nose, rvec, tvec, K, dist, cone_length=40, base_radius=8):
    """
    Рисует элегантный минималистичный конус направления
    """
    segments = 16
    cone_color = (0, 200, 255)  # Красивый голубовато-оранжевый
    
    # Создаем точки конуса: основание + кончик
    cone_points = []
    
    # Основание конуса (маленькое)
    for i in range(segments):
        angle = 2 * np.pi * i / segments
        x = base_radius * np.cos(angle)
        y = base_radius * np.sin(angle)
        cone_points.append([x, y, 0])
    
    # Кончик конуса
    cone_points.append([0, 0, cone_length])
    
    # Проекция в 2D
    cone_3d = np.float32(cone_points)
    pts, _ = cv2.projectPoints(cone_3d, rvec, tvec, K, dist)
    pts = np.int32(pts).reshape(-1, 2)
    
    base_pts = pts[:-1]  # Точки основания
    tip_pt = tuple(pts[-1])  # Кончик
    
    # Создаем overlay для красивого наложения
    overlay = img.copy()
    
    # Рисуем прозрачное основание
    if len(base_pts) > 2:
        cv2.fillPoly(overlay, [base_pts], cone_color)
    
    # Рисуем грани конуса (только несколько для минимализма)
    for i in range(0, segments, 4):  # Каждую 4-ю точку для чистоты
        cv2.line(overlay, tuple(base_pts[i]), tip_pt, cone_color, 1)
    
    # Контур основания
    cv2.polylines(overlay, [base_pts], True, cone_color, 1)
    
    # Центральная линия (самая важная)
    cv2.line(overlay, nose, tip_pt, (255, 100, 0), 2)
    
    # Накладываем с легкой прозрачностью
    cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)
    
    # Красивый кончик
    cv2.circle(img, tip_pt, 3, (255, 150, 0), -1)
    cv2.circle(img, tip_pt, 5, (255, 200, 100), 1)
    
    return tip_pt

def draw_direction_indicator(img, nose, tip_pt):
    """
    Рисует минималистичный индикатор направления
    """
    # Тонкая линия от носа к кончику конуса
    cv2.line(img, nose, tip_pt, (255, 255, 255), 1)
    
    # Точка в начале (носе)
    cv2.circle(img, nose, 3, (255, 255, 255), -1)

def visualize(img, nose, result):
    """
    Модная минималистичная визуализация направления головы
    """
    if 'rvec' not in result:
        return
    
    # 1. Сначала рисуем элегантный конус
    tip_pt = draw_elegant_cone(
        img, nose,
        result['rvec'], result['tvec'], result['K'],
        result.get('dist', np.zeros((4, 1))),
        cone_length=35, base_radius=6
    )
    
    # 2. Минималистичные оси
    draw_minimal_axes(
        img, nose,
        result['rvec'], result['tvec'], result['K'], 
        result.get('dist', np.zeros((4, 1))),
        length=20
    )
    
    # 3. Индикатор направления
    draw_direction_indicator(img, nose, tip_pt)
    
    # 4. Стильная подпись
    cv2.putText(img, "HEAD POSE", (10, 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

# Функция для тестирования с вашим JSON
def test_elegant_visualization(img, json_data):
    """
    Тестируем новую визуализацию с вашими данными
    """
    # Параметры камеры
    h, w = json_data["image_size"]
    K = np.array([[w, 0, w/2],
                  [0, w, h/2], 
                  [0, 0, 1]], dtype=np.float32)
    
    tvec = np.array([[0, 0, 500]], dtype=np.float32)
    
    # Углы из JSON
    pitch = np.radians(json_data["ground_truth"]["pitch"])
    yaw = np.radians(json_data["ground_truth"]["yaw"]) 
    roll = np.radians(json_data["ground_truth"]["roll"])
    
    rvec = np.array([pitch, yaw, roll], dtype=np.float32)
    
    # Точка носа
    nose = tuple(json_data["props"]["kp_nose_tip"])
    
    # Результат
    result = {
        'rvec': rvec,
        'tvec': tvec, 
        'K': K,
        'dist': np.zeros((4, 1))
    }
    
    # Визуализация
    visualize_elegant(img, nose, result)
    
    return img