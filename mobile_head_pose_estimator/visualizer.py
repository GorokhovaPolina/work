import cv2
import numpy as np

def draw_simple_axes(img, nose, scale=25):
    """
    Простые и понятные оси - все вправо и вверх!
    """
    # X - вправо (красный)
    cv2.arrowedLine(img, nose, (nose[0] + scale, nose[1]), (0, 0, 255), 2, tipLength=0.3)
    # Y - вверх (зеленый)  
    cv2.arrowedLine(img, nose, (nose[0], nose[1] - scale), (0, 255, 0), 2, tipLength=0.3)
    # Z - тоже вправо, но выше (синий)
    cv2.arrowedLine(img, nose, (nose[0] + scale, nose[1] - scale), (255, 0, 0), 2, tipLength=0.3)
    
    # Подписи
    cv2.putText(img, 'X', (nose[0] + scale + 5, nose[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
    cv2.putText(img, 'Y', (nose[0], nose[1] - scale - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    cv2.putText(img, 'Z', (nose[0] + scale + 5, nose[1] - scale), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

def draw_simple_cone(img, nose, rvec, tvec, K, dist, cone_length=40, base_radius=6):
    """
    ПРОСТОЙ конус: основание на кончике носа, высота = направлению головы
    """
    segments = 12
    
    # Все точки конуса
    cone_points = []
    
    # Основание - маленький круг ПРЯМО на кончике носа
    for i in range(segments):
        angle = 2 * np.pi * i / segments
        x = base_radius * np.cos(angle)
        y = base_radius * np.sin(angle)
        cone_points.append([x, y, 0])  # Z=0 - прямо в носу!
    
    # Кончик конуса - вперед по направлению головы
    cone_points.append([0, 0, -cone_length])  # Вперед!
    
    # Проекция в 2D
    cone_3d = np.float32(cone_points)
    pts, _ = cv2.projectPoints(cone_3d, rvec, tvec, K, dist)
    pts = np.int32(pts).reshape(-1, 2)
    
    base_pts = pts[:-1]  # Точки основания
    tip_pt = tuple(pts[-1])  # Кончик
    
    # 1. Рисуем основание (маленький круг на носу)
    cv2.circle(img, nose, base_radius, (0, 200, 255), -1)  # Заливка
    cv2.circle(img, nose, base_radius, (0, 150, 200), 2)   # Контур
    
    # 2. Рисуем ВСЕ грани от основания к кончику
    for pt in base_pts:
        cv2.line(img, nose, tip_pt, (0, 200, 255), 2)  # Центральная линия
        cv2.line(img, tuple(pt), tip_pt, (0, 200, 255), 1)  # Боковые грани
    
    # 3. Выделяем кончик
    cv2.circle(img, tip_pt, 4, (255, 100, 0), -1)
    
    return tip_pt

def visualize(img, nose, result):
    if 'rvec' not in result:
        return
    
    # 1. Простые оси (все вправо-вверх)
    draw_simple_axes(img, nose, scale=25)
    
    # 2. Простой конус
    tip_pt = draw_simple_cone(
        img, nose,
        result['rvec'], result['tvec'], result['K'],
        result.get('dist', np.zeros((4, 1))),
        cone_length=35, base_radius=5
    )
    
    # 3. Подпись
    cv2.putText(img, "HEAD DIRECTION", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

# Тестовая функция
def test_simple_visualization(img, json_data):
    """
    Тестируем простую визуализацию
    """
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
    
    nose = tuple(json_data["props"]["kp_nose_tip"])
    
    result = {
        'rvec': rvec,
        'tvec': tvec, 
        'K': K,
        'dist': np.zeros((4, 1))
    }
    
    visualize(img, nose, result)
    return img