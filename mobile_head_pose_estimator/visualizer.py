import cv2
import numpy as np
import math

def draw_fixed_axes(img, nose, scale=30):
    """
    Фиксированные оси в стандартной системе координат камеры
    Не зависят от поворота головы!
    """
    # Все оси выходят из носа, но направления фиксированы:
    # X - вправо (красный)
    cv2.arrowedLine(img, nose, (nose[0] + scale, nose[1]), (0, 0, 255), 2, tipLength=0.3)
    # Y - вверх (зеленый)  
    cv2.arrowedLine(img, nose, (nose[0], nose[1] - scale), (0, 255, 0), 2, tipLength=0.3)
    # Z - вглубь (синий) - рисуем короче и тоньше, т.к. это "в экран"
    cv2.arrowedLine(img, nose, (nose[0] - scale//2, nose[1]), (255, 0, 0), 1, tipLength=0.2)
    
    # Подписи
    cv2.putText(img, 'X', (nose[0] + scale + 5, nose[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
    cv2.putText(img, 'Y', (nose[0], nose[1] - scale - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    cv2.putText(img, 'Z', (nose[0] - scale//2 - 10, nose[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

def draw_clear_cone(img, nose, rvec, tvec, K, dist, cone_length=50, base_radius=10):
    """
    Четкий конус, идущий от носа в направлении головы
    """
    segments = 16
    cone_color = (0, 200, 255)  # Яркий оранжево-голубой
    
    # Точки конуса: основание в носу + кончик вперед
    cone_points = []
    
    # Основание (круг прямо на кончике носа)
    for i in range(segments):
        angle = 2 * np.pi * i / segments
        x = base_radius * np.cos(angle)
        y = base_radius * np.sin(angle)
        cone_points.append([x, y, 0])  # Z=0 - прямо в носу!
    
    # Кончик конуса (впереди по направлению головы)
    cone_points.append([0, 0, -cone_length])  # Отрицательный Z - вперед из экрана!
    
    # Проекция
    cone_3d = np.float32(cone_points)
    pts, _ = cv2.projectPoints(cone_3d, rvec, tvec, K, dist)
    pts = np.int32(pts).reshape(-1, 2)
    
    base_pts = pts[:-1]
    tip_pt = tuple(pts[-1])
    
    # Рисуем
    overlay = img.copy()
    
    # 1. Основание (полупрозрачное)
    if len(base_pts) > 2:
        cv2.fillPoly(overlay, [base_pts], cone_color)
    
    # 2. Грани конуса
    for i in range(segments):
        cv2.line(overlay, tuple(base_pts[i]), tip_pt, cone_color, 2)
    
    # 3. Контур основания
    cv2.polylines(overlay, [base_pts], True, (0, 150, 200), 2)
    
    # 4. Центральная линия
    cv2.line(overlay, nose, tip_pt, (255, 100, 0), 3)
    
    cv2.addWeighted(overlay, 0.8, img, 0.2, 0, img)
    
    # Выделяем точки
    cv2.circle(img, tip_pt, 4, (255, 150, 0), -1)
    cv2.circle(img, nose, 3, (255, 255, 255), -1)
    
    return tip_pt

def visualize_fixed(img, nose, result):
    """
    Визуализация с фиксированными осями
    """
    if 'rvec' not in result:
        return
    
    # 1. Фиксированные оси (всегда одинаковые!)
    draw_fixed_axes(img, nose, scale=25)
    
    # 2. Конус направления
    tip_pt = draw_clear_cone(
        img, nose,
        result['rvec'], result['tvec'], result['K'],
        result.get('dist', np.zeros((4, 1))),
        cone_length=40, base_radius=8
    )
    
    # 3. Информация
    cv2.putText(img, "HEAD POSE", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)