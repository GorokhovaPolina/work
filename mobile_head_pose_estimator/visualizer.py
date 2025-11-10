# visualizer.py
import cv2
import numpy as np

def draw_axes(img, nose, scale=50):
    """Рисует красивые оси с подписями"""
    # X — красная
    cv2.line(img, nose, (nose[0] + scale, nose[1]), (0, 0, 255), 3)
    cv2.putText(img, 'X', (nose[0] + scale + 5, nose[1]), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    # Y — зелёная  
    cv2.line(img, nose, (nose[0], nose[1] - scale), (0, 255, 0), 3)
    cv2.putText(img, 'Y', (nose[0], nose[1] - scale - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # Z — синяя
    cv2.line(img, nose, (nose[0] - scale, nose[1]), (255, 0, 0), 3)
    cv2.putText(img, 'Z', (nose[0] - scale - 15, nose[1]), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

def create_gradient_mask(shape, center, radius, inner_color, outer_color):
    """Создает градиентную маску для плавного перехода цвета"""
    mask = np.zeros(shape, dtype=np.uint8)
    y, x = np.ogrid[:shape[0], :shape[1]]
    dist_from_center = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    
    # Нормализуем расстояния
    dist_norm = np.clip(dist_from_center / radius, 0, 1)
    
    # Интерполяция цветов
    for i in range(3):
        mask[..., i] = (inner_color[i] * (1 - dist_norm) + 
                       outer_color[i] * dist_norm).astype(np.uint8)
    
    return mask

def draw_smooth_cone(img, nose, rvec, tvec, K, dist, length=90, radius=30, 
                    color=(0, 255, 255), segments=48, gradient=True):
    """
    Рисует красивый конус с градиентами и эффектами
    """
    # Генерация точек основания
    base_points = []
    for i in range(segments):
        angle = 2 * np.pi * i / segments
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        base_points.append([x, y, length])
    
    # Вершина конуса (нос) + точки основания
    cone_3d = np.float32([[0, 0, 0]] + base_points)
    
    # Проекция в 2D
    pts, _ = cv2.projectPoints(cone_3d, rvec, tvec, K, dist)
    pts = np.int32(pts).reshape(-1, 2)
    
    nose_pt = tuple(pts[0])
    base_pts = pts[1:]
    
    # Создаем маску для рисования
    overlay = img.copy()
    
    if gradient:
        # Градиент для основания
        base_center = np.mean(base_pts, axis=0).astype(int)
        base_radius = int(np.linalg.norm(base_pts[0] - base_center))
        
        # Темно-желтый к светлому
        inner_color = tuple(max(0, c - 80) for c in color)
        
        # Создаем градиентную маску
        gradient_mask = create_gradient_mask(
            img.shape[:2], base_center, base_radius, 
            inner_color, color
        )
        
        # Применяем градиент к области основания
        base_mask = np.zeros(img.shape[:2], dtype=np.uint8)
        cv2.fillPoly(base_mask, [base_pts], 255)
        
        # Накладываем градиент
        gradient_region = cv2.bitwise_and(gradient_mask, gradient_mask, mask=base_mask)
        cv2.addWeighted(gradient_region, 0.7, overlay, 0.3, 0, overlay)
    else:
        # Простая заливка
        cv2.fillPoly(overlay, [base_pts], color)
    
    # Рисуем грани с градиентом толщины
    for i, pt in enumerate(base_pts):
        # Градиент толщины линии от центра
        thickness = max(1, int(3 * (1 - i / len(base_pts))))
        alpha = 0.3 + 0.7 * (i / len(base_pts))  # Градиент прозрачности
        
        line_overlay = overlay.copy()
        cv2.line(line_overlay, nose_pt, tuple(pt), color, thickness)
        cv2.addWeighted(line_overlay, alpha, overlay, 1 - alpha, 0, overlay)
    
    # Контур основания
    cv2.polylines(overlay, [base_pts], isClosed=True, 
                  color=tuple(min(255, c + 30) for c in color), thickness=2)
    
    # Накладываем на оригинальное изображение
    cv2.addWeighted(overlay, 0.8, img, 0.2, 0, img)
    
    # Добавляем свечение вершины
    glow_size = 8
    glow_overlay = img.copy()
    cv2.circle(glow_overlay, nose_pt, glow_size, (255, 255, 200), -1)
    cv2.addWeighted(glow_overlay, 0.3, img, 0.7, 0, img)
    
    # Центральная точка вершины
    cv2.circle(img, nose_pt, 3, (255, 255, 255), -1)

def draw_direction_line(img, nose, rvec, tvec, K, dist, length=120, color=(255, 255, 0)):
    """Рисует линию направления из носа"""
    # Точка в направлении головы
    direction_3d = np.float32([[0, 0, length]])
    pts, _ = cv2.projectPoints(direction_3d, rvec, tvec, K, dist)
    direction_pt = tuple(np.int32(pts[0][0]))
    
    # Рисуем стрелку
    cv2.arrowedLine(img, nose, direction_pt, color, 3, tipLength=0.2)
    
def visualize(img, nose, result):
    """Основная функция визуализации"""
    # Рисуем оси
    draw_axes(img, nose)
    
    if 'rvec' in result:
        # Рисуем направление
        draw_direction_line(
            img, nose,
            result['rvec'], result['tvec'], result['K'], 
            result.get('dist', np.zeros((4,1))),
            length=120, color=(255, 200, 0)
        )
        
        # Рисуем красивый конус
        draw_smooth_cone(
            img, nose,
            result['rvec'], result['tvec'], result['K'], 
            result.get('dist', np.zeros((4,1))),
            length=90, radius=30, color=(0, 255, 255), segments=48, gradient=True
        )
    
    # Добавляем информационный текст
    info_text = f"Head Pose Estimation"
    cv2.putText(img, info_text, (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(img, info_text, (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 100, 255), 1)

# Дополнительная функция для отладки
def debug_visualization(img, nose, result):
    """Расширенная визуализация для отладки"""
    visualize(img, nose, result)
    
    if 'rvec' in result:
        # Добавляем углы поворота
        rmat, _ = cv2.Rodrigues(result['rvec'])
        angles = rotation_matrix_to_euler_angles(rmat)
        
        text_y = 60
        for i, (angle, axis) in enumerate(zip(angles, ['Pitch', 'Yaw', 'Roll'])):
            text = f"{axis}: {np.degrees(angle):.1f}°"
            cv2.putText(img, text, (10, text_y + i*25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

def rotation_matrix_to_euler_angles(R):
    """Конвертирует матрицу вращения в углы Эйлера"""
    sy = np.sqrt(R[0,0] * R[0,0] + R[1,0] * R[1,0])
    
    singular = sy < 1e-6

    if not singular:
        x = np.arctan2(R[2,1], R[2,2])
        y = np.arctan2(-R[2,0], sy)
        z = np.arctan2(R[1,0], R[0,0])
    else:
        x = np.arctan2(-R[1,2], R[1,1])
        y = np.arctan2(-R[2,0], sy)
        z = 0

    return np.array([x, y, z])
