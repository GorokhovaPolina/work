# visualizer.py
import cv2
import numpy as np

def draw_3d_axes(img, nose, rvec, tvec, K, dist, scale=50):
    """Рисует 3D оси с учетом перспективы и поворота"""
    # 3D точки для осей в системе координат головы
    axis_points = np.float32([
        [scale, 0, 0],   # X - красный
        [0, -scale, 0],  # Y - зеленый (отрицательный, т.к. в изображении Y вниз)
        [0, 0, scale],   # Z - синий
        [0, 0, 0]        # начало координат (нос)
    ])
    
    # Проецируем 3D точки в 2D
    pts, _ = cv2.projectPoints(axis_points, rvec, tvec, K, dist)
    pts = np.int32(pts).reshape(-1, 2)
    
    origin = tuple(pts[3])
    x_axis = tuple(pts[0])
    y_axis = tuple(pts[1])
    z_axis = tuple(pts[2])
    
    # Рисуем оси с 3D эффектом (более толстые у основания)
    # X ось - красная
    cv2.line(img, origin, x_axis, (0, 0, 255), 4)
    cv2.line(img, origin, x_axis, (50, 50, 255), 2)  # внутренняя линия для объема
    
    # Y ось - зеленая
    cv2.line(img, origin, y_axis, (0, 255, 0), 4)
    cv2.line(img, origin, y_axis, (50, 255, 50), 2)
    
    # Z ось - синяя
    cv2.line(img, origin, z_axis, (255, 0, 0), 4)
    cv2.line(img, origin, z_axis, (255, 50, 50), 2)
    
    # Стрелки на концах осей
    arrow_size = 8
    # X стрелка
    cv2.circle(img, x_axis, arrow_size, (0, 0, 255), -1)
    cv2.putText(img, 'X', (x_axis[0] + 10, x_axis[1]), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # Y стрелка
    cv2.circle(img, y_axis, arrow_size, (0, 255, 0), -1)
    cv2.putText(img, 'Y', (y_axis[0], y_axis[1] - 15), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Z стрелка
    cv2.circle(img, z_axis, arrow_size, (255, 0, 0), -1)
    cv2.putText(img, 'Z', (z_axis[0] - 15, z_axis[1]), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

def create_gradient_mask(shape, center, radius, inner_color, outer_color):
    """Создает градиентную маску для плавного перехода цвета"""
    mask = np.zeros((*shape, 3), dtype=np.uint8)
    y, x = np.ogrid[:shape[0], :shape[1]]
    dist_from_center = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    
    # Нормализуем расстояния и инвертируем (1 в центре, 0 на краю)
    dist_norm = np.clip(1 - (dist_from_center / radius), 0, 1)
    
    # Интерполяция цветов
    for i in range(3):
        mask[..., i] = (inner_color[i] * dist_norm + 
                       outer_color[i] * (1 - dist_norm)).astype(np.uint8)
    
    return mask

def draw_smooth_cone(img, nose, rvec, tvec, K, dist, length=90, radius=15, 
                    color=(0, 255, 255), segments=48, gradient=True):
    """
    Рисует красивый конус с градиентами и эффектами
    КОНУС РАЗВЕРНУТ: основание в носу, вершина в направлении взгляда
    """
    # Генерация точек основания (в носу) - УМЕНЬШЕННЫЙ РАДИУС
    base_points = []
    for i in range(segments):
        angle = 2 * np.pi * i / segments
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        base_points.append([x, y, 0])  # Основание в z=0 (нос)
    
    # Вершина конуса (в направлении взгляда)
    cone_3d = np.float32([[0, 0, -length]] + base_points)  # Отрицательный Z - конус смотрит от камеры
    
    # Проецируем в 2D
    pts, _ = cv2.projectPoints(cone_3d, rvec, tvec, K, dist)
    pts = np.int32(pts).reshape(-1, 2)
    
    tip_pt = tuple(pts[0])  # Вершина конуса (в направлении взгляда)
    base_pts = pts[1:]      # Точки основания (в носу)
    
    # Создаем маску для рисования
    overlay = img.copy()
    
    # Рисуем основание (в носу)
    if gradient and len(base_pts) > 2:
        try:
            # Вычисляем центр и радиус основания
            base_center = np.mean(base_pts, axis=0).astype(int)
            base_radius = max(5, int(0.8 * np.mean([np.linalg.norm(pt - base_center) for pt in base_pts])))
            
            # Цвета для градиента (от темного к светлому)
            inner_color = tuple(max(0, c - 80) for c in color)
            
            # Создаем градиентную маску
            gradient_mask = create_gradient_mask(
                img.shape[:2], base_center, base_radius,
                inner_color, color
            )
            
            # Создаем маску основания
            base_mask = np.zeros(img.shape[:2], dtype=np.uint8)
            cv2.fillPoly(base_mask, [base_pts], 255)
            
            # Накладываем градиент
            gradient_region = cv2.bitwise_and(gradient_mask, gradient_mask, mask=base_mask)
            cv2.addWeighted(gradient_region, 0.7, overlay, 0.3, 0, overlay)
            
        except Exception as e:
            # Если градиент не работает, используем простую заливку
            print(f"Gradient failed: {e}, using solid fill")
            cv2.fillPoly(overlay, [base_pts], color)
    else:
        # Простая заливка
        cv2.fillPoly(overlay, [base_pts], color)
    
    # Рисуем грани конуса (от основания к вершине)
    for i, pt in enumerate(base_pts):
        # Градиент толщины линии (тоньше к вершине)
        thickness = max(1, int(3 * (1 - i / len(base_pts))))
        cv2.line(overlay, tip_pt, tuple(pt), color, thickness)
    
    # Контур основания
    cv2.polylines(overlay, [base_pts], isClosed=True, 
                  color=tuple(min(255, c + 30) for c in color), thickness=2)
    
    # Накладываем на оригинальное изображение
    cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)
    
    # Добавляем свечение вершины
    glow_overlay = img.copy()
    cv2.circle(glow_overlay, tip_pt, 6, (255, 255, 200), -1)
    cv2.addWeighted(glow_overlay, 0.3, img, 0.7, 0, img)
    
    # Центральная точка вершины
    cv2.circle(img, tip_pt, 3, (255, 255, 255), -1)

def draw_simple_cone(img, nose, rvec, tvec, K, dist, length=90, radius=15, 
                    color=(0, 255, 255), segments=48):
    """
    Упрощенная версия конуса без градиентов
    КОНУС РАЗВЕРНУТ: основание в носу, вершина в направлении взгляда
    """
    # Генерация точек основания (в носу) - УМЕНЬШЕННЫЙ РАДИУС
    base_points = []
    for i in range(segments):
        angle = 2 * np.pi * i / segments
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        base_points.append([x, y, 0])  # Основание в z=0 (нос)
    
    # Вершина конуса (в направлении взгляда)
    cone_3d = np.float32([[0, 0, -length]] + base_points)  # Отрицательный Z - конус смотрит от камеры
    
    # Проецируем в 2D
    pts, _ = cv2.projectPoints(cone_3d, rvec, tvec, K, dist)
    pts = np.int32(pts).reshape(-1, 2)
    
    tip_pt = tuple(pts[0])  # Вершина конуса
    base_pts = pts[1:]      # Точки основания
    
    # Создаем overlay для плавного наложения
    overlay = img.copy()
    
    # Рисуем основание
    cv2.fillPoly(overlay, [base_pts], color)
    
    # Рисуем грани (от основания к вершине)
    for pt in base_pts:
        cv2.line(overlay, tip_pt, tuple(pt), color, 2)
    
    # Контур основания
    cv2.polylines(overlay, [base_pts], isClosed=True, 
                  color=(0, 200, 200), thickness=2)
    
    # Накладываем с прозрачностью
    cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)
    
    # Вершина
    cv2.circle(img, tip_pt, 4, (255, 255, 255), -1)

def draw_direction_line(img, nose, rvec, tvec, K, dist, length=120, color=(255, 255, 0)):
    """Рисует линию направления из носа"""
    # Точка в направлении головы
    direction_3d = np.float32([[0, 0, -length]])  # Отрицательный Z - направление от камеры
    pts, _ = cv2.projectPoints(direction_3d, rvec, tvec, K, dist)
    direction_pt = tuple(np.int32(pts[0][0]))
    
    # Рисуем стрелку
    cv2.arrowedLine(img, nose, direction_pt, color, 3, tipLength=0.2)

def visualize(img, nose, result, use_simple_cone=False):
    """Основная функция визуализации"""
    
    if 'rvec' in result:
        # Рисуем 3D оси
        draw_3d_axes(
            img, nose,
            result['rvec'], result['tvec'], result['K'], 
            result.get('dist', np.zeros((4,1))),
            scale=50
        )
        
        # Рисуем направление
        draw_direction_line(
            img, nose,
            result['rvec'], result['tvec'], result['K'], 
            result.get('dist', np.zeros((4,1))),
            length=120, color=(255, 200, 0)
        )
        
        # Рисуем конус (простой или сложный)
        if use_simple_cone:
            draw_simple_cone(
                img, nose,
                result['rvec'], result['tvec'], result['K'], 
                result.get('dist', np.zeros((4,1))),
                length=90, radius=15, color=(0, 255, 255), segments=48  # Уменьшен радиус
            )
        else:
            draw_smooth_cone(
                img, nose,
                result['rvec'], result['tvec'], result['K'], 
                result.get('dist', np.zeros((4,1))),
                length=90, radius=15, color=(0, 255, 255), segments=48, gradient=True  # Уменьшен радиус
            )
    
    # Добавляем информационный текст
    info_text = "Head Pose Estimation"
    cv2.putText(img, info_text, (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(img, info_text, (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 100, 255), 1)

# Дополнительная функция для отладки
def debug_visualization(img, nose, result, use_simple_cone=True):
    """Расширенная визуализация для отладки"""
    visualize(img, nose, result, use_simple_cone=use_simple_cone)
    
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