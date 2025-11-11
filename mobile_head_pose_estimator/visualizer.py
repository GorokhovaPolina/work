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

def draw_smooth_cone(img, nose, rvec, tvec, K, dist, length, radius, 
                    color=(0, 255, 255), segments=48, gradient=True):
    base_points = []
    for i in range(segments):
        angle = 2 * np.pi * i / segments
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        base_points.append([x, y, 0])  # Основание в z=0 (нос)
    
    # ИСПРАВЛЕНИЕ: вершина конуса теперь в направлении взгляда (отрицательное Z)
    cone_3d = np.float32([[0, 0, -length]] + base_points)  # Вершина на расстоянии length ВПЕРЕДИ
    
    # Проекция в 2D
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

def draw_direction_line(img, nose, rvec, tvec, K, dist, length, color=(255, 255, 0)):
    """Рисует линию направления из носа"""
    # ИСПРАВЛЕНИЕ: точка в направлении взгляда (отрицательное Z)
    direction_3d = np.float32([[0, 0, -length]])  # Вперед по оси Z
    pts, _ = cv2.projectPoints(direction_3d, rvec, tvec, K, dist)
    direction_pt = tuple(np.int32(pts[0][0]))
    
    # Рисуем стрелку
    cv2.arrowedLine(img, nose, direction_pt, color, 3, tipLength=0.2)

def visualize(img, nose, result, use_simple_cone=False):
    """Основная функция визуализации"""
    # Рисуем оси
    draw_axes(img, nose)
    
    if 'rvec' in result:
        # Рисуем направление
        draw_direction_line(
            img, nose,
            result['rvec'], result['tvec'], result['K'], 
            result.get('dist', np.zeros((4,1))),
            length=1.20, color=(255, 200, 0)
        )
        
        # Рисуем конус (простой или сложный)
        draw_smooth_cone(
            img, nose,
            result['rvec'], result['tvec'], result['K'], 
            result.get('dist', np.zeros((4,1))),
            length=9, radius=0.1, color=(0, 255, 255), segments=48, gradient=True
        )
    
    # Добавляем информационный текст
    info_text = "Head Pose Estimation"
    cv2.putText(img, info_text, (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(img, info_text, (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 100, 255), 1)