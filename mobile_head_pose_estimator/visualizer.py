# visualizer.py
import cv2
import numpy as np

def draw_axes(img, nose, scale=50):
    """Рисует красивые оси с подписами"""
    # X — красная (вправо)
    cv2.line(img, nose, (nose[0] + scale, nose[1]), (0, 0, 255), 2)
    cv2.putText(img, 'X', (nose[0] + scale + 5, nose[1]), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    
    # Y — зелёная (вверх)  
    cv2.line(img, nose, (nose[0], nose[1] - scale), (0, 255, 0), 2)
    cv2.putText(img, 'Y', (nose[0], nose[1] - scale - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    # Z — синяя (влево - вглубь экрана)
    cv2.line(img, nose, (nose[0] - scale, nose[1]), (255, 0, 0), 2)
    cv2.putText(img, 'Z', (nose[0] - scale - 15, nose[1]), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

def draw_burattino_nose(img, nose, rvec, tvec, K, dist, nose_length=40, base_radius=15, tip_radius=8, color=(0, 200, 255)):
    """
    Рисует нос как у Буратино - выступающий конус от лица
    
    - nose_length: длина носа (вперед по Z-оси)
    - base_radius: радиус основания носа (у лица)
    - tip_radius: радиус кончика носа
    - color: цвет носа (оранжево-желтый)
    """
    segments = 16
    
    # Создаем точки для цилиндра/конуса носа
    nose_points = []
    
    # Базовое кольцо (у лица)
    for i in range(segments):
        angle = 2 * np.pi * i / segments
        x = base_radius * np.cos(angle)
        y = base_radius * np.sin(angle)
        z = 0  # У лица
        nose_points.append([x, y, z])
    
    # Кончик носа
    for i in range(segments):
        angle = 2 * np.pi * i / segments
        x = tip_radius * np.cos(angle)
        y = tip_radius * np.sin(angle)
        z = nose_length  # Впереди
        nose_points.append([x, y, z])
    
    # Вершина кончика носа (самая дальняя точка)
    nose_points.append([0, 0, nose_length + tip_radius])
    
    # Преобразуем в numpy массив
    nose_3d = np.float32(nose_points)
    
    # Проекция в 2D
    pts, _ = cv2.projectPoints(nose_3d, rvec, tvec, K, dist)
    pts = np.int32(pts).reshape(-1, 2)
    
    # Разделяем точки
    base_pts = pts[:segments]           # Основание у лица
    tip_pts = pts[segments:2*segments]  # Кончик носа
    nose_tip = pts[-1]                  # Самая дальняя точка
    
    # Создаем overlay для плавного наложения
    overlay = img.copy()
    
    # === РИСУЕМ ОСНОВАНИЕ НОСА ===
    if len(base_pts) > 2:
        cv2.fillPoly(overlay, [base_pts], color)
    
    # === РИСУЕМ КОНЧИК НОСА ===
    if len(tip_pts) > 2:
        cv2.fillPoly(overlay, [tip_pts], color)
    
    # === РИСУЕМ БОКОВУЮ ПОВЕРХНОСТЬ ===
    for i in range(segments):
        # Соединяем основание с кончиком
        next_i = (i + 1) % segments
        cv2.line(overlay, tuple(base_pts[i]), tuple(tip_pts[i]), color, 2)
        cv2.line(overlay, tuple(base_pts[i]), tuple(base_pts[next_i]), color, 1)
        cv2.line(overlay, tuple(tip_pts[i]), tuple(tip_pts[next_i]), color, 1)
    
    # === РИСУЕМ ЦЕНТРАЛЬНУЮ ЛИНИЮ НОСА ===
    base_center = np.mean(base_pts, axis=0).astype(int)
    cv2.line(overlay, tuple(base_center), tuple(nose_tip), color, 3)
    
    # Накладываем с прозрачностью
    cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)
    
    # Выделяем кончик носа
    cv2.circle(img, tuple(nose_tip), 4, (255, 100, 0), -1)
    
    return nose_tip

def draw_simple_nose(img, nose, rvec, tvec, K, dist, nose_length=30, color=(0, 200, 255)):
    """
    Упрощенная версия носа Буратино - только конус
    """
    # Создаем простой конус
    segments = 12
    cone_points = [[0, 0, 0]]  # Основание в носу
    
    # Точки основания
    base_radius = 12
    for i in range(segments):
        angle = 2 * np.pi * i / segments
        x = base_radius * np.cos(angle)
        y = base_radius * np.sin(angle)
        cone_points.append([x, y, 0])
    
    # Кончик носа
    cone_points.append([0, 0, nose_length])
    
    # Проекция
    cone_3d = np.float32(cone_points)
    pts, _ = cv2.projectPoints(cone_3d, rvec, tvec, K, dist)
    pts = np.int32(pts).reshape(-1, 2)
    
    nose_base = tuple(pts[0])
    base_circle = pts[1:1+segments]
    nose_tip = tuple(pts[-1])
    
    # Рисуем
    overlay = img.copy()
    
    # Основание
    if len(base_circle) > 2:
        cv2.fillPoly(overlay, [base_circle], color)
    
    # Боковые грани
    for pt in base_circle:
        cv2.line(overlay, nose_base, tuple(pt), color, 2)
        cv2.line(overlay, tuple(pt), nose_tip, color, 2)
    
    # Центральная линия
    cv2.line(overlay, nose_base, nose_tip, (255, 150, 0), 3)
    
    cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)
    cv2.circle(img, nose_tip, 4, (255, 100, 0), -1)
    
    return nose_tip

def draw_head_direction(img, nose, rvec, tvec, K, dist, length=80, color=(255, 255, 0)):
    """Рисует направление взгляда"""
    # Точка в направлении головы (вперед по Z)
    direction_3d = np.float32([[0, 0, -length]])  # Отрицательный Z - вперед из экрана
    pts, _ = cv2.projectPoints(direction_3d, rvec, tvec, K, dist)
    direction_pt = tuple(np.int32(pts[0][0]))
    
    # Рисуем стрелку направления
    cv2.arrowedLine(img, nose, direction_pt, color, 2, tipLength=0.15)

def visualize(img, nose, result, nose_style="burattino"):
    """
    Основная функция визуализации
    
    nose_style: 
      - "burattino" - полноценный нос Буратино 
      - "simple" - простой конус
      - "none" - без носа
    """
    # Рисуем оси
    draw_axes(img, nose)
    
    if 'rvec' in result:
        # Рисуем направление взгляда
        draw_head_direction(
            img, nose,
            result['rvec'], result['tvec'], result['K'], 
            result.get('dist', np.zeros((4,1))),
            length=60, color=(255, 200, 0)
        )
        
        # Рисуем нос в зависимости от выбранного стиля
        if nose_style == "burattino":
            nose_tip = draw_burattino_nose(
                img, nose,
                result['rvec'], result['tvec'], result['K'], 
                result.get('dist', np.zeros((4,1))),
                nose_length=35, base_radius=10, tip_radius=6, color=(0, 200, 255)
            )
        elif nose_style == "simple":
            nose_tip = draw_simple_nose(
                img, nose,
                result['rvec'], result['tvec'], result['K'], 
                result.get('dist', np.zeros((4,1))),
                nose_length=30, color=(0, 200, 255)
            )
        
        # Отмечаем исходную точку носа
        cv2.circle(img, nose, 3, (255, 255, 255), -1)
        cv2.circle(img, nose, 5, (0, 100, 255), 1)
    
    # Информация
    cv2.putText(img, "Head Pose + Nose Direction", (5, 15), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

# Функция для тестирования с вашим JSON
def test_with_json_data(img, json_data):
    """
    Тестовая функция для работы с вашими данными JSON
    """
    # Создаем фиктивные параметры камеры (подходят для вашего размера изображения)
    h, w = json_data["image_size"]
    K = np.array([[w, 0, w/2],
                  [0, w, h/2], 
                  [0, 0, 1]], dtype=np.float32)
    
    # Фиксированная позиция камеры
    tvec = np.array([[0, 0, 500]], dtype=np.float32)
    
    # Преобразуем углы Эйлера в вектор вращения
    pitch = np.radians(json_data["ground_truth"]["pitch"])
    yaw = np.radians(json_data["ground_truth"]["yaw"]) 
    roll = np.radians(json_data["ground_truth"]["roll"])
    
    rvec = np.array([pitch, yaw, roll], dtype=np.float32)
    
    # Точка носа из JSON
    nose = tuple(json_data["props"]["kp_nose_tip"])
    
    # Собираем результат
    result = {
        'rvec': rvec,
        'tvec': tvec, 
        'K': K,
        'dist': np.zeros((4, 1))
    }
    
    # Визуализируем
    visualize(img, nose, result, nose_style="burattino")
    
    return img