import cv2
import numpy as np

def draw_3d_axes(img, rvec, tvec, K, dist, nose, axis_length=40):
    """
    Рисует классические 3D оси (X - красный, Y - зелёный, Z - синий)
    с учётом поворота головы.
    """
    # Концы осей в системе координат головы
    axis_points = np.float32([
        [0, 0, 0],           # начало
        [axis_length, 0, 0], # X
        [0, axis_length, 0], # Y
        [0, 0, -axis_length] # Z (вперёд — в отрицательную Z)
    ])

    # Проекция
    pts, _ = cv2.projectPoints(axis_points, rvec, tvec, K, dist)
    pts = np.int32(pts).reshape(-1, 2)
    origin = tuple(pts[0])
    x_end = tuple(pts[1])
    y_end = tuple(pts[2])
    z_end = tuple(pts[3])

    # Рисуем оси
    cv2.arrowedLine(img, origin, x_end, (0, 0, 255), 2, tipLength=0.2)   # X - красный
    cv2.arrowedLine(img, origin, y_end, (0, 255, 0), 2, tipLength=0.2)   # Y - зелёный
    cv2.arrowedLine(img, origin, z_end, (255, 0, 0), 2, tipLength=0.2)   # Z - синий

    # Подписи
    cv2.putText(img, 'X', (x_end[0] + 5, x_end[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    cv2.putText(img, 'Y', (y_end[0] + 5, y_end[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText(img, 'Z', (z_end[0] + 5, z_end[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    return origin  # возвращаем начало осей (должно совпадать с носом)

def draw_cone(img, rvec, tvec, K, dist, cone_length=40, base_radius=6, segments=16):
    """
    Конус: основание на кончике носа, высота — по направлению лица (вперёд = -Z).
    """
    cone_points = []

    # Основание: круг в плоскости XY, центр в (0,0,0), радиус base_radius
    for i in range(segments):
        angle = 2 * np.pi * i / segments
        x = base_radius * np.cos(angle)
        y = base_radius * np.sin(angle)
        cone_points.append([x, y, 0.0])  # Z=0 — основание на носу

    # Вершина конуса: вперёд по -Z
    cone_points.append([0, 0, -cone_length])

    cone_3d = np.float32(cone_points)
    pts, _ = cv2.projectPoints(cone_3d, rvec, tvec, K, dist)
    pts = np.int32(pts).reshape(-1, 2)

    base_pts = pts[:-1]
    tip_pt = tuple(pts[-1])

    # Рисуем основание (круг на носу)
    nose_2d = tuple(pts[0])  # проекция центра основания
    cv2.circle(img, nose_2d, base_radius, (100, 200, 255), -1)  # заливка
    cv2.circle(img, nose_2d, base_radius, (50, 150, 200), 2)    # контур

    # Рисуем боковые рёбра
    for pt in base_pts:
        cv2.line(img, pt, tip_pt, (100, 200, 255), 1)

    # Центральная ось конуса (от носа к вершине)
    cv2.line(img, nose_2d, tip_pt, (0, 150, 255), 2)

    # Выделяем вершину
    cv2.circle(img, tip_pt, 4, (255, 150, 0), -1)

    return tip_pt, nose_2d

def visualize(img, nose, result):
    if 'rvec' not in result:
        return

    rvec = result['rvec']
    tvec = result['tvec']
    K = result['K']
    dist = result.get('dist', np.zeros((4, 1)))

    # 1. Рисуем 3D оси
    origin = draw_3d_axes(img, rvec, tvec, K, dist, nose, axis_length=35)

    # 2. Рисуем конус
    tip_pt, nose_2d = draw_cone(img, rvec, tvec, K, dist, cone_length=40, base_radius=6)

    # 3. Подпись
    cv2.putText(img, "HEAD DIRECTION", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(img, "HEAD DIRECTION", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)

# Тестовая функция (без изменений, кроме вызова)
def test_simple_visualization(img, json_data):
    h, w = json_data["image_size"]
    K = np.array([[w, 0, w/2],
                  [0, w, h/2], 
                  [0, 0, 1]], dtype=np.float32)
    
    tvec = np.array([[0, 0, 500]], dtype=np.float32).T  # (3,1)

    pitch = np.radians(json_data["ground_truth"]["pitch"])
    yaw = np.radians(json_data["ground_truth"]["yaw"]) 
    roll = np.radians(json_data["ground_truth"]["roll"])
    
    rvec = np.array([pitch, yaw, roll], dtype=np.float32).reshape(3, 1)  # (3,1)

    nose = tuple(json_data["props"]["kp_nose_tip"])
    
    result = {
        'rvec': rvec,
        'tvec': tvec, 
        'K': K,
        'dist': np.zeros((4, 1))
    }
    
    visualize(img, nose, result)
    return img