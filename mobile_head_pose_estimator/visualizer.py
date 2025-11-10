import cv2
import numpy as np

def draw_head_axes(img, rvec, tvec, K, dist, axis_length=40):
    """
    Рисует 3D оси, привязанные к голове:
    - X (красный): к правому глазу
    - Y (зелёный): вверх (между глаз)
    - Z (синий): вперёд из лица
    """
    # Концы осей в локальной системе головы
    axis_points = np.float32([
        [0, 0, 0],           # центр (нос)
        [axis_length, 0, 0], # +X → право
        [0, axis_length, 0], # +Y → вверх
        [0, 0, -axis_length] # +Z → вперёд (в камеру)
    ])

    pts, _ = cv2.projectPoints(axis_points, rvec, tvec, K, dist)
    pts = np.int32(pts).reshape(-1, 2)
    origin = tuple(pts[0])
    x_end, y_end, z_end = tuple(pts[1]), tuple(pts[2]), tuple(pts[3])

    # Рисуем оси
    cv2.arrowedLine(img, origin, x_end, (0, 0, 255), 2, tipLength=0.2)     # X
    cv2.arrowedLine(img, origin, y_end, (0, 255, 0), 2, tipLength=0.2)     # Y
    cv2.arrowedLine(img, origin, z_end, (255, 0, 0), 2, tipLength=0.2)     # Z

    # Подписи
    cv2.putText(img, 'X', (x_end[0]+3, x_end[1]+5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
    cv2.putText(img, 'Y', (y_end[0]+3, y_end[1]+5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
    cv2.putText(img, 'Z', (z_end[0]+3, z_end[1]+5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1)

    return origin

def draw_mini_axes(img, rvec, size=60, margin=15):
    """
    Мини-визуализация осей в правом верхнем углу
    как в Blender: показывает ориентацию головы в мире
    """
    h, w = img.shape[:2]
    overlay = img.copy()
    x0, y0 = w - size - margin, margin

    # Фон
    cv2.rectangle(overlay, (x0-5, y0-5), (x0+size+5, y0+size+5), (30, 30, 30), -1)
    cv2.rectangle(overlay, (x0-5, y0-5), (x0+size+5, y0+size+5), (200, 200, 200), 1)

    # Центр мини-осей
    center = np.float32([[0, 0, 0]])
    axis_ends = np.float32([
        [20, 0, 0],   # X
        [0, 20, 0],   # Y
        [0, 0, -20]   # Z
    ])

    # Проекция с ортогональной камерой (вид сбоку)
    R, _ = cv2.Rodrigues(rvec)
    view_matrix = np.eye(4)
    view_matrix[:3, :3] = R

    # Ортогональная проекция: просто умножаем на R и масштабируем
    center_2d = (x0 + size//2, y0 + size//2)
    scale = size / 50.0

    def project_local(pt):
        rotated = R @ pt
        px = center_2d[0] + rotated[0] * scale
        py = center_2d[1] - rotated[1] * scale  # Y вверх
        return int(px), int(py)

    origin_2d = project_local([0,0,0])
    x_2d = project_local(axis_ends[0])
    y_2d = project_local(axis_ends[1])
    z_2d = project_local(axis_ends[2])

    # Рисуем оси
    cv2.arrowedLine(overlay, origin_2d, x_2d, (0,0,255), 2, tipLength=0.3)
    cv2.arrowedLine(overlay, origin_2d, y_2d, (0,255,0), 2, tipLength=0.3)
    cv2.arrowedLine(overlay, origin_2d, z_2d, (255,0,0), 2, tipLength=0.3)

    # Подписи
    cv2.putText(overlay, 'X', (x_2d[0]+2, x_2d[1]+5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,255), 1)
    cv2.putText(overlay, 'Y', (y_2d[0]+2, y_2d[1]+5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0), 1)
    cv2.putText(overlay, 'Z', (z_2d[0]+2, z_2d[1]+5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,0,0), 1)

    # Надпись
    cv2.putText(overlay, 'HEAD', (x0, y0-5), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200,200,200), 1)

    # Накладываем с прозрачностью
    cv2.addWeighted(overlay, 0.8, img, 0.2, 0, img)

def draw_cone(img, rvec, tvec, K, dist, cone_length=45, base_radius=7, segments=16):
    """
    Конус: основание на носу, вершина — вперёд по Z
    """
    cone_points = []
    for i in range(segments):
        angle = 2 * np.pi * i / segments
        x = base_radius * np.cos(angle)
        y = base_radius * np.sin(angle)
        cone_points.append([x, y, 0])
    cone_points.append([0, 0, -cone_length])

    pts, _ = cv2.projectPoints(np.float32(cone_points), rvec, tvec, K, dist)
    pts = np.int32(pts).reshape(-1, 2)

    base_pts = pts[:-1]
    tip_pt = tuple(pts[-1])
    nose_2d = tuple(pts[0])

    # Основание
    cv2.circle(img, nose_2d, base_radius, (100, 200, 255), -1)
    cv2.circle(img, nose_2d, base_radius, (50, 150, 200), 2)

    # Рёбра
    for pt in base_pts:
        cv2.line(img, pt, tip_pt, (100, 200, 255), 1)
    cv2.line(img, nose_2d, tip_pt, (0, 150, 255), 2)

    # Вершина
    cv2.circle(img, tip_pt, 4, (255, 150, 0), -1)

    return tip_pt

def visualize(img, nose, result):
    if 'rvec' not in result:
        return

    rvec = result['rvec']
    tvec = result['tvec']
    K = result['K']
    dist = result.get('dist', np.zeros((4, 1)))

    # 1. Основные оси на голове
    origin = draw_head_axes(img, rvec, tvec, K, dist, axis_length=40)

    # 2. Конус направления
    tip_pt = draw_cone(img, rvec, tvec, K, dist, cone_length=45, base_radius=7)

    # 3. Мини-карта в углу
    draw_mini_axes(img, rvec, size=60, margin=15)

    # 4. Подпись
    cv2.putText(img, "HEAD DIRECTION", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    cv2.putText(img, "HEAD DIRECTION", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 1)

# Тест
def test_simple_visualization(img, json_data):
    h, w = json_data["image_size"]
    K = np.array([[w, 0, w/2], [0, w, h/2], [0, 0, 1]], dtype=np.float32)
    tvec = np.array([[0, 0, 500]], dtype=np.float32).T

    pitch = np.radians(json_data["ground_truth"]["pitch"])
    yaw = np.radians(json_data["ground_truth"]["yaw"])
    roll = np.radians(json_data["ground_truth"]["roll"])
    rvec = np.array([pitch, yaw, roll], dtype=np.float32).reshape(3, 1)

    nose = tuple(json_data["props"]["kp_nose_tip"])

    result = {'rvec': rvec, 'tvec': tvec, 'K': K, 'dist': np.zeros((4,1))}
    visualize(img, nose, result)
    return img