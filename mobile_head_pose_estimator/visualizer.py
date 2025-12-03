import cv2
import numpy as np
import math

def create_gradient_mask(shape, center, radius, inner_color, outer_color):
    """Создает градиентную маску для плавного перехода цвета"""
    mask = np.zeros((*shape, 3), dtype=np.uint8)
    y, x = np.ogrid[:shape[0], :shape[1]]
    dist_from_center = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    dist_norm = np.clip(1 - (dist_from_center / radius), 0, 1)
    for i in range(3):
        mask[..., i] = (inner_color[i] * dist_norm + outer_color[i] * (1 - dist_norm)).astype(np.uint8)
    return mask

def euler_to_rotation_matrix(yaw_deg, pitch_deg, roll_deg):
    yaw = math.radians(yaw_deg)
    pitch = math.radians(pitch_deg)
    roll = math.radians(roll_deg)

    cx, sx = math.cos(yaw), math.sin(yaw)    # для вращения вокруг X
    cy, sy = math.cos(pitch), math.sin(pitch) # для вращения вокруг Y
    cz, sz = math.cos(roll), math.sin(roll)   # для вращения вокруг Z

    Rx = np.array([[1,   0,   0],
                   [0,  cx, -sx],
                   [0,  sx,  cx]], dtype=float)

    Ry = np.array([[ cy, 0, sy],
                   [  0, 1,  0],
                   [-sy, 0, cy]], dtype=float)

    Rz = np.array([[cz, -sz, 0],
                   [sz,  cz, 0],
                   [ 0,   0, 1]], dtype=float)

    return Rz @ Ry @ Rx

def draw_perfect_cone_by_angles(img, nose, yaw_deg, pitch_deg, roll_deg,
                                length=180, radius=55, segments=64,
                                base_color=(0, 0, 0),  # Черный у основания
                                tip_color=(255, 255, 255),  # Белый у вершины
                                gradient=True):
    # R = euler_to_rotation_matrix(yaw_deg, pitch_deg, roll_deg)
    R = euler_to_rotation_matrix(-pitch_deg, -roll_deg, -yaw_deg)

    base_points = []
    for i in range(segments):
        angle = 2 * np.pi * i / segments
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        base_points.append([x, y, 0])

    cone_3d = np.float32([[0, 0, length]] + base_points)  # вершина + основание
    cone_rotated = cone_3d @ R.T

    # Проекция
    pts = []
    for pt in cone_rotated:
        x = int(nose[0] + pt[0])
        y = int(nose[1] - pt[1])
        pts.append([x, y])
    pts = np.array(pts, np.int32)

    tip_pt = tuple(pts[0])
    base_pts = pts[1:]

    overlay = img.copy()

    cv2.fillPoly(overlay, [base_pts], base_color)

    # === ГРАНИ С ГРАДИЕНТОМ ===
    for i, pt in enumerate(base_pts):
        t = i / len(base_pts)
        color_ratio = t 
        r = int(base_color[0] * (1 - color_ratio) + tip_color[0] * color_ratio)
        g = int(base_color[1] * (1 - color_ratio) + tip_color[1] * color_ratio)
        b = int(base_color[2] * (1 - color_ratio) + tip_color[2] * color_ratio)
        line_color = (b, g, r)
        thickness = max(1, int(4 * (1 - t**0.7)))
        cv2.line(overlay, tip_pt, tuple(pt), line_color, thickness)

    # === КОНТУР ОСНОВАНИЯ ===
    bright_color = tuple(min(255, c + 60) for c in base_color)
    cv2.polylines(overlay, [base_pts], True, bright_color, 3)

    # === НАЛОЖЕНИЕ ===
    cv2.addWeighted(overlay, 0.65, img, 0.35, 0, img)

    # === СВЕЧЕНИЕ ВЕРШИНЫ ===
    glow = img.copy()
    cv2.circle(glow, tip_pt, 10, tip_color, -1)  # Используем цвет вершины
    cv2.addWeighted(glow, 0.3, img, 0.7, 0, img)
    cv2.circle(img, tip_pt, 4, tip_color, -1)  # Используем цвет вершины

    # === ОСИ ===
    axes = np.float32([[60,0,0], [0,60,0], [0,0,60]]) @ R.T
    def p(i): return (int(nose[0] + axes[i][0]), int(nose[1] - axes[i][1]))
    cv2.line(img, nose, p(0), (0,0,255), 3)
    cv2.line(img, nose, p(1), (0,255,0), 3)
    cv2.line(img, nose, p(2), (255,0,0), 3)

# def draw_perfect_cone_by_angles(img, nose, yaw_deg, pitch_deg, roll_deg,
#                                 length=180, radius=55, segments=64,
#                                 base_color=(0, 0, 0),  # Черный у основания
#                                 tip_color=(255, 255, 255)):
#     # Локальная геометрия
#     angles = np.linspace(0, 2 * np.pi, segments, endpoint=False)
#     circle = np.stack([radius * np.cos(angles),
#                        radius * np.sin(angles),
#                        np.full_like(angles, length)], axis=1)

#     tip = np.array([0.0, 0.0, 0.0])
#     pts = np.vstack([tip, circle])  # (segments+1, 3)

#     # Поворот
#     R = euler_to_rotation_matrix(yaw_deg, pitch_deg, roll_deg)
#     cone_rotated = (R @ pts.T).T  # (N,3)

#     # Ортографическая проекция: просто отбрасываем Z
#     pts2d = cone_rotated[:, :2]

#     # В пикселях: сохраняем nose как проекцию вершины
#     pts2d_pixels = []
#     for pt in pts2d:
#         x = int(round(nose[0] + pt[0]))
#         y = int(round(nose[1] - pt[1]))  # минус: 3D Y вверх -> в image y вниз
#         pts2d_pixels.append([x, y])
#     pts2d_pixels = np.array(pts2d_pixels, np.int32)

#     tip_pt = tuple(pts2d_pixels[0])
#     base_pts = pts2d_pixels[1:]

#     overlay = img.copy()

#     # Основание
#     cv2.fillPoly(overlay, [base_pts], base_color)

#     # Грани с градиентом (как у вас было)
#     for i, pt in enumerate(base_pts):
#         next_pt = base_pts[(i + 1) % len(base_pts)]
#         tri = np.array([tip_pt, tuple(pt), tuple(next_pt)], dtype=np.int32)
#         depth = float(i) / len(base_pts)
#         col = tuple(int(base_color[j] * (1 - depth) + tip_color[j] * depth) for j in range(3))
#         cv2.fillConvexPoly(overlay, tri, col)

#     cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)

def visualize(img, nose, result):
    """УНИВЕРСАЛЬНАЯ ВИЗУАЛИЗАЦИЯ ПО УГЛАМ"""
    if 'yaw' in result and 'pitch' in result and 'roll' in result:
        yaw, pitch, roll = result['yaw'], result['pitch'], result['roll']
    elif 'sin_b' in result:
        sin_b = result['sin_b']
        cos_minor = result['cos_minor']
        if sin_b == -8.0: return
        pitch = np.degrees(np.arcsin(sin_b))
        yaw = np.degrees(np.arccos(cos_minor)) if abs(cos_minor) <= 1 else 0
        roll = 0
    else:
        return

    draw_perfect_cone_by_angles(img, nose, yaw, pitch, roll,
                                base_color=(0, 0, 0),
                                tip_color=(255, 255, 255))