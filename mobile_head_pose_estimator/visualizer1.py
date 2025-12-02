import math
import numpy as np
import cv2

def euler_to_rotation_matrix(yaw_deg, pitch_deg, roll_deg, degrees=True):
    if degrees:
        yaw = math.radians(yaw_deg)
        pitch = math.radians(pitch_deg)
        roll = math.radians(roll_deg)
    else:
        yaw, pitch, roll = yaw_deg, pitch_deg, roll_deg

    cx, sx = math.cos(yaw), math.sin(yaw)   # yaw around X
    cy, sy = math.cos(pitch), math.sin(pitch) # pitch around Y
    cz, sz = math.cos(roll), math.sin(roll)   # roll around Z

    Rx = np.array([[1, 0, 0],
                   [0, cx, -sx],
                   [0, sx,  cx]], dtype=float)

    Ry = np.array([[ cy, 0, sy],
                   [  0, 1,  0],
                   [-sy, 0, cy]], dtype=float)

    Rz = np.array([[cz, -sz, 0],
                   [sz,  cz, 0],
                   [ 0,   0, 1]], dtype=float)

    return Rz @ Ry @ Rx

def _make_cone_points(length=180.0, radius=55.0, segments=64):
    angles = np.linspace(0.0, 2.0 * np.pi, segments, endpoint=False)
    circle = np.stack([
        radius * np.cos(angles),
        radius * np.sin(angles),
        np.full_like(angles, length)
    ], axis=1)  # (segments,3)
    tip = np.array([[0.0, 0.0, 0.0]])
    pts = np.vstack([tip, circle])
    return pts  # shape (segments+1, 3)

def _ensure_positive_z(Z, min_z=1e-3, add=200.0):
    if np.min(Z) <= min_z:
        Z = Z + (abs(np.min(Z)) + add)
    return Z

def draw_head_cone(img, nose, result,
                   length=180.0, radius=55.0, segments=64,
                   base_color=(0, 0, 0), tip_color=(255,255,255),
                   focal_length=None, z_offset=300.0, alpha=0.7,
                   draw_axes=False):
    h, w = img.shape[:2]
    if focal_length is None:
        focal_length = max(h, w) * 1.0

    # Получаем yaw/pitch/roll
    if all(k in result for k in ('yaw','pitch','roll')):
        yaw = float(result['yaw'])
        pitch = float(result['pitch'])
        roll = float(result['roll'])
    elif 'sin_b' in result:
        # Нежный fallback: pitch из sin_b; yaw из cos_minor теряет знак, поэтому ставим 0
        sin_b = float(result.get('sin_b', 0.0))
        sin_b = max(-1.0, min(1.0, sin_b))
        pitch = math.degrees(math.asin(sin_b))
        yaw = 0.0
        roll = 0.0
    else:
        raise ValueError("result must contain yaw,pitch,roll or sin_b/cos_minor")

    # Генерируем локальную геометрию конуса
    pts3d_local = _make_cone_points(length=length, radius=radius, segments=segments)  # (N,3)

    # Поворачиваем точки
    R = euler_to_rotation_matrix(yaw, pitch, roll, degrees=True)
    pts3d_rot = (R @ pts3d_local.T).T  # (N,3)

    # Сдвигаем по Z чтобы точки были перед "камерой"
    pts3d_rot[:,2] += z_offset

    X = pts3d_rot[:,0]
    Y = pts3d_rot[:,1]
    Z = pts3d_rot[:,2]
    Z = _ensure_positive_z(Z, add=z_offset)

    # Перспективная проекция
    proj_x = (X / Z) * focal_length
    proj_y = (Y / Z) * focal_length

    px = (nose[0] + proj_x).astype(int)
    py = (nose[1] - proj_y).astype(int)  # минус: в 3D Y положителен вверх, в изображении y вниз

    tip_pt = (int(px[0]), int(py[0]))
    base_pts = list(zip(px[1:], py[1:]))

    overlay = img.copy()

    # Заполнение основания (многоугольник) и граней с градиентом по индексам
    try:
        cv2.fillPoly(overlay, [np.array(base_pts, dtype=np.int32)], base_color)
    except Exception:
        pass

    # Грани (треугольники tip-base_i-base_{i+1})
    for i in range(len(base_pts)):
        p1 = base_pts[i]
        p2 = base_pts[(i+1) % len(base_pts)]
        tri = np.array([tip_pt, p1, p2], dtype=np.int32)
        depth = float(i) / len(base_pts)  # простая аппроксимация глубины
        col = tuple(int(base_color[c] * (1 - depth) + tip_color[c] * depth) for c in range(3))
        cv2.fillConvexPoly(overlay, tri, col)

    cv2.addWeighted(overlay, alpha, img, 1.0 - alpha, 0, img)

    # Контуры основания
    try:
        cv2.polylines(img, [np.array(base_pts, dtype=np.int32)], True, (255,255,255), 1)
    except Exception:
        pass

    # Круг на носу
    cv2.circle(img, (int(nose[0]), int(nose[1])), max(2, int(radius*0.18)), (0,255,255), -1)

    # Отладочные локальные оси (короткие линии от вершины)
    if draw_axes:
        # длина осей в локальной системе (мм/пикс/просто единицы)
        axis_len = min(h,w) * 0.12
        axes_local = np.array([
            [axis_len, 0.0, 0.0],  # X - red
            [0.0, axis_len, 0.0],  # Y - green
            [0.0, 0.0, axis_len],  # Z - blue
        ])
        axes_rot = (R @ axes_local.T).T  # (3,3)
        axes_rot[:,2] += z_offset
        ax_px = (nose[0] + (axes_rot[:,0] / axes_rot[:,2]) * focal_length).astype(int)
        ax_py = (nose[1] - (axes_rot[:,1] / axes_rot[:,2]) * focal_length).astype(int)

        origin = (int(nose[0]), int(nose[1]))
        # X - red
        cv2.line(img, origin, (int(ax_px[0]), int(ax_py[0])), (0,0,255), 2)
        # Y - green
        cv2.line(img, origin, (int(ax_px[1]), int(ax_py[1])), (0,255,0), 2)
        # Z - blue
        cv2.line(img, origin, (int(ax_px[2]), int(ax_py[2])), (255,0,0), 2)

    return img


# ---- Example helper  ----
def visualize(img, nose, pose_result):
    if nose is None:
        raise ValueError("landmarks must contain 'nose' key with (x,y)")

    return draw_head_cone(img, nose, pose_result, draw_axes=False)