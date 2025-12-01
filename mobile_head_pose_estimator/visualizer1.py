import cv2
import numpy as np
import math

def create_gradient_mask(shape, center, radius, inner_color, outer_color):
    mask = np.zeros((*shape[:2], 3), dtype=np.uint8)
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
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cr, sr = math.cos(roll), math.sin(roll)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    return Rz @ Ry @ Rx

def draw_perfect_cone_by_angles(img, nose, yaw_deg, pitch_deg, roll_deg,
                                length=180, radius=55, segments=64,
                                base_color=(0, 0, 0), tip_color=(255, 255, 255)):
    R = euler_to_rotation_matrix(yaw_deg, pitch_deg, roll_deg)

    base_points = []
    for i in range(segments):
        angle = 2 * np.pi * i / segments
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        base_points.append([x, y, 0])

    cone_3d = np.float32([[0, 0, length]] + base_points)
    cone_rotated = cone_3d @ R.T

    pts = []
    for pt in cone_rotated:
        x = int(nose[0] + pt[0])
        y = int(nose[1] - pt[1])  # Инверсия y для совпадения с изображением
        pts.append([x, y])
    pts = np.array(pts, np.int32)

    tip_pt = tuple(pts[0])
    base_pts = pts[1:]

    overlay = img.copy()

    # Заполнение основания
    cv2.fillPoly(overlay, [base_pts], base_color)

    # Грани с градиентом (используем маску для плавности)
    max_dist = np.max(np.linalg.norm(base_pts - np.array(tip_pt), axis=1))
    gradient_mask = create_gradient_mask(img.shape, tip_pt, max_dist * 1.2, tip_color, base_color)
    overlay = cv2.addWeighted(overlay, 1.0, gradient_mask, 0.5, 0)

    # Контур основания
    bright_color = tuple(min(255, c + 60) for c in base_color)
    cv2.polylines(overlay, [base_pts], True, bright_color, 3)

    # Наложение
    cv2.addWeighted(overlay, 0.65, img, 0.35, 0, img)

    # Свечение вершины
    glow = img.copy()
    cv2.circle(glow, tip_pt, 10, tip_color, -1)
    cv2.addWeighted(glow, 0.3, img, 0.7, 0, img)
    cv2.circle(img, tip_pt, 4, tip_color, -1)

    # Оси
    axes = np.float32([[60,0,0], [0,60,0], [0,0,60]]) @ R.T
    def p(i): return (int(nose[0] + axes[i][0]), int(nose[1] - axes[i][1]))
    cv2.line(img, nose, p(0), (0,0,255), 3)  # red x
    cv2.line(img, nose, p(1), (0,255,0), 3)  # green y
    cv2.line(img, nose, p(2), (255,0,0), 3)  # blue z

def visualize(img, nose, result):
    if 'rvec' in result and 'tvec' in result and 'K' in result and 'dist' in result:
        rvec = result['rvec']
        tvec = result['tvec']
        camera_matrix = result['K']
        dist_coeffs = result['dist']

        length = 0.2  # в единицах модели (~20 см)
        radius = 0.1  # радиус основания
        segments = 64

        tip = np.array([[0, 0, length]], dtype=np.float32)
        base_points = []
        for i in range(segments):
            angle = 2 * math.pi * i / segments
            x = radius * math.cos(angle)
            y = radius * math.sin(angle)
            base_points.append([x, y, 0])
        base_points = np.array(base_points, dtype=np.float32)

        cone_3d = np.vstack((tip, base_points))

        image_points, _ = cv2.projectPoints(cone_3d, rvec, tvec, camera_matrix, dist_coeffs)
        image_points = image_points.squeeze().astype(np.int32)

        tip_pt = tuple(image_points[0])
        base_pts = image_points[1:]

        base_color = (0, 0, 0)
        tip_color = (255, 255, 255)

        overlay = img.copy()
        cv2.fillPoly(overlay, [base_pts], base_color)

        # Градиент для линий
        for i, pt in enumerate(base_pts):
            t = i / segments
            color = (int(255 * t), int(255 * t), int(255 * t))  # gray gradient
            cv2.line(overlay, tip_pt, tuple(pt), color, 2)

        cv2.addWeighted(overlay, 0.65, img, 0.35, 0, img)

        # Оси с проекцией
        axes_3d = np.float32([[0,0,0], [0.1,0,0], [0,0.1,0], [0,0,0.1]])
        axes_2d, _ = cv2.projectPoints(axes_3d, rvec, tvec, camera_matrix, dist_coeffs)
        axes_2d = axes_2d.squeeze().astype(int)
        origin = tuple(axes_2d[0])
        cv2.line(img, origin, tuple(axes_2d[1]), (0,0,255), 3)  # red x
        cv2.line(img, origin, tuple(axes_2d[2]), (0,255,0), 3)  # green y
        cv2.line(img, origin, tuple(axes_2d[3]), (255,0,0), 3)  # blue z

    elif 'yaw' in result and 'pitch' in result and 'roll' in result:
        draw_perfect_cone_by_angles(img, nose, result['yaw'], result['pitch'], result['roll'])

    elif 'sin_b' in result:
        sin_b = result['sin_b']
        cos_minor = result['cos_minor']
        if sin_b == -8.0: return
        pitch = np.degrees(np.arcsin(sin_b))
        yaw = np.degrees(np.arccos(cos_minor)) if abs(cos_minor) <= 1 else 0
        roll = 0
        draw_perfect_cone_by_angles(img, nose, yaw, pitch, roll)
