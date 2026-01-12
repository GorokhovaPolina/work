import cv2
import numpy as np
import math

def create_gradient_mask(shape, center, radius, inner_color, outer_color):
    mask = np.zeros((*shape, 3), dtype=np.uint8)
    y, x = np.ogrid[:shape[0], :shape[1]]
    dist_from_center = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    dist_norm = np.clip(1 - (dist_from_center / radius), 0, 1)
    for i in range(3):
        mask[..., i] = (inner_color[i] * dist_norm + outer_color[i] * (1 - dist_norm)).astype(np.uint8)
    return mask

def euler_to_rotation_matrix(pitch_deg, yaw_deg, roll_deg):
    pitch = math.radians(pitch_deg)
    yaw = math.radians(yaw_deg)
    roll = math.radians(roll_deg)

    Rx = np.array([[1, 0, 0],
                   [0, math.cos(pitch), -math.sin(pitch)],
                   [0, math.sin(pitch), math.cos(pitch)]], dtype=float)

    Ry = np.array([[math.cos(yaw), 0, math.sin(yaw)],
                   [0, 1, 0],
                   [-math.sin(yaw), 0, math.cos(yaw)]], dtype=float)

    Rz = np.array([[math.cos(roll), -math.sin(roll), 0],
                   [math.sin(roll), math.cos(roll), 0],
                   [0, 0, 1]], dtype=float)

    return Rz @ Ry @ Rx

def draw_perfect_cone_by_angles(img, nose, yaw_deg, pitch_deg, roll_deg,
                                length=180, radius=55, segments=64,
                                base_color=(0, 0, 0),  # Черный у основания
                                tip_color=(255, 255, 255),  # Белый у вершины
                                gradient=True, rvec=None, tvec=None, K=None, dist=None):
    nose = tuple(nose)  # Убедимся, что nose - tuple (x, y)

    if rvec is not None and tvec is not None and K is not None and dist is not None:
        # === Perspective projection mode ===
        # Base at Z=0 (nose), tip at Z=length (away) or -length (toward camera)
        base_points = []
        for i in range(segments):
            angle = 2 * math.pi * i / segments
            x = radius * math.cos(angle)
            y = radius * math.sin(angle)
            base_points.append([x, y, 0])

        cone_3d = np.array(base_points + [[0, 0, -length]], dtype=np.float32)  # base + tip

        # Project points
        (pts2d, _) = cv2.projectPoints(cone_3d, rvec, tvec, K, dist)
        pts = pts2d[:, 0, :].astype(np.int32)

        base_pts = pts[:segments]
        tip_pt = tuple(pts[segments])

        # Если проекция неудачная (NaN или out of bounds), fallback to orthographic
        if np.any(np.isnan(pts)):
            rvec = None  # Fallback

    if rvec is None:
        # === Orthographic fallback ===
        R = euler_to_rotation_matrix(-pitch_deg, yaw_deg, -roll_deg)
        base_points = []
        for i in range(segments):
            angle = 2 * math.pi * i / segments
            x = radius * math.cos(angle)
            y = radius * math.sin(angle)
            base_points.append([x, y, 0])

        cone_3d = np.float32([[0, 0, length]] + base_points)  # tip + base
        cone_rotated = cone_3d @ R.T

        pts = []
        for pt in cone_rotated:
            x = int(nose[0] + pt[0])
            y = int(nose[1] - pt[1])  # Инверсия Y для image coords
            pts.append([x, y])
        pts = np.array(pts, np.int32)

        tip_pt = tuple(pts[0])
        base_pts = pts[1:]

    # === Drawing (common for both modes) ===
    overlay = img.copy()

    # Fill base
    cv2.fillPoly(overlay, [base_pts], base_color)

    # Gradient lines from tip to base
    for i, pt in enumerate(base_pts):
        t = i / len(base_pts)
        color_ratio = t
        r = int(base_color[0] * (1 - color_ratio) + tip_color[0] * color_ratio)
        g = int(base_color[1] * (1 - color_ratio) + tip_color[1] * color_ratio)
        b = int(base_color[2] * (1 - color_ratio) + tip_color[2] * color_ratio)
        line_color = (b, g, r)
        thickness = max(1, int(4 * (1 - t**0.7)))
        cv2.line(overlay, tip_pt, tuple(pt), line_color, thickness)

    # Base outline
    bright_color = tuple(min(255, c + 60) for c in base_color)
    cv2.polylines(overlay, [base_pts], True, bright_color, 3)

    # Overlay on img
    cv2.addWeighted(overlay, 0.65, img, 0.35, 0, img)

    # Tip glow
    glow = img.copy()
    # cv2.circle(glow, tip_pt, 10, tip_color, -1)
    cv2.addWeighted(glow, 0.3, img, 0.7, 0, img)
    cv2.circle(img, tip_pt, 4, tip_color, -1)

    # === Axes ===
    if rvec is not None and tvec is not None and K is not None and dist is not None:
        axes_3d = np.float32([[0,0,0], [60,0,0], [0,60,0], [0,0,60]])
        (proj_pts, _) = cv2.projectPoints(axes_3d, rvec, tvec, K, dist)
        proj_pts = proj_pts[:,0,:].astype(np.int32)
        origin = tuple(proj_pts[0])
        cv2.line(img, origin, tuple(proj_pts[1]), (0,0,255), 3)  # X blue
        cv2.line(img, origin, tuple(proj_pts[2]), (0,255,0), 3)  # Y green
        cv2.line(img, origin, tuple(proj_pts[3]), (255,0,0), 3)  # Z red
    else:
        axes = np.float32([[60,0,0], [0,60,0], [0,0,60]]) @ R.T
        def p(i): return (int(nose[0] + axes[i][0]), int(nose[1] - axes[i][1]))
        cv2.line(img, nose, p(0), (0,0,255), 3)
        cv2.line(img, nose, p(1), (0,255,0), 3)
        cv2.line(img, nose, p(2), (255,0,0), 3)

def visualize(img, nose, result):
    """УНИВЕРСАЛЬНАЯ ВИЗУАЛИЗАЦИЯ ПО УГЛАМ"""
    rvec = result.get('rvec')
    tvec = result.get('tvec')
    K = result.get('K')
    dist = result.get('dist')

    if 'yaw' in result and 'pitch' in result and 'roll' in result:
        yaw, pitch, roll = result['yaw'], result['pitch'], result['roll']
    elif 'sin_b' in result:
        sin_b = result['sin_b']
        cos_minor = result['cos_minor']
        if sin_b == -8.0: return
        pitch = np.degrees(np.arcsin(sin_b))
        yaw = np.degrees(np.arccos(cos_minor)) if abs(cos_minor) <= 1 else 0
        roll = 0
        rvec = tvec = K = dist = None  # Force orthographic for geom
    else:
        return

    draw_perfect_cone_by_angles(img, nose, yaw, pitch, roll,
                                base_color=(0, 0, 0),
                                tip_color=(255, 255, 255),
                                rvec=rvec, tvec=tvec, K=K, dist=dist)