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
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cr, sr = math.cos(roll), math.sin(roll)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    return Rz @ Ry @ Rx

def draw_perfect_cone_by_angles(img, nose, yaw_deg, pitch_deg, roll_deg,
                                length=180, radius=55, segments=64,
                                color=(0, 255, 255), gradient=True):
    R = euler_to_rotation_matrix(yaw_deg, pitch_deg, roll_deg)

    # ОСНОВАНИЕ НА НОСУ (z=0), ВЕРШИНА ВПЕРЕДИ (z=length)
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

    tip_pt = tuple(pts[0])      # ВЕРШИНА — в сторону взгляда
    base_pts = pts[1:]          # ОСНОВАНИЕ — на носу

    overlay = img.copy()

    # === ГРАДИЕНТНОЕ ОСНОВАНИЕ ===
    if gradient and len(base_pts) > 2:
        try:
            base_center = np.mean(base_pts, axis=0).astype(int)
            base_radius = max(10, int(0.9 * np.mean([np.linalg.norm(np.array(pt) - base_center) for pt in base_pts])))
            inner_color = tuple(max(0, c - 100) for c in color)
            outer_color = color

            gradient_mask = create_gradient_mask(img.shape[:2], base_center, base_radius, inner_color, outer_color)
            base_mask = np.zeros(img.shape[:2], dtype=np.uint8)
            cv2.fillPoly(base_mask, [base_pts], 255)
            gradient_region = cv2.bitwise_and(gradient_mask, gradient_mask, mask=base_mask)
            cv2.addWeighted(gradient_region, 0.75, overlay, 0.25, 0, overlay)
        except:
            cv2.fillPoly(overlay, [base_pts], color)
    else:
        cv2.fillPoly(overlay, [base_pts], color)

    # === ГРАНИ С ГРАДИЕНТОМ ТОЛЩИНЫ ===
    for i, pt in enumerate(base_pts):
        t = i / len(base_pts)
        thickness = max(1, int(4 * (1 - t**0.7)))
        line_color = tuple(int(c * (1 - 0.4 * t)) for c in color)
        cv2.line(overlay, tip_pt, tuple(pt), line_color, thickness)

    # === КОНТУР ОСНОВАНИЯ ===
    bright_color = tuple(min(255, c + 60) for c in color)
    cv2.polylines(overlay, [base_pts], True, bright_color, 3)

    # === НАЛОЖЕНИЕ ===
    cv2.addWeighted(overlay, 0.65, img, 0.35, 0, img)

    # === СВЕЧЕНИЕ ВЕРШИНЫ ===
    glow = img.copy()
    cv2.circle(glow, tip_pt, 10, (255, 255, 200), -1)
    cv2.addWeighted(glow, 0.3, img, 0.7, 0, img)
    cv2.circle(img, tip_pt, 4, (255, 255, 255), -1)

    # === ОСИ ===
    axes = np.float32([[60,0,0], [0,60,0], [0,0,60]]) @ R.T
    def p(i): return (int(nose[0] + axes[i][0]), int(nose[1] - axes[i][1]))
    cv2.line(img, nose, p(0), (0,0,255), 3)
    cv2.line(img, nose, p(1), (0,255,0), 3)
    cv2.line(img, nose, p(2), (255,0,0), 3)

    # === ПОДПИСЬ ===
    text = f"Yaw: {yaw_deg:+.1f}°  Pitch: {pitch_deg:+.1f}°  Roll: {roll_deg:+.1f}°"
    cv2.putText(img, text, (10, 35), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 0), 2)
    cv2.putText(img, "Head Pose Estimation", (10, img.shape[0]-15), cv2.FONT_HERSHEY_DUPLEX, 0.8, (200, 255, 255), 2)

    return img

def visualize(img, nose, result):
    """УНИВЕРСАЛЬНАЯ ВИЗУАЛИЗАЦИЯ ПО УГЛАМ"""
    if img is None:
        return None
    
    if 'yaw' in result and 'pitch' in result and 'roll' in result:
        yaw, pitch, roll = result['yaw'], result['pitch'], result['roll']
    elif 'sin_b' in result:
        sin_b = result['sin_b']
        cos_minor = result['cos_minor']
        if sin_b == -8.0:
            return None
        pitch = np.degrees(np.arcsin(sin_b))
        yaw = np.degrees(np.arccos(cos_minor)) if abs(cos_minor) <= 1 else 0
        roll = 0
    else:
        return None
    
    img_out = draw_perfect_cone_by_angles(img.copy(), nose, yaw, pitch, roll)
    if img_out.dtype != np.uint8:
        img_out = np.clip(img_out, 0, 255).astype(np.uint8)
    return img_out

    # draw_perfect_cone_by_angles(img, nose, yaw, pitch, roll)