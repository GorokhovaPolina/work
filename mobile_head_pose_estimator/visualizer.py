import cv2
import numpy as np

def draw_axes(img, nose, scale=50):
    cv2.line(img, nose, (nose[0] + scale, nose[1]), (0, 0, 255), 2)     # X (Yaw) — красная
    cv2.line(img, nose, (nose[0], nose[1] - scale), (0, 255, 0), 2)     # Y (Pitch) — зелёная
    cv2.line(img, nose, (nose[0] - scale, nose[1]), (255, 0, 0), 2)     # Z (Roll) — синяя

def draw_cone(img, nose, rvec, tvec, K, dist, length=80, radius=25, color=(0, 255, 255)):
    cone_3d = np.float32([
        [0, 0, 0],
        [-radius, -radius, length],
        [ radius, -radius, length],
        [ radius,  radius, length],
        [-radius,  radius, length]
    ])
    pts, _ = cv2.projectPoints(cone_3d, rvec, tvec, K, dist)
    pts = np.int32(pts).reshape(-1, 2)
    nose_pt = tuple(pts[0])
    base = pts[1:]
    cv2.fillPoly(img, [base], color)
    for i in range(4):
        cv2.line(img, nose_pt, tuple(base[i]), color, 2)
        cv2.line(img, tuple(base[i]), tuple(base[(i+1)%4]), color, 2)

def visualize(img, nose, result):
    if 'rvec' not in result or 'K' not in result:
        return
    draw_axes(img, nose)
    draw_cone(img, nose, result['rvec'], result['tvec'], result['K'], result.get('dist', np.zeros((4,1))))