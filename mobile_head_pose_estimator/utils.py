import numpy as np
from scipy import optimize
from sklearn.linear_model import RANSACRegressor
import math
from scipy.spatial.transform import Rotation as R

def find_rotation_coeffs(left_eye, right_eye, nose):
    """
    Вычисление коэффициентов как в rotation_checker.cpp (sinB, cosMinor)
    """
    a = np.array([left_eye[0], -left_eye[1]], dtype=np.float32)
    b = np.array([right_eye[0], -right_eye[1]], dtype=np.float32)
    c = np.array([nose[0], -nose[1]], dtype=np.float32)

    def l2(x, y):
        return math.sqrt(x**2 + y**2)

    def get_cos(x1, y1, x2, y2):
        return (x1 * x2 + y1 * y2) / (l2(x1, y1) * l2(x2, y2))

    # Проверка расположения
    if a[0] >= b[0] - 1 or (c[1] >= a[1] and c[1] >= b[1]):
        return (-8.0, -8.0)

    # cosMinor
    cos_minor = get_cos(c[0] - a[0], c[1] - a[1], c[0] - b[0], c[1] - b[1])
    sin_minor = math.sqrt(1 - cos_minor**2)

    # Перемещения
    c -= a
    b -= a
    a = np.array([0.0, 0.0])

    cos_a = (c[0] * b[0] + c[1] * b[1]) / (l2(c[0], c[1]) * l2(b[0], b[1]))
    ac = l2(c[0], c[1])

    c -= b
    a -= b
    b = np.array([0.0, 0.0])

    cos_b = (c[0] * a[0] + c[1] * a[1]) / (l2(c[0], c[1]) * l2(a[0], a[1]))

    bc = l2(c[0], c[1])
    ah = bc * cos_b
    ab = l2(a[0], a[1])

    if cos_a < 0 or cos_b < 0:
        return (-8.0, -8.0)

    sin_b = ah / ab

    return (sin_b, cos_minor)

def calculate_mae(preds, gts, key=None):
    """MAE для ключа или средний по всем (если key=None)"""
    if key:
        return np.mean([abs(p[key] - g[key]) for p, g in zip(preds, gts)])
    else:
        mae_yaw = calculate_mae(preds, gts, 'yaw')
        mae_pitch = calculate_mae(preds, gts, 'pitch')
        mae_roll = calculate_mae(preds, gts, 'roll')
        return (mae_yaw + mae_pitch + mae_roll) / 3

def calculate_rmse(preds, gts, key=None):
    """RMSE для ключа или средний по всем"""
    if key:
        return np.sqrt(np.mean([(p[key] - g[key])**2 for p, g in zip(preds, gts)]))
    else:
        rmse_yaw = calculate_rmse(preds, gts, 'yaw')
        rmse_pitch = calculate_rmse(preds, gts, 'pitch')
        rmse_roll = calculate_rmse(preds, gts, 'roll')
        return (rmse_yaw + rmse_pitch + rmse_roll) / 3

def accuracy_theta(preds, gts, theta, mode='max'):
    """Accuracy_θ: доля где overall_error <= theta (mode='max' или 'mean')"""
    count = 0
    for p, g in zip(preds, gts):
        errors = [abs(p[k] - g[k]) for k in ['yaw', 'pitch', 'roll']]
        err = max(errors) if mode == 'max' else np.mean(errors)
        if err <= theta:
            count += 1
    return count / len(preds) if len(preds) > 0 else 0

def euler_to_rot_matrix(yaw, pitch, roll):
    """Конвертация Euler (degrees) в rotation matrix"""
    r = R.from_euler('xyz', [yaw, pitch, roll], degrees=True)
    return r.as_matrix()

def geodesic_distance(R_gt, R_pred):
    """Geodesic Distance в градусах"""
    trace = np.trace(np.dot(R_gt.T, R_pred))
    return np.degrees(np.arccos(np.clip((trace - 1) / 2, -1, 1)))

def mean_geodesic_distance(preds, gts):
    """Средний Geodesic Distance по всем кадрам"""
    dists = []
    for p, g in zip(preds, gts):
        R_p = euler_to_rot_matrix(p['yaw'], p['pitch'], p['roll'])
        R_g = euler_to_rot_matrix(g['yaw'], g['pitch'], g['roll'])
        dists.append(geodesic_distance(R_g, R_p))
    return np.mean(dists) if dists else 0

def kalman_filter_update(state, measurement, Q=0.1, R=0.1):
    """Простой 1D Kalman filter для сглаживания углов"""
    prediction = state[0] + state[1]
    prediction_error = Q
    kalman_gain = prediction_error / (prediction_error + R)
    updated_value = prediction + kalman_gain * (measurement - prediction)
    updated_velocity = state[1]
    return np.array([updated_value, updated_velocity])