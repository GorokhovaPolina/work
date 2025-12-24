import os, sys
import glob
import numpy as np
import math
import scipy.io
from estimator import MobileHeadPoseEstimator

AFLW2000_DIR = "../AFLW2000" 

LEFT_EYE_OUTER_IDX = 36
LEFT_EYE_INNER_IDX = 39
RIGHT_EYE_OUTER_IDX = 45
RIGHT_EYE_INNER_IDX = 42
NOSE_TIP_IDX = 33

IMG_WIDTH = 450
IMG_HEIGHT = 450

def extract_landmarks_from_mat(mat):
    if 'pt3d_68' in mat:
        pts = mat['pt3d_68'][:2, :]
    elif 'pt2d' in mat:
        pts = mat['pt2d']
    else:
        return None

    x = pts[0, :]
    y = pts[1, :]

    left_eye_x = (x[LEFT_EYE_OUTER_IDX] + x[LEFT_EYE_INNER_IDX]) / 2.0
    left_eye_y = (y[LEFT_EYE_OUTER_IDX] + y[LEFT_EYE_INNER_IDX]) / 2.0

    right_eye_x = (x[RIGHT_EYE_OUTER_IDX] + x[RIGHT_EYE_INNER_IDX]) / 2.0
    right_eye_y = (y[RIGHT_EYE_OUTER_IDX] + y[RIGHT_EYE_INNER_IDX]) / 2.0

    nose_x = x[NOSE_TIP_IDX]
    nose_y = y[NOSE_TIP_IDX]

    return {
        'left_eye': (left_eye_x, left_eye_y),
        'right_eye': (right_eye_x, right_eye_y),
        'nose': (nose_x, nose_y)
    }

def get_ground_truth_from_mat(mat):
    if 'Pose_Para' not in mat:
        return None
    pose_para = mat['Pose_Para'].flatten()
    pitch_rad, yaw_rad, roll_rad = pose_para[:3]
    return {
        'pitch': math.degrees(pitch_rad),
        'yaw': math.degrees(yaw_rad),
        'roll': math.degrees(roll_rad)
    }

def main():
    estimator = MobileHeadPoseEstimator(mode='pnp')
    mat_files = sorted(glob.glob(os.path.join(AFLW2000_DIR, "*.mat")))
    if not mat_files:
        print("Ошибка - нет мат файлов")
        return

    total_samples = len(mat_files)
    print(f"Найдено {total_samples} сэмплов в AFLW2000.")

    # Счетчики ошибок
    geom_errors = {'yaw': [], 'pitch': [], 'roll': []}
    pnp_errors = {'yaw': [], 'pitch': [], 'roll': []}
    skipped = 0

    for mat_path in mat_files:
        filename = os.path.basename(mat_path).replace('.mat', '')
        
        try:
            mat = scipy.io.loadmat(mat_path)
        except Exception as e:
            print(f"Ошибка загрузки {filename}: {e}")
            skipped += 1
            continue

        landmarks = extract_landmarks_from_mat(mat)
        if landmarks is None:
            print(f"Нет лендмарков в {filename}")
            skipped += 1
            continue

        gt = get_ground_truth_from_mat(mat)
        if gt is None:
            print(f"Нет ground truth в {filename}")
            skipped += 1
            continue

        # Подготавливаем data в формате вашего кода
        data = {
            'image_size': (IMG_WIDTH, IMG_HEIGHT),
            'landmarks': landmarks
        }

        # Вычисляем позу
        result = estimator.calculator.calculate_pose(data, mode='both')

        # Парсим результаты
        if 'geom' not in result or 'pnp' not in result:
            print(f"Ошибка расчета для {filename}")
            skipped += 1
            continue

        geom_result = result['geom']
        pnp_result = result['pnp']

        # Для geom всегда есть, для pnp проверяем success
        if 'error' in geom_result:
            print(f"Geom ошибка в {filename}: {geom_result['error']}")
            continue

        # Сбор ошибок для geom
        geom_errors['yaw'].append(abs(geom_result['yaw'] - gt['yaw']))
        geom_errors['pitch'].append(abs(geom_result['pitch'] - gt['pitch']))
        geom_errors['roll'].append(abs(geom_result['roll'] - gt['roll']))

        # Для pnp
        if pnp_result.get('success', False):
            pnp_errors['yaw'].append(abs(pnp_result['yaw'] - gt['yaw']))
            pnp_errors['pitch'].append(abs(pnp_result['pitch'] - gt['pitch']))
            pnp_errors['roll'].append(abs(pnp_result['roll'] - gt['roll']))
        else:
            print(f"PnP failed для {filename}: {pnp_result.get('error', 'Unknown')}")

    # Вычисляем MAE
    processed = total_samples - skipped

    if processed == 0:
        print("Нет обработанных сэмплов.")
        return

    # MAE для geom (всегда полный, предполагаем)
    geom_mae_yaw = np.mean(geom_errors['yaw'])
    geom_mae_pitch = np.mean(geom_errors['pitch'])
    geom_mae_roll = np.mean(geom_errors['roll'])
    geom_mae_avg = (geom_mae_yaw + geom_mae_pitch + geom_mae_roll) / 3.0

    # MAE для pnp (может быть меньше сэмплов, если failures)
    pnp_count = len(pnp_errors['yaw'])
    if pnp_count > 0:
        pnp_mae_yaw = np.mean(pnp_errors['yaw'])
        pnp_mae_pitch = np.mean(pnp_errors['pitch'])
        pnp_mae_roll = np.mean(pnp_errors['roll'])
        pnp_mae_avg = (pnp_mae_yaw + pnp_mae_pitch + pnp_mae_roll) / 3.0
    else:
        pnp_mae_yaw = pnp_mae_pitch = pnp_mae_roll = pnp_mae_avg = float('nan')

    # Вывод результатов
    print(f"Обработано: {processed}/{total_samples} (пропущено: {skipped})")
    print("\nGeometric метод:")
    print(f"  MAE Yaw:   {geom_mae_yaw:.2f}°")
    print(f"  MAE Pitch: {geom_mae_pitch:.2f}°")
    print(f"  MAE Roll:  {geom_mae_roll:.2f}°")
    print(f"  Средний MAE: {geom_mae_avg:.2f}°")

    print("\nPnP метод (успешных: {pnp_count}):")
    print(f"  MAE Yaw:   {pnp_mae_yaw:.2f}°")
    print(f"  MAE Pitch: {pnp_mae_pitch:.2f}°")
    print(f"  MAE Roll:  {pnp_mae_roll:.2f}°")
    print(f"  Средний MAE: {pnp_mae_avg:.2f}°")

if name == "__main__":
    main()