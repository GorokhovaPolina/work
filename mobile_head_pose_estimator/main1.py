import glob
import os
import json
import cv2
import numpy as np
from estimator import MobileHeadPoseEstimator
from visualizer import visualize
from pose_calculator import GeometricPoseCalculator

def print_coeffs_result(name, result, gt=None):
    sin_b = result.get('sin_b', -8.0)
    cos_minor = result.get('cos_minor', -8.0)
    
    if sin_b == -8.0:
        print(f"{name:<20} | coeffs → FAILED (лицо не фронтально или ошибка)")
        return False
    
    print(f"{name:<20} | coeffs → sin_b={sin_b:+.4f}, cos_minor={cos_minor:+.4f}")
    
    if gt:
        # Приблизительно: sin_b ≈ sin(pitch), cos_minor ≈ cos(yaw)
        yaw_est = np.degrees(np.arccos(cos_minor)) if abs(cos_minor) <= 1 else 999
        pitch_est = np.degrees(np.arcsin(sin_b))
        mae = abs(yaw_est - gt['yaw']) + abs(pitch_est - gt['pitch'])
        mae /= 2
        status = "GOOD" if mae < 15 else "BAD"
        print(f"{'':<20} | → Yaw≈{yaw_est:+6.2f}° | Pitch≈{pitch_est:+6.2f}° | MAE≈{mae:5.2f}° [{status}]")
    return True

def print_pnp_result(name, result, gt=None):
    yaw = result['yaw']
    pitch = result['pitch']
    roll = result['roll']
    
    if gt:
        err_y = abs(yaw - gt['yaw'])
        err_p = abs(pitch - gt['pitch'])
        err_r = abs(roll - gt['roll'])
        mae = (err_y + err_p + err_r) / 3
        status = "EXCELLENT" if mae < 2 else "GOOD" if mae < 5 else "WARNING"
        print(f"{name:<20} | PnP    → Yaw: {yaw:+6.2f}° (Δ{err_y:4.2f})")
        print(f"{'':<20} |          Pitch: {pitch:+6.2f}° (Δ{err_p:4.2f})")
        print(f"{'':<20} |          Roll:  {roll:+6.2f}° (Δ{err_r:4.2f})")
        print(f"{'':<20} | → MAE = {mae:.2f}° [{status}]")
    else:
        print(f"{name:<20} | PnP    → Yaw: {yaw:+6.2f}° | Pitch: {pitch:+6.2f}° | Roll: {roll:+6.2f}°")
    return True

def main():
    # ДВА ЭСТИМАТОРА
    estimator_coeffs = MobileHeadPoseEstimator(mode='coeffs')
    estimator_pnp = MobileHeadPoseEstimator(mode='pnp')

    json_files = sorted(glob.glob("jsons/snapshot_*.json"))
    if not json_files:
        print("ОШИБКА: Нет JSON в jsons/")
        return

    os.makedirs("output", exist_ok=True)
    total = len(json_files)
    coeffs_ok = pnp_ok = 0

    for json_path in json_files:
        filename = os.path.basename(json_path).replace('.json', '')
        
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except:
            print(f"{filename} → JSON ошибка")
            continue

        gt = data.get("ground_truth")

        # === coeffs ===
        result_coeffs = estimator_coeffs.process_json(json_path)
        coeffs_success = print_coeffs_result(filename, result_coeffs or {}, gt)
        if coeffs_success: coeffs_ok += 1

        # === PnP ===
        result_pnp = estimator_pnp.process_json(json_path)
        pnp_success = print_pnp_result(filename, result_pnp or {}, gt)
        if pnp_success: pnp_ok += 1

        calculator = GeometricPoseCalculator()
        result_geom = calculator.calculate_pose(data)

        # === ВИЗУАЛИЗАЦИЯ ===
        img_path = json_path.replace('jsons', '../mydataset').replace('.json', '.jpg')
        if os.path.exists(img_path) and result_geom:
            img = cv2.imread(img_path)
            nose = tuple(map(int, data['props']['kp_nose_tip']))
            visualize(img, nose, result_pnp)
            out = f"output/{filename}_vis.jpg"
            cv2.imwrite(out, img)
            print(f"   КОНУС: {out} (по направлению головы)\n")
        else:
            print(f"   Нет изображения или PnP failed\n")

    # === ИТОГИ ===
    print("="*80)
    print(f"ГОТОВО! Обработано: {total}")
    print(f"   coeffs         → успешно: {coeffs_ok}/{total}")
    print(f"   PnP (solvePnP) → успешно: {pnp_ok}/{total}")
    print("="*80)

if __name__ == "__main__":
    main()