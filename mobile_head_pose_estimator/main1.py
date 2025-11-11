# main.py
import glob
import os
import json
import cv2
import numpy as np
from estimator import MobileHeadPoseEstimator
from visualizer import visualize

def print_header():
    print("\n" + "="*60)
    print("    HPE EVALUATION — Mobile Head Pose Estimator")
    print("    Yaw (поворот влево/вправо) | Pitch (вверх/вниз) | Roll (наклон)")
    print("="*60)

def print_result(name, result, gt=None):
    yaw = result['yaw']
    pitch = result['pitch']
    roll = result['roll']

    if gt:
        err_y = abs(yaw - gt['yaw'])
        err_p = abs(pitch - gt['pitch'])
        err_r = abs(roll - gt['roll'])
        mae = (err_y + err_p + err_r) / 3
        status = "GOOD" if mae < 5 else "WARNING"
        print(f"{name:<20} | Yaw: {yaw:+6.2f}° (GT: {gt['yaw']:+6.2f} → Δ{err_y:4.2f}°)")
        print(f"{'':<20} | Pitch: {pitch:+6.2f}° (GT: {gt['pitch']:+6.2f} → Δ{err_p:4.2f}°)")
        print(f"{'':<20} | Roll:  {roll:+6.2f}° (GT: {gt['roll']:+6.2f} → Δ{err_r:4.2f}°)")
        print(f"{'':<20} | → MAE = {mae:.2f}° [{status}]")
    else:
        print(f"{name:<20} | Yaw: {yaw:+6.2f}° | Pitch: {pitch:+6.2f}° | Roll: {roll:+6.2f}°")

    print("-" * 60)

def main():
    print_header()

    estimator = MobileHeadPoseEstimator(mode='pnp')
    json_files = sorted(glob.glob("jsons/snapshot_*.json"))
    
    if not json_files:
        print("ОШИБКА: Нет JSON-файлов в папке 'jsons/'")
        return

    os.makedirs("output", exist_ok=True)
    success = 0

    for json_path in json_files:
        filename = os.path.basename(json_path).replace('.json', '')
        
        # === Читаем JSON ===
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            print(f"{filename} → ОШИБКА чтения JSON: {e}")
            continue

        # === Оценка ===
        result = estimator.process_json(json_path)
        if not result or 'error' in result:
            print(f"{filename} → FAILED (no pose)")
            continue

        # === Выводим результат ===
        print_result(filename, result)

        # === Визуализация ===
        img_path = json_path.replace('jsons', 'frames').replace('.json', '.jpg')
        if not os.path.exists(img_path):
            print(f"   ВНИМАНИЕ: Нет изображения: {img_path}")
            success += 1
            continue

        img = cv2.imread(img_path)
        if img is None:
            print(f"   ВНИМАНИЕ: Не загрузилось изображение")
            success += 1
            continue

        nose = tuple(map(int, data['props']['kp_nose_tip']))
        visualize(img, nose, result)

        out_path = f"output/{filename}_vis.jpg"
        cv2.imwrite(out_path, img)
        print(f"   Визуализация: {out_path} (ЯРКИЙ КОНУС → направление головы)")
        
        success += 1

    print(f"\nГОТОВО! Обработано: {success}/{len(json_files)}")
    print("   Конусы сохранены в папке 'output/'")
    print("   Использован метод: PnP (solvePnP + 3D модель)")
    print("   coeffs — НЕ ИСПОЛЬЗУЕТСЯ (ломается на углах >30°)")
    print("="*60)

if __name__ == "__main__":
    main()