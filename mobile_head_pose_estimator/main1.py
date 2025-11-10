# main1.py
import glob
import os
import time
import psutil
import cv2
import numpy as np
import json
from estimator import MobileHeadPoseEstimator

def main():
    json_dir = "jsons"
    frames_dir = "../mydataset"
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    estimator = MobileHeadPoseEstimator(mode='pnp')

    files = sorted(glob.glob(os.path.join(json_dir, "snapshot_*.json")))
    if not files:
        print("Нет JSON-файлов")
        return

    print(f"[INFO] Найдено {len(files)} файлов\n")
    start = time.time()
    ok = 0

    for json_path in files:
        # === ЗАГРУЗКА JSON ===
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            print(f"[ERROR] JSON: {e}")
            continue

        result = estimator.process_json(json_path)
        name = os.path.basename(json_path)

        if not result or 'error' in result:
            print(f"{name}: FAILED")
            continue

        if 'sin_b' in result:
            print(f"{name}: sin_b={result['sin_b']:.3f}, cos_minor={result['cos_minor']:.3f}")
        else:
            e = result
            print(f"{name}: Yaw={e['yaw']:+6.2f}°  Pitch={e['pitch']:+6.2f}°  Roll={e['roll']:+6.2f}°")

        # === ВИЗУАЛИЗАЦИЯ ===
        img_name = name.replace('.json', '.jpg')
        img_path = os.path.join(frames_dir, img_name)
        if not os.path.exists(img_path):
            print(f"[WARN] Нет изображения: {img_name}")
            ok += 1
            continue

        img = cv2.imread(img_path)
        if img is None:
            print(f"[WARN] Не загрузилось: {img_name}")
            ok += 1
            continue

        # Нос
        try:
            nose = tuple(map(int, data['props']['kp_nose_tip']))
        except KeyError:
            print(f"[WARN] Нет kp_nose_tip")
            ok += 1
            continue

        # === Оси ===
        scale = 50
        cv2.line(img, nose, (nose[0] + scale, nose[1]), (0, 0, 255), 2)     # X (Yaw) — красная
        cv2.line(img, nose, (nose[0], nose[1] - scale), (0, 255, 0), 2)     # Y (Pitch) — зелёная
        cv2.line(img, nose, (nose[0] - scale, nose[1]), (255, 0, 0), 2)     # Z (Roll) — синяя

        # === Вектор направления (куда смотрит голова) ===
        if 'rvec' in result and 'tvec' in result and 'K' in result:
            rvec = result['rvec']
            tvec = result['tvec']
            K = result['K']
            dist = result.get('dist', np.zeros((4,1)))

            axis = np.float32([[0, 0, 1000]])  # точка вперёд
            imgpts, _ = cv2.projectPoints(axis, rvec, tvec, K, dist)
            end = tuple(map(int, imgpts[0][0]))
            cv2.arrowedLine(img, nose, end, (255, 255, 0), 2, tipLength=0.2)

        # Сохранение
        out_path = os.path.join(output_dir, name.replace('.json', '_vis.jpg'))
        cv2.imwrite(out_path, img)
        print(f"[INFO] Сохранено: {out_path}")

        ok += 1

    fps = len(files) / (time.time() - start)
    mem = psutil.Process().memory_info().rss / 1024**2
    print(f"\nУспешно: {ok}/{len(files)} | FPS: {fps:.1f} | Память: {mem:.1f} MB")

if __name__ == "__main__":
    main()
