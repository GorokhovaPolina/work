import glob
import os
import time
import psutil
from estimator import MobileHeadPoseEstimator

def load_json_files(json_dir):
    if not os.path.isdir(json_dir):
        print(f"[ERROR] Папка не найдена: {json_dir}")
        return []
    files = sorted(glob.glob(os.path.join(json_dir, "snapshot_*.json")))
    if not files:
        print(f"[WARN] Нет snapshot_*.json в {json_dir}")
    return files

def main():
    json_dir = "jsons"
    estimator = MobileHeadPoseEstimator()
    json_files = load_json_files(json_dir)
    if not json_files:
        return

    print(f"[INFO] Найдено {len(json_files)} JSON-файлов\n")
    start_time = time.time()
    success_count = 0

    for i, json_path in enumerate(json_files):
        result = estimator.process_json(json_path)
        filename = os.path.basename(json_path)

        if result is None:
            print(f"{filename}: FAILED (no pose)")
            continue

        e = result['euler']
        print(f"{filename}: "
              f"Yaw={e['yaw']:+6.2f}°  "
              f"Pitch={e['pitch']:+6.2f}°  "
              f"Roll={e['roll']:+6.2f}°")
        success_count += 1

    total_time = time.time() - start_time
    fps = len(json_files) / total_time if total_time > 0 else 0
    mem_mb = psutil.Process().memory_info().rss / (1024 * 1024)

    print("\n" + "="*50)
    print(f"Обработано: {len(json_files)}")
    print(f"Успешно: {success_count}")
    print(f"FPS: {fps:.2f}")
    print(f"Память: {mem_mb:.1f} MB")
    print("="*50)

if __name__ == "__main__":
    main()
