import glob
import time
import psutil
from estimator import Estimator

def main():
    files = sorted(glob.glob("jsons/snapshot_*.json"))
    if not files:
        print("Нет JSON-файлов в папке jsons/")
        return

    est = Estimator(mode='pnp')  # pnp, coeffs, geometric
    print(f"[INFO] Найдено {len(files)} файлов\n")
    start = time.time()
    ok = 0

    for path in files:
        res = est.process(path)
        name = path.split('\\')[-1]

        if not res or 'error' in res:
            print(f"{name}: FAILED")
            continue

        if 'sin_b' in res:
            print(f"{name}: sin_b={res['sin_b']:.3f}, cos_minor={res['cos_minor']:.3f}")
        else:
            e = res
            print(f"{name}: Yaw={e['yaw']:+6.2f}°  Pitch={e['pitch']:+6.2f}°  Roll={e['roll']:+6.2f}°")
        ok += 1

    fps = len(files) / (time.time() - start)
    mem = psutil.Process().memory_info().rss / 1024**2
    print(f"\nУспешно: {ok}/{len(files)} | FPS: {fps:.1f} | Память: {mem:.1f} MB")

if __name__ == "__main__":
    main()