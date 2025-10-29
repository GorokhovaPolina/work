import numpy as np
import matplotlib.pyplot as plt
import time
import glob
import os
from estimator import MobileHeadPoseEstimator
from utils import (calculate_mae, calculate_rmse, accuracy_theta, mean_geodesic_distance)

def load_test_frames(frames_dir=None):
    """Загрузка отдельных кадров из директории (без видео)"""
    if frames_dir and os.path.isdir(frames_dir):
        image_files = sorted(glob.glob(os.path.join(frames_dir, '*.jpg')) + 
                             glob.glob(os.path.join(frames_dir, '*.png')))
        frames = [cv2.imread(f) for f in image_files if cv2.imread(f) is not None]
        return frames, image_files
    return [], None

def load_ground_truth_from_lst(lst_path='ground_truth.lst'):
    """Загрузка ground truth из .lst (формат: image_path yaw pitch roll)"""
    gt = []
    with open(lst_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 4:
                gt.append({'image_path': parts[0], 'yaw': float(parts[1]), 'pitch': float(parts[2]), 'roll': float(parts[3])})
    return gt

def match_gt_to_frames(gt, image_paths):
    """Сопоставление gt с кадрами по basename"""
    gt_dict = {os.path.basename(g['image_path']): {'yaw': g['yaw'], 'pitch': g['pitch'], 'roll': g['roll']} for g in gt}
    matched_gt = []
    for path in image_paths or []:
        basename = os.path.basename(path)
        if basename in gt_dict:
            matched_gt.append(gt_dict[basename])
        else:
            matched_gt.append(None)
    return matched_gt

def test_prototype():
    """
    Тест: обработка отдельных кадров (без видео/GUI)
    """
    # Пути
    frames_dir = '/path/to/frames/'  # Директория с отдельными .jpg/.png кадрами
    lst_path = 'ground_truth.lst'  # .lst с gt (опционально)
    
    estimator = MobileHeadPoseEstimator()
    test_frames, image_paths = load_test_frames(frames_dir=frames_dir)
    ground_truth_raw = load_ground_truth_from_lst(lst_path) if os.path.exists(lst_path) else None
    ground_truth = match_gt_to_frames(ground_truth_raw, image_paths) if ground_truth_raw and image_paths else None
    
    if not test_frames:
        print("Нет кадров для обработки.")
        return
    
    results = []
    deviations = []
    start_time = time.time()
    for i, frame in enumerate(test_frames):
        result = estimator.process_frame(frame)
        results.append(result)
        
        if 'coeffs' in result:
            coeffs = result['coeffs']
            print(f"Frame {i}: Coeffs={coeffs}")
            if coeffs[0] < 0.1 or coeffs[0] > 0.9 or abs(coeffs[1]) > 1:
                print("Verdict: Failed")
            else:
                print("Verdict: Passed")
        else:
            euler = result['euler']
            matrix = result['matrix']
            print(f"Frame {i}: Euler={euler}, Matrix=\n{matrix}")
        
        if ground_truth and i < len(ground_truth) and ground_truth[i]:
            gt = ground_truth[i]
            dev = {k: abs(euler[k] - gt[k]) for k in ['yaw', 'pitch', 'roll']}
            deviations.append(dev)
            print(f"Deviation for frame {i}: {dev}")
    
    total_time = time.time() - start_time
    fps = len(test_frames) / total_time if total_time > 0 else 0
    print(f"FPS: {fps}")
    
    if ground_truth and len(ground_truth) == len(results):
        eulers = [r['euler'] for r in results if 'euler' in r]
        filtered_gts = [g for g in ground_truth if g]
        filtered_preds = [eulers[i] for i in range(len(ground_truth)) if ground_truth[i]]
        
        mae_yaw = calculate_mae(filtered_preds, filtered_gts, 'yaw')
        mae_pitch = calculate_mae(filtered_preds, filtered_gts, 'pitch')
        mae_roll = calculate_mae(filtered_preds, filtered_gts, 'roll')
        mean_mae = calculate_mae(filtered_preds, filtered_gts)
        print(f"MAE: Yaw={mae_yaw:.2f}, Pitch={mae_pitch:.2f}, Roll={mae_roll:.2f}, Mean={mean_mae:.2f}")
        
        rmse_yaw = calculate_rmse(filtered_preds, filtered_gts, 'yaw')
        rmse_pitch = calculate_rmse(filtered_preds, filtered_gts, 'pitch')
        rmse_roll = calculate_rmse(filtered_preds, filtered_gts, 'roll')
        mean_rmse = calculate_rmse(filtered_preds, filtered_gts)
        print(f"RMSE: Yaw={rmse_yaw:.2f}, Pitch={rmse_pitch:.2f}, Roll={rmse_roll:.2f}, Mean={mean_rmse:.2f}")
        
        acc_theta = accuracy_theta(filtered_preds, filtered_gts, estimator.config.theta, mode='max')
        print(f"Accuracy_{estimator.config.theta}: {acc_theta:.2f}")
        
        mean_geo = mean_geodesic_distance(filtered_preds, filtered_gts)
        print(f"Mean Geodesic Distance: {mean_geo:.2f}")
        
        # Plot (не GUI, консольный, можно убрать)
        plt.plot([p['yaw'] for p in filtered_preds], label='Predicted Yaw')
        plt.plot([g['yaw'] for g in filtered_gts], label='GT Yaw')
        plt.legend()
        plt.show()
    
    import psutil
    print(f"Memory usage: {psutil.Process().memory_info().rss / (1024 * 1024)} MB")

if __name__ == "__main__":
    test_prototype()