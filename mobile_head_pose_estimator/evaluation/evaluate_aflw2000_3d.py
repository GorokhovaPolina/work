import numpy as np
import json
import os
import glob
from pathlib import Path
import matplotlib.pyplot as plt
from estimator import MobileHeadPoseEstimator

def load_aflw2000_ground_truth(json_dir):
    """
    Загружает ground truth из созданных JSON файлов
    """
    gt_data = {}
    
    json_files = glob.glob(os.path.join(json_dir, "*.json"))
    
    for json_path in json_files:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        img_name = os.path.basename(json_path).replace('.json', '.jpg')
        
        gt_data[img_name] = {
            'yaw': data['pose_angles']['yaw'],
            'pitch': data['pose_angles']['pitch'],
            'roll': data['pose_angles']['roll'],
            'image_size': data['image_size']
        }
    
    return gt_data

def evaluate_pose_estimation(gt_data, json_dir, methods=['pnp', 'geom'], 
                           max_samples=None, save_errors=False):
    """
    Оценка качества оценки позы
    
    Аргументы:
        gt_data: словарь с ground truth
        json_dir: папка с JSON файлами
        methods: список методов для оценки ['pnp', 'geom', 'coeffs']
        max_samples: максимальное количество образцов
        save_errors: сохранять ли детальные ошибки
    
    Возвращает:
        dict с метриками
    """
    
    # Инициализируем эстиматоры
    estimators = {}
    for method in methods:
        estimators[method] = MobileHeadPoseEstimator(mode=method)
    
    # Результаты
    results = {method: {'yaw': [], 'pitch': [], 'roll': []} for method in methods}
    errors_detail = []
    
    # Обрабатываем изображения
    img_names = list(gt_data.keys())
    if max_samples:
        img_names = img_names[:max_samples]
    
    processed = 0
    skipped = 0
    
    for img_name in img_names:
        json_name = img_name.replace('.jpg', '.json')
        json_path = os.path.join(json_dir, json_name)
        
        if not os.path.exists(json_path):
            skipped += 1
            continue
        
        gt = gt_data[img_name]
        
        # Оценка каждым методом
        method_results = {}
        valid = True
        
        for method, estimator in estimators.items():
            try:
                result = estimator.process_json(json_path)
                
                if result and 'yaw' in result:
                    method_results[method] = {
                        'yaw': result['yaw'],
                        'pitch': result['pitch'],
                        'roll': result['roll']
                    }
                else:
                    valid = False
                    break
                    
            except Exception as e:
                print(f"Ошибка для {img_name} методом {method}: {e}")
                valid = False
                break
        
        if valid:
            # Сохраняем результаты
            for method in methods:
                if method in method_results:
                    results[method]['yaw'].append(method_results[method]['yaw'])
                    results[method]['pitch'].append(method_results[method]['pitch'])
                    results[method]['roll'].append(method_results[method]['roll'])
            
            # Сохраняем детальные ошибки
            if save_errors:
                error_entry = {'image': img_name, 'gt': gt}
                for method in methods:
                    if method in method_results:
                        for angle in ['yaw', 'pitch', 'roll']:
                            error = abs(method_results[method][angle] - gt[angle])
                            error_entry[f'{method}_{angle}_error'] = error
                            error_entry[f'{method}_{angle}_pred'] = method_results[method][angle]
                
                errors_detail.append(error_entry)
            
            processed += 1
        
        else:
            skipped += 1
        
        # Прогресс
        if processed % 100 == 0:
            print(f"Обработано: {processed} | Пропущено: {skipped}")
    
    print(f"\nВсего обработано: {processed} из {len(img_names)}")
    print(f"Пропущено: {skipped}")
    
    # Рассчитываем метрики
    metrics = {}
    
    for method in methods:
        if len(results[method]['yaw']) == 0:
            print(f"Предупреждение: метод {method} не дал результатов")
            continue
        
        metrics[method] = {}
        
        for angle in ['yaw', 'pitch', 'roll']:
            pred = np.array(results[method][angle])
            
            # Соответствующие ground truth
            gt_vals = []
            for i, img_name in enumerate(img_names):
                if i < len(pred):
                    gt_vals.append(gt_data[img_name][angle])
            
            gt_vals = np.array(gt_vals[:len(pred)])
            
            # Базовые метрики
            errors = np.abs(pred - gt_vals)
            
            metrics[method][f'{angle}_mae'] = float(np.mean(errors))
            metrics[method][f'{angle}_rmse'] = float(np.sqrt(np.mean(errors**2)))
            metrics[method][f'{angle}_std'] = float(np.std(errors))
            
            # Процентили ошибок
            for p in [25, 50, 75, 90, 95]:
                metrics[method][f'{angle}_p{p}'] = float(np.percentile(errors, p))
            
            # Точность при разных порогах
            for threshold in [5, 10, 15, 20, 30]:
                acc = np.mean(errors < threshold) * 100
                metrics[method][f'{angle}_acc@{threshold}'] = float(acc)
    
    # Сохраняем детальные ошибки
    if save_errors and errors_detail:
        with open('detailed_errors.json', 'w', encoding='utf-8') as f:
            json.dump(errors_detail, f, indent=2, ensure_ascii=False)
    
    return metrics, results, errors_detail if save_errors else None

def plot_comparison(metrics, output_dir="results"):
    """
    Визуализация сравнения методов
    """
    os.makedirs(output_dir, exist_ok=True)
    
    methods = list(metrics.keys())
    angles = ['yaw', 'pitch', 'roll']
    
    # 1. График MAE для каждого угла
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for idx, angle in enumerate(angles):
        ax = axes[idx]
        
        mae_values = []
        labels = []
        
        for method in methods:
            if f'{angle}_mae' in metrics[method]:
                mae_values.append(metrics[method][f'{angle}_mae'])
                labels.append(method.upper())
        
        if mae_values:
            bars = ax.bar(range(len(mae_values)), mae_values, color=['blue', 'orange', 'green'])
            ax.set_xticks(range(len(mae_values)))
            ax.set_xticklabels(labels)
            ax.set_ylabel('MAE (degrees)')
            ax.set_title(f'{angle.capitalize()} Error')
            ax.grid(True, alpha=0.3)
            
            # Добавляем значения на столбцы
            for bar, value in zip(bars, mae_values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                       f'{value:.2f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'mae_comparison.png'), dpi=150)
    plt.close()
    
    # 2. График точности при разных порогах
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    thresholds = [5, 10, 15, 20, 30]
    
    for idx, angle in enumerate(angles):
        ax = axes[idx]
        
        for method in methods:
            acc_values = []
            for thresh in thresholds:
                key = f'{angle}_acc@{thresh}'
                if key in metrics[method]:
                    acc_values.append(metrics[method][key])
            
            if acc_values:
                ax.plot(thresholds[:len(acc_values)], acc_values, 
                       marker='o', label=method.upper(), linewidth=2)
        
        ax.set_xlabel('Error Threshold (degrees)')
        ax.set_ylabel('Accuracy (%)')
        ax.set_title(f'{angle.capitalize()} Accuracy')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_ylim([0, 105])
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'accuracy_curves.png'), dpi=150)
    plt.close()
    
    # 3. Распределение ошибок (Box plot)
    # Нужны детальные ошибки для этого
    
    print(f"Графики сохранены в папку: {output_dir}")

def print_metrics_table(metrics):
    """
    Красивая таблица с результатами
    """
    print("\n" + "="*80)
    print("РЕЗУЛЬТАТЫ ОЦЕНКИ НА AFLW2000-3D")
    print("="*80)
    
    for method in metrics.keys():
        print(f"\n{method.upper()} METHOD:")
        print("-"*60)
        print(f"{'Angle':<10} {'MAE':<8} {'RMSE':<8} {'Std':<8} {'Acc@10°':<10} {'Acc@20°':<10}")
        print("-"*60)
        
        for angle in ['yaw', 'pitch', 'roll']:
            mae = metrics[method].get(f'{angle}_mae', 'N/A')
            rmse = metrics[method].get(f'{angle}_rmse', 'N/A')
            std = metrics[method].get(f'{angle}_std', 'N/A')
            acc10 = metrics[method].get(f'{angle}_acc@10', 'N/A')
            acc20 = metrics[method].get(f'{angle}_acc@20', 'N/A')
            
            if mae != 'N/A':
                print(f"{angle:<10} {mae:<8.2f} {rmse:<8.2f} {std:<8.2f} "
                      f"{acc10:<10.1f} {acc20:<10.1f}")
    
    print("\n" + "="*80)
    print("ЛУЧШИЙ РЕЗУЛЬТАТ ПО КАЖДОЙ МЕТРИКЕ:")
    print("-"*80)
    
    for angle in ['yaw', 'pitch', 'roll']:
        best_method = None
        best_mae = float('inf')
        
        for method in metrics.keys():
            mae = metrics[method].get(f'{angle}_mae', float('inf'))
            if mae < best_mae:
                best_mae = mae
                best_method = method
        
        if best_method:
            print(f"{angle.capitalize():<10} → {best_method.upper():<8} (MAE = {best_mae:.2f}°)")

def analyze_error_by_angle_range(gt_data, errors_detail, methods=['pnp', 'geom']):
    """
    Анализ ошибок в зависимости от угла поворота
    """
    angle_ranges = {
        'small': (-15, 15),
        'medium': (-45, 45),
        'large': (-90, 90)
    }
    
    analysis = {}
    
    for range_name, (min_angle, max_angle) in angle_ranges.items():
        analysis[range_name] = {}
        
        for method in methods:
            # Фильтруем ошибки по диапазону углов
            filtered_errors = []
            
            for error_entry in errors_detail:
                yaw_gt = error_entry['gt']['yaw']
                
                if min_angle <= yaw_gt <= max_angle:
                    for angle in ['yaw', 'pitch', 'roll']:
                        key = f'{method}_{angle}_error'
                        if key in error_entry:
                            filtered_errors.append(error_entry[key])
            
            if filtered_errors:
                analysis[range_name][method] = {
                    'mean_error': float(np.mean(filtered_errors)),
                    'std_error': float(np.std(filtered_errors)),
                    'count': len(filtered_errors)
                }
    
    return analysis

if __name__ == "__main__":
    # Настройки
    JSON_DIR = "aflw2000_json"
    OUTPUT_DIR = "evaluation_results"
    
    # Шаг 1: Загружаем ground truth
    print("Загрузка ground truth...")
    gt_data = load_aflw2000_ground_truth(JSON_DIR)
    print(f"Загружено {len(gt_data)} аннотаций")
    
    # Шаг 2: Оцениваем
    print("\nОценка качества...")
    methods_to_evaluate = ['pnp', 'geom']  # Можно добавить 'coeffs'
    
    metrics, results, errors_detail = evaluate_pose_estimation(
        gt_data, JSON_DIR, 
        methods=methods_to_evaluate,
        max_samples=None,  # Все изображения
        save_errors=True
    )
    
    # Шаг 3: Выводим результаты
    print_metrics_table(metrics)
    
    # Шаг 4: Визуализируем
    plot_comparison(metrics, OUTPUT_DIR)
    
    # Шаг 5: Анализ ошибок по диапазонам углов
    if errors_detail:
        angle_analysis = analyze_error_by_angle_range(gt_data, errors_detail, methods_to_evaluate)
        
        print("\n" + "="*80)
        print("АНАЛИЗ ОШИБОК ПО ДИАПАЗОНАМ УГЛОВ YAW:")
        print("="*80)
        
        for range_name, data in angle_analysis.items():
            print(f"\nДиапазон: {range_name}")
            print("-"*40)
            for method, stats in data.items():
                print(f"  {method.upper():<6} → Mean: {stats['mean_error']:.2f}° | "
                      f"Std: {stats['std_error']:.2f}° | Count: {stats['count']}")
    
    # Шаг 6: Сохраняем метрики
    with open(os.path.join(OUTPUT_DIR, 'final_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\nВсе результаты сохранены в папке: {OUTPUT_DIR}")