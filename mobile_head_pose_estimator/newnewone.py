import glob
import os
import json
import cv2
import numpy as np
from estimator import MobileHeadPoseEstimator
from visualizer1 import visualize
from pose_calculator import GeometricPoseCalculator

def get_user_input():
    """Получаем все пути от пользователя"""
    print("="*60)
    print("НАСТРОЙКА ПУТЕЙ:")
    print("="*60)
    
    # Папка с JSON разметкой ключевых точек
    markup_base_path = "../" + input("Путь к папке с разметкой ключевых точек: ").strip()
    if not os.path.exists(markup_base_path):
        print(f"ОШИБКА: Папка '{markup_base_path}' не существует!")
        return None
    
    # Паттерн поиска JSON файлов внутри базовой папки
    markup_pattern = input("Паттерн поиска JSON файлов [runlist_item_*/clip_*/snapshot_*.json]: ").strip()
    if not markup_pattern:
        markup_pattern = "runlist_item_*/clip_*/snapshot_*.json"
    
    markup_path = os.path.join(markup_base_path, markup_pattern)
    
    # Папка с ground truth
    gt_path = input("Путь к папке с разметкой УГЛОВ: ").strip()
    if not gt_path or not os.path.exists(gt_path):
        print(f"ВНИМАНИЕ: Папка '{gt_path}' не существует. MAE не будет считаться.")
        calculate_mae = False
    else:
        calculate_mae = True

    # Папка для сохранения углов для сравнения
    angles_output = input("Путь для сохранения углов для сравнения [/res_jsons]: ").strip()
    if angles_output:
        os.makedirs(angles_output, exist_ok=True)
    else:
        angles_output = "res_jsons"
        os.makedirs(angles_output, exist_ok=True)
    
    # Визуализация
    visualize_choice = input("Сохранять визуализацию? (y/n) [n]: ").strip().lower()
    if visualize_choice == 'y':
        # Папка с изображениями
        images_path = input("Путь к папке с изображениями: ").strip()
        if not images_path or not os.path.exists(images_path):
            print(f"ОШИБКА: Папка '{images_path}' не существует!")
            return None
        visual_output = input("Путь для сохранения визуализации [/visual_output]: ").strip()
        if not visual_output:
            visual_output = "visual_output"
        os.makedirs(visual_output, exist_ok=True)
    else:
        visual_output = None
        images_path = None
    
    print("="*60)
    
    return {
        'markup_base_path': markup_base_path,
        'markup_pattern': markup_pattern,
        'markup_path': markup_path,
        'gt_path': gt_path,
        'angles_output': angles_output,
        'images_path': images_path,
        'visual_output': visual_output,
        'calculate_mae': calculate_mae
    }

def print_coeffs_result(name, result, gt=None):
    sin_b = result.get('sin_b', -8.0)
    cos_minor = result.get('cos_minor', -8.0)
    
    if sin_b == -8.0:
        print(f"{name:<30} | coeffs → FAILED (лицо не фронтально или ошибка)")
        return False
    
    print(f"{name:<30} | coeffs → sin_b={sin_b:+.4f}, cos_minor={cos_minor:+.4f}")
    
    if gt:
        yaw_est = np.degrees(np.arccos(cos_minor)) if abs(cos_minor) <= 1 else 999
        pitch_est = np.degrees(np.arcsin(sin_b))
        mae = abs(yaw_est - gt['yaw']) + abs(pitch_est - gt['pitch'])
        mae /= 2
        status = "GOOD" if mae < 15 else "BAD"
        print(f"{'':<30} | → Yaw≈{yaw_est:+6.2f}° | Pitch≈{pitch_est:+6.2f}° | MAE≈{mae:5.2f}° [{status}]")
    return True

def print_pnp_result(name, result, gt=None):
    yaw = result['yaw']
    pitch = result['pitch']
    roll = result['roll']
    mae = 0
    if gt:
        err_y = abs(yaw - gt['yaw'])
        err_p = abs(pitch - gt['pitch'])
        err_r = abs(roll - gt['roll'])
        mae = (err_y + err_p + err_r) / 3
        status = "EXCELLENT" if mae < 2 else "GOOD" if mae < 5 else "WARNING"
        print(f"{name:<30} | PnP    → Yaw: {yaw:+6.2f}° (Δ{err_y:4.2f})")
        print(f"{'':<30} |          Pitch: {pitch:+6.2f}° (Δ{err_p:4.2f})")
        print(f"{'':<30} |          Roll:  {roll:+6.2f}° (Δ{err_r:4.2f})")
        print(f"{'':<30} | → MAE = {mae:.2f}° [{status}]")
    else:
        print(f"{name:<30} | PnP    → Yaw: {yaw:+6.2f}° | Pitch: {pitch:+6.2f}° | Roll: {roll:+6.2f}°")
    return mae

def save_angles_for_comparison(angles_output, filename, result_pnp):
    """Сохраняем углы"""
    if not angles_output:
        return
    
    angles_data = {
        'filename': filename,
        'head_pose': {
            'yaw': float(result_pnp['yaw']),
            'pitch': float(result_pnp['pitch']),
            'roll': float(result_pnp['roll'])
        }
    }
    
    angles_file = os.path.join(angles_output, f"{filename}_angles.json")
    with open(angles_file, 'w', encoding='utf-8') as f:
        json.dump(angles_data, f, indent=2, ensure_ascii=False)
    
    print(f"   Angles saved: {angles_file}")

def extract_unique_filename(json_path, base_path):
    """Извлекаем уникальное имя файла из полного пути"""
    # Убираем базовый путь
    relative_path = os.path.relpath(json_path, base_path)
    # Заменяем слеши на подчеркивания для создания уникального имени
    unique_name = relative_path.replace(os.sep, '_').replace('.json', '')
    return unique_name

def load_all_ground_truth(gt_path):
    """Загружаем ВСЕ ground truth файлы из папки"""
    if not gt_path or not os.path.exists(gt_path):
        return []
    
    gt_files = sorted(glob.glob(os.path.join(gt_path, "*.json")))
    gt_data_list = []
    
    print(f"Найдено {len(gt_files)} файлов ground truth")
    
    for gt_file in gt_files:
        try:
            with open(gt_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Извлекаем углы из разных возможных структур
            angles = None
            if 'ground_truth' in data:
                angles = data['ground_truth']
            elif 'head_pose' in data:
                angles = data['head_pose']
            elif all(key in data for key in ['yaw', 'pitch', 'roll']):
                angles = data
            elif 'yaw' in data and 'pitch' in data and 'roll' in data:
                angles = data
            
            if angles:
                gt_data_list.append({
                    'file': gt_file,
                    'filename': os.path.basename(gt_file),
                    'angles': angles
                })
                print(f"  Загружен: {os.path.basename(gt_file)}")
            else:
                print(f"  Пропущен (неправильный формат): {os.path.basename(gt_file)}")
                
        except Exception as e:
            print(f"  Ошибка загрузки {os.path.basename(gt_file)}: {e}")
            continue
    
    return gt_data_list

def find_corresponding_image(images_path, unique_name):
    """Ищем соответствующее изображение по уникальному имени"""
    if not images_path:
        return None
    
    # Пробуем разные варианты
    possible_names = [
        unique_name.replace('_snapshot_', '_') + '.jpg',
        unique_name + '.jpg',
        unique_name.split('_')[-1] + '.jpg',  # Только номер
    ]
    
    for img_name in possible_names:
        img_path = os.path.join(images_path, img_name)
        if os.path.exists(img_path):
            return img_path
    
    # Пробуем поиск по расширениям
    for ext in ['.jpg', '.png', '.jpeg', '.bmp']:
        img_pattern = os.path.join(images_path, f"*{ext}")
        img_files = glob.glob(img_pattern)
        for img_file in img_files:
            img_base = os.path.basename(img_file).replace(ext, '')
            # Пробуем сопоставить по номеру
            if unique_name.endswith(img_base) or img_base in unique_name:
                return img_file
    
    return None

def main():
    paths = get_user_input()
    if not paths:
        return
    
    estimator_coeffs = MobileHeadPoseEstimator(mode='coeffs')
    estimator_pnp = MobileHeadPoseEstimator(mode='pnp')
    
    # Получаем список JSON файлов по паттерну
    json_files = sorted(glob.glob(paths['markup_path']))
    
    if not json_files:
        print(f"ОШИБКА: Нет JSON файлов по паттерну '{paths['markup_path']}'")
        return
    
    print(f"Найдено {len(json_files)} JSON файлов разметки ключевых точек")
    
    # Загружаем ВСЕ ground truth файлы
    gt_data_list = []
    if paths['calculate_mae']:
        gt_data_list = load_all_ground_truth(paths['gt_path'])
    
    # Проверяем, что количество файлов совпадает
    if paths['calculate_mae'] and len(gt_data_list) > 0:
        if len(gt_data_list) != len(json_files):
            print(f"\n   ВНИМАНИЕ: Количество файлов не совпадает!")
            print(f"  JSON файлов разметки: {len(json_files)}")
            print(f"  JSON файлов углов: {len(gt_data_list)}")
            print("  Буду использовать первые {min(len(json_files), len(gt_data_list))} файлов")
    
    total = len(json_files)
    coeffs_ok = pnp_ok = 0
    files_with_gt = 0
    total_mae = 0
    
    print("\n" + "="*80)
    print("НАЧАЛО ОБРАБОТКИ")
    print("="*80)
    
    for i, json_path in enumerate(json_files):
        # Создаем уникальное имя из пути
        unique_name = extract_unique_filename(json_path, paths['markup_base_path'])
        print(f"\n[{i+1}/{total}] Обработка: {unique_name}")
        print(f"  JSON: {json_path}")
        
        # Загружаем данные разметки ключевых точек
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            print(f"Ошибка загрузки JSON: {e}")
            continue
        
        # Получаем ground truth для этого файла (если есть)
        gt = None
        if paths['calculate_mae'] and i < len(gt_data_list):
            gt = gt_data_list[i]['angles']
            print(f"Ground truth: {gt_data_list[i]['filename']}")
        elif paths['calculate_mae']:
            print(f"Ground truth не найден (индекс {i} вне диапазона)")
        
        # Обработка coeffs методом
        result_coeffs = estimator_coeffs.process_json(json_path)
        coeffs_success = print_coeffs_result(unique_name, result_coeffs or {}, gt)
        if coeffs_success: 
            coeffs_ok += 1
        
        # Обработка PnP методом
        result_pnp = estimator_pnp.process_json(json_path)
        if result_pnp:
            pnp_ok += 1
            
            # Вычисляем и выводим результат PnP
            mae = print_pnp_result(unique_name, result_pnp, gt)
            
            # Суммируем MAE если есть ground truth
            if gt:
                files_with_gt += 1
                total_mae += mae
        else:
            print(f"{unique_name:<30} | PnP    → FAILED")
        
        # Геометрический расчет
        calculator = GeometricPoseCalculator()
        result_geom = calculator.calculate_pose(data)
        
        # Сохраняем результаты в JSON
        result_data = {
            'filename': unique_name,
            'source_json': json_path,
            'pnp_result': result_pnp or {},
            'coeffs_result': result_coeffs or {},
            'geometric_result': result_geom or {}
        }
        
        if gt:
            result_data['ground_truth'] = gt
            result_data['gt_source'] = gt_data_list[i]['filename'] if i < len(gt_data_list) else "unknown"
            if result_pnp:
                err_y = abs(result_pnp['yaw'] - gt['yaw'])
                err_p = abs(result_pnp['pitch'] - gt['pitch'])
                err_r = abs(result_pnp['roll'] - gt['roll'])
                result_data['errors'] = {
                    'yaw_error': float(err_y),
                    'pitch_error': float(err_p),
                    'roll_error': float(err_r),
                    'mae': float((err_y + err_p + err_r) / 3)
                }
        
        # Сохраняем результаты в папку res_jsons
        result_filename = f"{unique_name}_result.json"
        result_path = os.path.join(paths['angles_output'], result_filename)
        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, indent=2, ensure_ascii=False)
        print(f"Результаты сохранены: {result_path}")
        
        # Сохраняем углы для сравнения
        if result_pnp and paths['angles_output']:
            save_angles_for_comparison(paths['angles_output'], unique_name, result_pnp)
        
        # Визуализация
        if paths['visual_output'] and result_pnp and paths['images_path']:
            # Ищем соответствующее изображение
            img_path = find_corresponding_image(paths['images_path'], unique_name)
            
            if img_path and os.path.exists(img_path):
                img = cv2.imread(img_path)
                if img is not None:
                    # Рисуем ключевые точки и направление
                    if 'props' in data and 'kp_nose_tip' in data['props']:
                        nose = tuple(map(int, data['props']['kp_nose_tip']))
                        visualize(img, nose, result_pnp)
                    
                    visual_path = os.path.join(paths['visual_output'], f"{unique_name}_vis.jpg")
                    cv2.imwrite(visual_path, img)
                    print(f"Визуализация сохранена: {visual_path}")
            else:
                if img_path:
                    print(f"Изображение не найдено: {img_path}")
                else:
                    print(f"Не удалось найти соответствующее изображение для {unique_name}")
    
    # Итоговая статистика
    print("\n" + "="*80)
    print("ИТОГИ ОБРАБОТКИ")
    print("="*80)
    print(f"Обработано файлов: {total}")
    print(f"  coeffs метод:    {coeffs_ok}/{total} успешно")
    print(f"  PnP метод:       {pnp_ok}/{total} успешно")
    
    if paths['calculate_mae'] and files_with_gt > 0:
        avg_mae = total_mae / files_with_gt
        print(f"\nСтатистика по размеченным данным:")
        print(f"  Файлов с ground truth: {files_with_gt}/{total}")
        print(f"  Средний MAE (PnP):     {avg_mae:.2f}°")
        
        # Оценка качества
        if avg_mae < 2:
            rating = "ОТЛИЧНО"
        elif avg_mae < 5:
            rating = "ХОРОШО"
        elif avg_mae < 10:
            rating = "УДОВЛЕТВОРИТЕЛЬНО"
        else:
            rating = "ПЛОХО"
        print(f"  Оценка точности:      {rating}")
    else:
        print(f"\nMAE не рассчитывался (нет ground truth разметки)")
    
    print(f"\nРезультаты сохранены:")
    if paths['angles_output']:
        print(f"  Углы для сравнения:  {paths['angles_output']}")
    if paths['visual_output']:
        print(f"  Визуализация:        {paths['visual_output']}")
    print("="*80)

if __name__ == "__main__":
    main()
