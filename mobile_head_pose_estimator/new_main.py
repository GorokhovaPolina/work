import glob
import os
import json
import cv2
import numpy as np
import re  # для работы с регулярными выражениями
from estimator import MobileHeadPoseEstimator
from visualizer1 import visualize
from pose_calculator import GeometricPoseCalculator

def get_user_input():
    """Получаем все пути от пользователя"""
    print("="*60)
    print("НАСТРОЙКА ПУТЕЙ:")
    print("="*60)
    
    # Папка с JSON разметкой ключевых точек
    markup_base_path = "../" + input("Путь к БАЗОВОЙ папке с разметкой ключевых точек: ").strip()
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

def extract_clip_number_from_path(json_path):
    """Извлекаем номер клипа из пути к JSON файлу"""
    # Ищем паттерн clip_XXXX в пути
    match = re.search(r'clip_(\d+)', json_path)
    if match:
        return int(match.group(1))
    return None

def extract_number_from_gt_filename(filename):
    """Извлекаем номер из имени файла ground truth"""
    # Ищем все числа в имени файла
    numbers = re.findall(r'\d+', filename)
    if numbers:
        # Возвращаем последнее число (например, 9 из IMG_0001_9.json)
        return int(numbers[-1])
    return None

def create_mapping_table(json_files, gt_files):
    """Создаем таблицу соответствия между JSON файлами разметки и ground truth"""
    mapping = {}
    
    print("\nСОЗДАНИЕ ТАБЛИЦЫ СООТВЕТСТВИЯ:")
    print("-" * 50)
    
    # Сначала попробуем по номерам клипов
    for json_path in json_files:
        clip_num = extract_clip_number_from_path(json_path)
        json_name = os.path.basename(json_path)
        
        # Ищем соответствующий ground truth файл
        matched_gt = None
        
        for gt_file in gt_files:
            gt_num = extract_number_from_gt_filename(os.path.basename(gt_file))
            
            # Проверяем разные варианты соответствия
            if clip_num is not None and gt_num is not None:
                # Вариант 1: прямое соответствие номеров
                if clip_num == gt_num:
                    matched_gt = gt_file
                    mapping[json_path] = gt_file
                    print(f"  {json_name} (clip_{clip_num:04d}) → {os.path.basename(gt_file)} (номер {gt_num})")
                    break
                
                # Вариант 2: возможно, номер в gt - это номер изображения в клипе
                # Или другая логика (например, clip_0056 → IMG_0001_9.json)
                # Нужно понять вашу логику соответствия
        
        if matched_gt is None:
            mapping[json_path] = None
            print(f"  {json_name} (clip_{clip_num:04d}) → НЕ НАЙДЕН")
    
    return mapping

def load_ground_truth_for_json(json_path, gt_mapping, gt_cache):
    """Загружаем ground truth для конкретного JSON файла по таблице соответствия"""
    if json_path not in gt_mapping or gt_mapping[json_path] is None:
        return None
    
    gt_file = gt_mapping[json_path]
    
    # Используем кэш если уже загружали
    if gt_file in gt_cache:
        return gt_cache[gt_file]
    
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
        
        # Сохраняем в кэш
        gt_cache[gt_file] = angles
        return angles
        
    except Exception as e:
        print(f"  Ошибка загрузки ground truth {os.path.basename(gt_file)}: {e}")
        return None

def find_corresponding_image(images_path, json_path):
    """Ищем соответствующее изображение для JSON файла"""
    if not images_path:
        return None
    
    # Извлекаем базовое имя
    json_name = os.path.basename(json_path).replace('.json', '')
    
    # Пробуем разные варианты имен
    possible_names = [
        json_name + '.jpg',
        json_name + '.png',
        json_name + '.jpeg',
        json_name + '.bmp',
    ]
    
    for img_name in possible_names:
        img_path = os.path.join(images_path, img_name)
        if os.path.exists(img_path):
            return img_path
    
    # Если не нашли по точному имени, ищем по номеру клипа
    clip_num = extract_clip_number_from_path(json_path)
    if clip_num is not None:
        # Ищем файлы, содержащие номер клипа
        for ext in ['.jpg', '.png', '.jpeg', '.bmp']:
            img_pattern = os.path.join(images_path, f"*{clip_num}*{ext}")
            img_files = glob.glob(img_pattern)
            if img_files:
                return img_files[0]  # Берем первое найденное
    
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
    
    # Получаем список всех ground truth файлов
    gt_files = []
    gt_mapping = {}
    gt_cache = {}
    
    if paths['calculate_mae']:
        gt_files = sorted(glob.glob(os.path.join(paths['gt_path'], "*.json")))
        print(f"Найдено {len(gt_files)} файлов ground truth")
        
        # Создаем таблицу соответствия
        gt_mapping = create_mapping_table(json_files, gt_files)
    
    total = len(json_files)
    coeffs_ok = pnp_ok = 0
    files_with_gt = 0
    total_mae = 0
    
    print("\n" + "="*80)
    print("НАЧАЛО ОБРАБОТКИ")
    print("="*80)
    
    for i, json_path in enumerate(json_files):
        json_name = os.path.basename(json_path).replace('.json', '')
        clip_num = extract_clip_number_from_path(json_path)
        
        print(f"\n[{i+1}/{total}] Обработка: {json_name}")
        if clip_num is not None:
            print(f"  Клип: clip_{clip_num:04d}")
        print(f"  JSON: {json_path}")
        
        # Загружаем данные разметки ключевых точек
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            print(f"Ошибка загрузки JSON: {e}")
            continue
        
        # Получаем ground truth для этого файла
        gt = None
        if paths['calculate_mae']:
            gt = load_ground_truth_for_json(json_path, gt_mapping, gt_cache)
            if gt:
                print(f"Ground truth: {os.path.basename(gt_mapping[json_path])}")
            else:
                print(f"Ground truth не найден")
        
        # Обработка coeffs методом
        result_coeffs = estimator_coeffs.process_json(json_path)
        coeffs_success = print_coeffs_result(json_name, result_coeffs or {}, gt)
        if coeffs_success: 
            coeffs_ok += 1
        
        # Обработка PnP методом
        result_pnp = estimator_pnp.process_json(json_path)
        if result_pnp:
            pnp_ok += 1
            
            # Вычисляем и выводим результат PnP
            mae = print_pnp_result(json_name, result_pnp, gt)
            
            # Суммируем MAE если есть ground truth
            if gt:
                files_with_gt += 1
                total_mae += mae
        else:
            print(f"{json_name:<30} | PnP    → FAILED")
        
        # Геометрический расчет
        calculator = GeometricPoseCalculator()
        result_geom = calculator.calculate_pose(data)
        
        # Сохраняем результаты в JSON
        result_data = {
            'filename': json_name,
            'clip_number': clip_num,
            'source_json': json_path,
            'pnp_result': result_pnp or {},
            'coeffs_result': result_coeffs or {},
            'geometric_result': result_geom or {}
        }
        
        if gt:
            result_data['ground_truth'] = gt
            result_data['gt_source'] = os.path.basename(gt_mapping[json_path]) if json_path in gt_mapping else "unknown"
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
        result_filename = f"{json_name}_result.json"
        result_path = os.path.join(paths['angles_output'], result_filename)
        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, indent=2, ensure_ascii=False)
        print(f"Результаты сохранены: {result_path}")
        
        # Сохраняем углы для сравнения
        if result_pnp and paths['angles_output']:
            save_angles_for_comparison(paths['angles_output'], json_name, result_pnp)
        
        # Визуализация
        if paths['visual_output'] and result_pnp and paths['images_path']:
            # Ищем соответствующее изображение
            img_path = find_corresponding_image(paths['images_path'], json_path)
            
            if img_path and os.path.exists(img_path):
                img = cv2.imread(img_path)
                if img is not None:
                    # Рисуем ключевые точки и направление
                    if 'props' in data and 'kp_nose_tip' in data['props']:
                        nose = tuple(map(int, data['props']['kp_nose_tip']))
                        visualize(img, nose, result_pnp)
                    
                    visual_path = os.path.join(paths['visual_output'], f"{json_name}_vis.jpg")
                    cv2.imwrite(visual_path, img)
                    print(f"Визуализация сохранена: {visual_path}")
            else:
                if img_path:
                    print(f"Изображение не найдено: {img_path}")
                else:
                    print(f"Не удалось найти соответствующее изображение для {json_name}")
    
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
