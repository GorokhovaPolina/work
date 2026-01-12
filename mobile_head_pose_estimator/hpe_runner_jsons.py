import glob
import os
import json
import cv2
import numpy as np
from estimator import MobileHeadPoseEstimator
from pose_calculator import GeometricPoseCalculator

def get_user_input():
    """Получаем все пути от пользователя"""
    print("="*60)
    print("НАСТРОЙКА ПУТЕЙ:")
    print("="*60)
    
    # Папка с JSON разметкой ключевых точек
    markup_base_path =  input("Путь к БАЗОВОЙ папке с разметкой ключевых точек: ").strip()
    if not os.path.exists(markup_base_path):
        print(f"ОШИБКА: Папка '{markup_base_path}' не существует!")
        return None
    
    # Паттерн поиска JSON файлов внутри базовой папки
    markup_pattern = input("Паттерн поиска JSON файлов [runlist_item_*/clip_*/snapshot_*.json]: ").strip()
    if not markup_pattern:
        markup_pattern = "runlist_item_*/clip_*/snapshot_*.json"
    
    markup_path = os.path.join(markup_base_path, markup_pattern)

    # Папка для сохранения углов для сравнения
    angles_output = input("Путь для сохранения углов для сравнения [/res_jsons]: ").strip()
    if angles_output:
        os.makedirs(angles_output, exist_ok=True)
    else:
        angles_output = "res_jsons"
        os.makedirs(angles_output, exist_ok=True)
    
    print("="*60)
    
    return {
        'markup_base_path': markup_base_path,
        'markup_pattern': markup_pattern,
        'markup_path': markup_path,
        'angles_output': angles_output,
    }

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

def extract_unique_filename(json_path, base_path):
    """Извлекаем уникальное имя файла из полного пути"""
    # Убираем базовый путь
    relative_path = os.path.relpath(json_path, base_path)
    # Заменяем слеши на подчеркивания для создания уникального имени
    unique_name = relative_path.replace(os.sep, '_').replace('.json', '')
    return unique_name

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
    
    print(f"Найдено {len(json_files)} JSON файлов")
    
    total = len(json_files)
    pnp_ok = 0
    for json_path in json_files:
        unique_name = extract_unique_filename(json_path, paths['markup_base_path'])
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            print(f"Ошибка загрузки JSON: {e}")
            continue
        
        result_pnp = estimator_pnp.process_json(json_path)
        if result_pnp:
            pnp_ok += 1
        else:
            print(f"{unique_name:<30} | PnP    → FAILED")
        calculator = GeometricPoseCalculator()
        result_geom = calculator.calculate_pose(data)
        
        # Сохраняем результаты в JSON
        result_data = {
            'filename': unique_name,
            'source_json': json_path,
            'pnp_result': result_pnp or {},
            'geometric_result': result_geom or {}
        }
        
        # Сохраняем углы для сравнения
        if result_pnp and paths['angles_output']:
            save_angles_for_comparison(paths['angles_output'], unique_name, result_pnp)
    
    # Итоговая статистика
    print(f"Обработано файлов: {total}")
    print(f"  Geom метод:       {pnp_ok}/{total} успешно")
    print("="*60)
    print(f"Результаты сохранены:")
    if paths['angles_output']:
        print(f"  Углы для сравнения:  {paths['angles_output']}")
    print("="*60)

if __name__ == "__main__":
    main()