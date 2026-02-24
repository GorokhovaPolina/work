#!/usr/bin/env python3

import os
import json
import math
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R


def rotation_matrix_to_euler_angles(rot_matrix, convention='xyz'):
    """
    Преобразование матрицы поворота в углы Эйлера (yaw, pitch, roll)
    
    Args:
        rot_matrix: Матрица поворота 3x3 в виде списка или numpy массива
        convention: Соглашение о порядке осей (по умолчанию 'xyz' - roll, pitch, yaw)
    
    Returns:
        Словарь с углами в градусах: {'yaw': yaw, 'pitch': pitch, 'roll': roll}
    """
    rot_matrix = np.array(rot_matrix).reshape(3, 3)
    rotation = R.from_matrix(rot_matrix)
    euler_rad = rotation.as_euler(convention, degrees=False)
    euler_deg = np.degrees(euler_rad)
    if convention == 'xyz':
        roll, pitch, yaw = euler_deg
    elif convention == 'zyx':
        yaw, pitch, roll = euler_deg
    else:
        # По умолчанию последний угол - yaw
        yaw = euler_deg[-1]
        pitch = euler_deg[1]
        roll = euler_deg[0]
    
    return {
        'yaw': float(yaw),
        'pitch': float(pitch),
        'roll': float(roll)
    }


def alternative_rotation_to_euler(rot_matrix):
    """
    Альтернативный метод вычисления углов Эйлера из матрицы поворота
    без использования внешних библиотек (кроме numpy)
    
    Формулы из статьи Medium:
    yaw = atan2(r13, r33)
    pitch = -arcsin(r23)
    roll = atan2(r21, r22)
    """
    rot_matrix = np.array(rot_matrix).reshape(3, 3)
    
    # Извлекаем элементы матрицы
    r11, r12, r13 = rot_matrix[0]
    r21, r22, r23 = rot_matrix[1]
    r31, r32, r33 = rot_matrix[2]
    
    # Вычисляем углы в радианах
    # Yaw (поворот вокруг оси Z)
    yaw = -math.atan2(r32, r33)
    
    # Pitch (поворот вокруг оси Y)
    sy = math.sqrt(r32 * r32 +  r33 * r33)
    pitch = -math.atan2(r31, sy)
    # pitch = -math.asin(r23)
    
    # Roll (поворот вокруг оси X)
    roll = -math.atan2(r21, r11)
    
    # Преобразуем в градусы
    return {
        'yaw': float(round(np.degrees(yaw), 3)),
        'pitch': float(round(np.degrees(pitch), 3)),
        'roll': float(round(np.degrees(roll), 3))
    }


def parse_groundtruth_file(file_path):
    df = pd.read_csv(file_path, sep='\t')
    return df


def parse_calib_file(file_path):
    calib_data = {}
    
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            if 'camera_matrix:' in line:
                # Парсим матрицу камеры
                matrix_lines = []
                for _ in range(3):
                    next_line = f.readline().strip()
                    matrix_lines.append(next_line)
                
                # Объединяем и парсим числа
                matrix_str = ' '.join(matrix_lines)
                matrix_str = matrix_str.replace('[', '').replace(']', '').replace(',', '')
                matrix_values = [float(x) for x in matrix_str.split()]
                calib_data['camera_matrix'] = np.array(matrix_values).reshape(3, 3)
            
            elif 'image_width:' in line:
                calib_data['image_width'] = int(line.split(':')[1].strip())
            
            elif 'image_height:' in line:
                calib_data['image_height'] = int(line.split(':')[1].strip())
            
            elif 'dist_coeffs:' in line:
                # Парсим коэффициенты дисторсии
                dist_str = line.split(':')[1].strip()
                dist_str = dist_str.replace('[', '').replace(']', '').replace(',', '')
                dist_values = [float(x) for x in dist_str.split()]
                calib_data['dist_coeffs'] = np.array(dist_values)
    
    return calib_data


def create_pose_annotation(row, method='scipy', convention='xyz'):
    """
    Создание аннотации позы из строки данных
    
    Args:
        row: Строка DataFrame с данными
        method: Метод вычисления ('scipy' или 'alternative')
        convention: Соглашение об осях для scipy метода
    
    Returns:
        Словарь с аннотацией
    """
    # Извлекаем матрицу поворота из строки
    rot_matrix = [
        [row['rotMatrix_11'], row['rotMatrix_12'], row['rotMatrix_13']],
        [row['rotMatrix_21'], row['rotMatrix_22'], row['rotMatrix_23']],
        [row['rotMatrix_31'], row['rotMatrix_32'], row['rotMatrix_33']]
    ]
    
    # Вычисляем углы
    if method == 'scipy':
        angles = rotation_matrix_to_euler_angles(rot_matrix, convention)
    else:
        angles = alternative_rotation_to_euler(rot_matrix)
    
    # Создаем аннотацию
    annotation = {
        "head_pose": angles
    }
    
    return annotation


def process_dataset(groundtruth_path, output_dir, method='scipy', convention='xyz'):
    """
    Обработка всего датасета
    
    Args:
        groundtruth_path: Путь к файлу groundtruth.txt
        output_dir: Директория для сохранения результатов
        method: Метод вычисления углов
        convention: Соглашение об осях
    """
    # Создаем выходную директорию, если её нет
    os.makedirs(output_dir, exist_ok=True)
    
    # Парсим данные
    print(f"Чтение файла: {groundtruth_path}")
    df = parse_groundtruth_file(groundtruth_path)
    
    print(f"Найдено {len(df)} записей")
    
    # Создаем файлы аннотаций
    for idx, row in df.iterrows():
        # Создаем аннотацию
        annotation = create_pose_annotation(row, method, convention)
        
        # Формируем имя файла
        filename = f"clip_{idx:04d}.json"
        filepath = os.path.join(output_dir, filename)
        
        # Сохраняем в JSON
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(annotation, f, indent=4, ensure_ascii=False)
        
        if (idx + 1) % 10000 == 0:
            print(f"Обработано {idx + 1}/{len(df)} записей")
    
    print(f"Готово! Создано {len(df)} файлов в {output_dir}")



def main():
    # Настройки (hardcode)
    groundtruth_file = "C:/Users/polina/source/repos/work/parser_fdki/groudhtruth.txt"
    output_directory = 'head_pose_annotations'  # Директория для сохранения
    
    # Проверяем существование файла
    if not os.path.exists(groundtruth_file):
        print(f"Ошибка: файл {groundtruth_file} не найден!")
        return

    print("\nРасчет")
    process_dataset(
        groundtruth_file,
        output_directory,
        method='alternative',
        convention=None
    )


if __name__ == "__main__":
    main()
