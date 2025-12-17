import os
import sys
from pathlib import Path

def main():
    if not os.path.exists("AFLW2000"):
        print("\nОШИБКА: Папка AFLW2000 не найдена!")
        print("Скачайте датасет с:")
        print("  - https://www.kaggle.com/datasets/mohamedadlyi/aflw2000-3d")
        print("  - Или: http://www.cbsr.ia.ac.cn/users/xiangyuzhu/projects/3DDFA/main.htm")
        print("\nСтруктура должна быть:")
        print("  AFLW2000/AFLW2000/*.jpg")
        print("  AFLW2000-3D/AFLW2000/*.mat")
        return
    
    # Шаг 1: Конвертация
    json_dir = "aflw2000_json"
    if not os.path.exists(json_dir) or len(os.listdir(json_dir)) < 1000:
        print("\nШаг 1: Конвертация AFLW2000-3D в JSON...")
        from evaluation.process_aflw2000_3d import convert_aflw2000_to_json
        
        image_dir = "../AFLW2000"
        mat_dir = "../AFLW2000"
        
        if not os.path.exists(image_dir):
            print(f"Папка {image_dir} не найдена!")
            return
        
        successful, failed = convert_aflw2000_to_json(
            image_dir, mat_dir, 
            json_dir, "aflw2000_visualized"
        )
        
        if successful < 100:
            print("Слишком мало успешно конвертированных файлов!")
            return
    else:
        print(f"\nШаг 1: JSON файлы уже существуют в {json_dir}")
    
    # Шаг 2: Оценка
    print("\nШаг 2: Оценка качества методов...")
    from evaluation.evaluate_aflw2000_3d import (
        load_aflw2000_ground_truth,
        evaluate_pose_estimation,
        print_metrics_table,
        plot_comparison
    )
    
    # Загружаем ground truth
    gt_data = load_aflw2000_ground_truth(json_dir)
    print(f"Загружено {len(gt_data)} изображений с аннотациями")
    
    # Оцениваем
    print("\nЗапуск оценки...")
    
    # Для быстрой проверки:
    # max_samples = 100
    # Для полной оценки:
    max_samples = None
    
    metrics, results, errors_detail = evaluate_pose_estimation(
        gt_data, json_dir,
        methods=['pnp', 'geom'],
        max_samples=max_samples,
        save_errors=True
    )
    
    # Результаты
    print_metrics_table(metrics)
    
    # Визуализация
    output_dir = "evaluation_results"
    plot_comparison(metrics, output_dir)
    

    print("\nШаг 3: Создание итогового отчета...")
    create_summary_report(metrics, output_dir)
    
    print("\n" + "="*60)
    print("ОЦЕНКА ЗАВЕРШЕНА УСПЕШНО!")
    print("="*60)
    print(f"Все результаты сохранены в папке: {output_dir}")
    print(f"Детальные ошибки: detailed_errors.json")
    print(f"Графики: {output_dir}/*.png")
    print(f"Метрики: {output_dir}/final_metrics.json")

def create_summary_report(metrics, output_dir):
    """Создает текстовый отчет"""
    report_path = os.path.join(output_dir, "summary_report.txt")
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write("ОТЧЕТ ПО ОЦЕНКЕ POSE ESTIMATION\n")
        f.write("Датасет: AFLW2000-3D\n")
        f.write("="*60 + "\n\n")
        
        f.write("РЕЗУЛЬТАТЫ:\n\n")
        
        for method in metrics.keys():
            f.write(f"{method.upper()} METHOD:\n")
            f.write("-"*40 + "\n")
            
            for angle in ['yaw', 'pitch', 'roll']:
                mae = metrics[method].get(f'{angle}_mae', 'N/A')
                rmse = metrics[method].get(f'{angle}_rmse', 'N/A')
                acc10 = metrics[method].get(f'{angle}_acc@10', 'N/A')
                
                if mae != 'N/A':
                    f.write(f"  {angle.capitalize():<6}: MAE={mae:.2f}°, RMSE={rmse:.2f}°, Acc@10°={acc10:.1f}%\n")
            
            f.write("\n")
        
        # Лучший метод!
        f.write("\nЛУЧШИЙ МЕТОД ПО КАЖДОМУ УГЛУ:\n")
        f.write("-"*40 + "\n")
        
        for angle in ['yaw', 'pitch', 'roll']:
            best_method = None
            best_mae = float('inf')
            
            for method in metrics.keys():
                mae = metrics[method].get(f'{angle}_mae', float('inf'))
                if mae < best_mae:
                    best_mae = mae
                    best_method = method
            
            if best_method:
                f.write(f"  {angle.capitalize():<6}: {best_method.upper()} (MAE={best_mae:.2f}°)\n")
    
    print(f"Отчет сохранен: {report_path}")

if __name__ == "__main__":
    main()
