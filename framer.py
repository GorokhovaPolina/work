import cv2
import os

def simple_video_to_dataset(video_path, output_dir, every_n_frame=1):
    # Создаем папку
    os.makedirs(output_dir, exist_ok=True)
    
    # Открываем видео
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    saved_count = 0
    
    # Файл аннотаций
    lst_file = open(os.path.join(output_dir, "../annotations.lst"), "w")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_count % every_n_frame == 0:
            filename = f"snapshot_{saved_count}.jpg"
            filepath = os.path.join(output_dir, filename)
            
            # Сохраняем кадр
            cv2.imwrite(filepath, frame)
            
            # Записываем в аннотации
            lst_file.write(f"{filename}\n")
            saved_count += 1
            
        frame_count += 1
    
    cap.release()
    lst_file.close()
    print(f"Сохранено {saved_count} кадров в {output_dir}")

# Использование
if __name__ == "__main__":
    simple_video_to_dataset(
        video_path="video_2025-11-07_09-22-04.mp4",
        output_dir="my_dataset",
        every_n_frame=5
    )
