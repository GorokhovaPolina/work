import cv2
import os

def simple_video_to_dataset(video_path, output_dir, every_n_frame=1):
    # Папка с output
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    saved_count = 0
    
    # Создание lst файла
    lst_file = open(os.path.join(output_dir, "annotations.lst"), "w")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_count % every_n_frame == 0:
            filename = f"snapshot_{saved_count}.jpg"
            filepath = os.path.join(output_dir, filename)
            cv2.imwrite(filepath, frame)
            lst_file.write(f"{filename}\n")
            saved_count += 1
            
        frame_count += 1
    
    cap.release()
    lst_file.close()
    print(f"Сохранено {saved_count} кадров в {output_dir}")

if __name__ == "__main__":
    simple_video_to_dataset(
        video_path="C:/Users/polina/source/repos/work/video_2025-12-01_16-05-27.mp4", # ПУТЬ К ВИДЕО
        output_dir="imgs_Nastya",
        every_n_frame=1 
    )
