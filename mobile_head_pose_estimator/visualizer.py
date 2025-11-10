# visualizer.py
import cv2
import numpy as np

def draw_axes(img, nose, scale=25):  # Уменьшил масштаб для маленького изображения
    """Рисуем оси с правильным масштабом"""
    h, w = img.shape[:2]
    scale = min(scale, w//4, h//4)  # Ограничиваем масштаб
    
    # X — красная (вправо)
    if nose[0] + scale < w:
        cv2.line(img, nose, (nose[0] + scale, nose[1]), (0, 0, 255), 2)
        cv2.putText(img, 'X', (nose[0] + scale + 3, nose[1]), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
    
    # Y — зелёная (вверх)  
    if nose[1] - scale > 0:
        cv2.line(img, nose, (nose[0], nose[1] - scale), (0, 255, 0), 2)
        cv2.putText(img, 'Y', (nose[0], nose[1] - scale - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    
    # Z — синяя (влево)
    if nose[0] - scale > 0:
        cv2.line(img, nose, (nose[0] - scale, nose[1]), (255, 0, 0), 2)
        cv2.putText(img, 'Z', (nose[0] - scale - 12, nose[1]), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

def draw_visible_nose(img, nose, rvec, tvec, K, dist, nose_length=25, color=(0, 200, 255)):
    """
    Рисуем УПРОЩЕННЫЙ и ВИДИМЫЙ нос для маленьких изображений
    """
    # Параметры, подходящие для маленького изображения
    base_radius = 8   # Радиус основания
    tip_radius = 3    # Радиус кончика
    
    # Создаем простой конус: основание -> кончик
    cone_points = []
    
    # Основание (круг вокруг носа)
    for i in range(8):  # Всего 8 точек для плавности
        angle = 2 * np.pi * i / 8
        x = base_radius * np.cos(angle)
        y = base_radius * np.sin(angle) 
        z = 0  # У лица
        cone_points.append([x, y, z])
    
    # Кончик носа
    cone_points.append([0, 0, nose_length])
    
    # Проекция в 2D
    cone_3d = np.float32(cone_points)
    pts, _ = cv2.projectPoints(cone_3d, rvec, tvec, K, dist)
    pts = np.int32(pts).reshape(-1, 2)
    
    base_pts = pts[:-1]  # Точки основания
    nose_tip = tuple(pts[-1])  # Кончик носа
    
    # Проверяем, виден ли кончик носа
    h, w = img.shape[:2]
    if not (0 <= nose_tip[0] < w and 0 <= nose_tip[1] < h):
        print(f"Кончик носа за пределами изображения: {nose_tip}")
        # Корректируем длину носа
        return draw_visible_nose(img, nose, rvec, tvec, K, dist, 
                               nose_length=nose_length-5, color=color)
    
    # Рисуем ОСНОВАНИЕ (круг вокруг носа)
    if len(base_pts) > 2:
        cv2.polylines(img, [base_pts], True, color, 2)
    
    # Рисуем ЛИНИИ от основания к кончику
    for base_pt in base_pts[::2]:  # Каждую вторую точку для избежания нагромождения
        cv2.line(img, tuple(base_pt), nose_tip, color, 2)
    
    # Выделяем КОНЧИК НОСА
    cv2.circle(img, nose_tip, 4, (255, 100, 0), -1)
    cv2.circle(img, nose_tip, 6, (255, 150, 0), 1)
    
    # Выделяем ОСНОВАНИЕ (точку носа на лице)
    cv2.circle(img, nose, 3, (255, 255, 255), -1)
    cv2.circle(img, nose, 5, (0, 100, 255), 1)
    
    return nose_tip

def visualize_for_small_image(img, nose, result):
    """
    Специальная визуализация для маленьких изображений (147x181)
    """
    # Рисуем оси
    draw_axes(img, nose, scale=20)
    
    if 'rvec' in result:
        # Рисуем нос
        nose_tip = draw_visible_nose(
            img, nose,
            result['rvec'], result['tvec'], result['K'], 
            result.get('dist', np.zeros((4,1))),
            nose_length=20,  # Укороченный нос для маленького изображения
            color=(0, 200, 255)  # Яркий оранжево-желтый
        )
        
        # Рисуем направление (стрелку от носа к кончику)
        cv2.arrowedLine(img, nose, nose_tip, (255, 255, 0), 2, tipLength=0.3)
    
    # Информация
    cv2.putText(img, "Nose Direction", (5, 15), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    # Отладочная информация
    debug_info = f"Img: {img.shape[1]}x{img.shape[0]} Nose: {nose}"
    cv2.putText(img, debug_info, (5, img.shape[0] - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200, 200, 200), 1)

# ИСПРАВЛЕННАЯ тестовая функция для вашего JSON
def test_with_json_data(img, json_data):
    """
    Тестовая функция с ПРАВИЛЬНЫМИ параметрами камеры
    """
    h, w = json_data["image_size"]
    
    # ПРАВИЛЬНАЯ матрица камеры для маленького изображения
    K = np.array([[200, 0, w/2],    # fx, fy - меньше для широкоугольного обзора
                  [0, 200, h/2], 
                  [0, 0, 1]], dtype=np.float32)
    
    # Ближе к объекту для маленького изображения
    tvec = np.array([[0, 0, 200]], dtype=np.float32)  # Было 500 - слишком далеко!
    
    # Углы из JSON (ваши данные: pitch=-10, yaw=2.5, roll=-2)
    pitch = np.radians(json_data["ground_truth"]["pitch"])
    yaw = np.radians(json_data["ground_truth"]["yaw"]) 
    roll = np.radians(json_data["ground_truth"]["roll"])
    
    rvec = np.array([pitch, yaw, roll], dtype=np.float32)
    
    # Точка носа из JSON
    nose = tuple(json_data["props"]["kp_nose_tip"])
    
    # Собираем результат
    result = {
        'rvec': rvec,
        'tvec': tvec, 
        'K': K,
        'dist': np.zeros((4, 1))
    }
    
    # Используем специальную визуализацию для маленьких изображений
    visualize_for_small_image(img, nose, result)
    
    return img

# Простая функция для быстрого теста
def quick_test():
    """
    Быстрый тест на черном изображении
    """
    # Создаем черное изображение вашего размера
    img = np.zeros((181, 147, 3), dtype=np.uint8)
    
    # Ваши данные
    json_data = {
        "image_size": [147, 181],
        "props": {"kp_nose_tip": [78, 108]},
        "ground_truth": {"yaw": 2.5, "pitch": -10.0, "roll": -2.0}
    }
    
    result_img = test_with_json_data(img, json_data)
    cv2.imshow('Nose Direction Test', result_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Запустите этот тест сначала!
if __name__ == "__main__":
    quick_test()