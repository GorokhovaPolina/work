class MobileConfig:
    """
    Конфигурация для мобильной оптимизации
    """
    def __init__(self):
        # Детекция
        self.detection_interval = 15  # кадров между редетекцией
        self.detection_confidence_threshold = 0.7
        
        # Трекинг
        self.tracking_confidence_decay = 0.95
        self.min_tracking_confidence = 0.3
        self.max_frames_without_detection = 10
        
        # Сглаживание
        self.pose_smoothing_factor = 0.8
        self.kalman_filter_q = 0.1  # шум процесса
        self.kalman_filter_r = 0.1  # шум измерения
        
        # Производительность
        self.target_fps = 60
        self.max_processing_time_ms = 16  # для 60 FPS

        # Новое: для метрик и вердиктов
        self.theta = 5.0  # Порог для Accuracy_θ и Passed/Failed (в градусах)
        self.angle_ranges = {'yaw': 60, 'pitch': 30, 'roll': 30}  # Диапазоны для Invalid*