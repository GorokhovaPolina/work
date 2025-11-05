import cv2
import numpy as np

class FastLandmarkDetector:
    """
    Детектор с Haar Cascades (без MediaPipe, для Python 3.13+)
    """
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        self.nose_cascade = cv2.CascadeClassifier('haarcascade_mcs_nose.xml')
        self.mouth_cascade = cv2.CascadeClassifier('haarcascade_mcs_mouth.xml')
        
        if self.face_cascade.empty() or self.eye_cascade.empty() or self.nose_cascade.empty() or self.mouth_cascade.empty():
            raise ValueError("Cascades не загружены. Скачайте XML для носа/рта.")

    def detect(self, image):
        preprocessed = self._preprocess_image(image)
        gray = cv2.cvtColor(preprocessed, cv2.COLOR_BGR2GRAY)
        
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        if len(faces) == 0:
            return None
        (x, y, w, h) = faces[0]
        roi_gray = gray[y:y+h, x:x+w]
        
        eyes = self.eye_cascade.detectMultiScale(roi_gray)
        if len(eyes) < 2:
            return None
        eyes = sorted(eyes, key=lambda e: e[0])
        left_eye = (x + eyes[0][0] + eyes[0][2]//2, y + eyes[0][1] + eyes[0][3]//2)
        right_eye = (x + eyes[1][0] + eyes[1][2]//2, y + eyes[1][1] + eyes[1][3]//2)
        
        noses = self.nose_cascade.detectMultiScale(roi_gray, 1.1, 3)
        if len(noses) == 0:
            return None
        nose = (x + noses[0][0] + noses[0][2]//2, y + noses[0][1] + noses[0][3]//2)
        
        mouths = self.mouth_cascade.detectMultiScale(roi_gray, 1.1, 3)
        if len(mouths) == 0:
            return None
        mouth = (x + mouths[0][0] + mouths[0][2]//2, y + mouths[0][1] + mouths[0][3]//2)
        
        return {'nose': nose, 'left_eye': left_eye, 'right_eye': right_eye, 'mouth': mouth}
    
    def _preprocess_image(self, image):
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        if len(image.shape) == 3:
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            lab[:,:,0] = clahe.apply(lab[:,:,0])
            image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        image = cv2.GaussianBlur(image, (5, 5), 0)
        return image
