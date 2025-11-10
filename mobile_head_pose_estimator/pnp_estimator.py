from head_pose_estimator import HeadPoseEstimator
import cv2
import numpy as np

class PnPEstimator(HeadPoseEstimator):
    def estimate(self, landmarks, image_size):
        w, h = image_size
        model = np.array([...], dtype=np.float32)
        pts = np.array([...], dtype=np.float32)
        K = np.array([[w,0,w/2],[0,h,h/2],[0,0,1]], dtype=np.float32)
        success, rvec, tvec = cv2.solvePnP(model, pts, K, np.zeros((4,1)), flags=cv2.SOLVEPNP_EPNP)
        # ... extract euler ...
        return {'yaw': yaw, 'pitch': pitch, 'roll': roll, 'rvec': rvec, 'tvec': tvec, 'K': K}