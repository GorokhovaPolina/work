from abc import ABC, abstractmethod

class HeadPoseEstimator(ABC):
    @abstractmethod
    def estimate(self, landmarks, image_size):
        pass