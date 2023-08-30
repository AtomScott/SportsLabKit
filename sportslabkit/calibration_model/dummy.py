from sportslabkit.logger import logger

from .base import BaseCalibrationModel


class DummyCalibrationModel(BaseCalibrationModel):
    def __init__(self, homographies, mode='constant'):
        super().__init__()
        self.homographies = homographies
        self.mode = mode
        self.image_count = 0

    def forward(self, x):
        if self.mode == 'constant':
            return self.homographies
        else:
            return self.homographies[self.image_count]
        
    def reset_image_count(self):
        self.image_count = 0
        logger.debug("Resetting image count")