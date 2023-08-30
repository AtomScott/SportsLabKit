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
        
    def forward(self, x):
        # Return the precomputed detections based on image_count
        if self.input_is_batched:
            start_index = self.image_count
            end_index = self.image_count + len(x)
            self.image_count += len(x)
            if self.image_count >= len(self.homographies):
                self.reset_image_count()
            return self.homographies[start_index:end_index]
        else:
            homograhy = self.homographies[self.image_count]
            self.image_count += 1
            results = [homograhy]
            self.reset_image_count()
            return results

    def reset_image_count(self):
        self.image_count = 0
        logger.debug("Resetting image count")