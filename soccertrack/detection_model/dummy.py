from soccertrack.detection_model.base import BaseDetectionModel
from soccertrack.logger import logger


class DummyDetectionModel(BaseDetectionModel):
    def __init__(self, detections, *args, **kwargs):
        super().__init__(
            model_name=None, model_repo=None, model_ckpt=None, model_config=None
        )
        self.precomputed_detections = detections
        self.image_count = 0

    def load(self):
        # No model to load for the dummy detection model
        pass

    def forward(self, x):
        # Return the precomputed detections based on image_count
        if self.input_is_batched:
            start_index = self.image_count
            end_index = self.image_count + len(x)
            self.image_count += len(x)
            if self.image_count >= len(self.precomputed_detections):
                self.reset_image_count()
            return self.precomputed_detections[start_index:end_index]
        else:
            detections = self.precomputed_detections[self.image_count]
            self.image_count += 1

            results = [
                [
                    [d.box[0], d.box[1], d.box[2], d.box[3], d.score, d.class_id]
                    for d in detections
                ]
            ]
            if self.image_count >= len(self.precomputed_detections):
                self.reset_image_count()
            return results

    def reset_image_count(self):
        self.image_count = 0
        logger.debug("Resetting image count")
