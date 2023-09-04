from sportslabkit.logger import logger

from .base import BaseDetectionModel


class DummyDetectionModel(BaseDetectionModel):
    def __init__(self, detections):
        super().__init__()
        self.precomputed_detections = detections
        self.image_count = 0

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
            results = [detections]
            # results = [[[d.box[0], d.box[1], d.box[2], d.box[3], d.score, d.class_id] for d in detections]]
            if self.image_count >= len(self.precomputed_detections):
                self.reset_image_count()
            return results

    def reset_image_count(self):
        self.image_count = 0
        logger.debug("Resetting image count")

    @staticmethod
    def from_bbdf(bbdf):
        # No model to load for the dummy detection model
        precomputed_detections = []
        cols = ["bb_left", "bb_top", "bb_width", "bb_height", "conf", "class"]
        for frame_idx, frame_df in bbdf.iter_frames():
            long_df = frame_df.to_long_df()
            long_df["class"] = 0
            d = (
                long_df[cols]
                .rename(
                    columns={
                        "bb_left": "bbox_left",
                        "bb_top": "bbox_top",
                        "bb_width": "bbox_width",
                        "bb_height": "bbox_height",
                    }
                )
                .to_dict(orient="records")
            )
            precomputed_detections.append(d)
        return DummyDetectionModel(precomputed_detections)
