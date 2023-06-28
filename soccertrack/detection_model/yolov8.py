from typing import Any, Dict
from dataclasses import dataclass
from typing import Optional

import numpy as np

try:
    from ultralytics import YOLO
except ImportError:
    print(
        "The ultralytics module is not installed. Please install it using the following command:\n"
        "pip install ultralytics"
    )

from soccertrack.detection_model.base import BaseDetectionModel, BaseConfig


@dataclass
class ModelConfigTemplate(BaseConfig):
    ckpt: str = ""
    conf: float = 0.25
    iou: float = 0.45
    agnostic: bool = False
    multi_label: bool = False
    classes: Optional[list[str]] = None
    max_det: int = 1000
    amp: bool = False


@dataclass
class InferenceConfigTemplate(BaseConfig):
    augment: bool = False
    imgsz: int = 640
    verbose: bool = False
    conf: float = 0.25
    iou: float = 0.45


class YOLOv8(BaseDetectionModel):
    """YOLO model wrapper.
    Receives the arguments controlling inference as 'inference_config' when initialized.
    """

    def load(self):
        model_ckpt = self.model_config["ckpt"]
        model = YOLO(model_ckpt)
        return model

    def forward(self, x, **kwargs):
        def to_dict(res):
            if len(res) == 0:
                return [{}]
            return [
                {
                    "bbox_left": r[0] - r[2] / 2,
                    "bbox_top": r[1] - r[3] / 2,
                    "bbox_width": r[2],
                    "bbox_height": r[3],
                    "conf": r[4],
                    "class": r[5],
                }
                for r in res
            ]

        inference_config = self.inference_config
        inference_config.update(kwargs)
        results = self.model(x, **inference_config, task="detect")
        preds = []
        for result in results:
            xywh = result.boxes.xywh.detach().cpu().numpy()
            conf = result.boxes.conf.detach().cpu().numpy()
            cls = result.boxes.cls.detach().cpu().numpy()
            res = np.concatenate([xywh, conf.reshape(-1, 1), cls.reshape(-1, 1)], axis=1)
            preds.append(to_dict(res))

        return preds

    @property
    def model_config_template(self):
        return ModelConfigTemplate

    @property
    def inference_config_template(self):
        return InferenceConfigTemplate


class YOLOv8n(YOLOv8):
    def __init__(self, model_config={}, inference_config={}):
        model_config["ckpt"] = model_config.get("ckpt", "yolov8n")
        super().__init__(model_config, inference_config)


class YOLOv8s(YOLOv8):
    def __init__(self, model_config={}, inference_config={}):
        model_config["ckpt"] = model_config.get("ckpt", "yolov8s")
        super().__init__(model_config, inference_config)


class YOLOv8m(YOLOv8):
    def __init__(self, model_config={}, inference_config={}):
        model_config["ckpt"] = model_config.get("ckpt", "yolov8m")
        super().__init__(model_config, inference_config)


class YOLOv8l(YOLOv8):
    def __init__(self, model_config={}, inference_config={}):
        model_config["ckpt"] = model_config.get("ckpt", "yolov8l")
        super().__init__(model_config, inference_config)


class YOLOv8x(YOLOv8):
    def __init__(self, model_config={}, inference_config={}):
        model_config["ckpt"] = model_config.get("ckpt", "yolov8x")
        super().__init__(model_config, inference_config)
