from typing import Optional

import numpy as np


try:
    from ultralytics import YOLO
except ImportError:
    print("The ultralytics module is not installed. Please install it using the following command:\n" "pip install ultralytics")

from sportslabkit.detection_model.base import BaseDetectionModel


class YOLOv8(BaseDetectionModel):
    """YOLO model wrapper.
    Receives the arguments controlling inference as 'inference_config' when initialized.
    """

    hparam_search_space = {
        "max_det": {"type": "int", "low": 20, "high": 50},
        "imgsz": {"type": "int", "low": 1280, "high": 3840},
        "conf": {"type": "float", "low": 0.1, "high": 1.0},
        "iou": {"type": "float", "low": 0.1, "high": 1.0},
    }

    def __init__(
        self,
        model: str = "",
        agnostic_nms: bool = False,
        multi_label: bool = False,
        classes: Optional[list[str]] = None,
        max_det: int = 1000,
        amp: bool = False,
        imgsz: int = 640,
        conf: float = 0.25,
        iou: float = 0.45,
        device: str = "cpu",
        verbose: bool = False,
    ):
        """
        Initializes the base detection model.

        Args:
            model_config (Optional[dict]): The configuration for the model. This is optional and can be used to pass additional parameters to the model.
        """
        super().__init__()
        self.agnostic_nms = agnostic_nms
        self.classes = classes
        self.max_det = max_det
        self.amp = amp
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        self.device = device
        self.verbose = verbose

        self.model = YOLO(model=model)

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

        x = [_x[..., ::-1] for _x in x]
        results = self.model(
            x,
            agnostic_nms=kwargs.get("agnostic_nms", self.agnostic_nms),
            classes=kwargs.get("classes", self.classes),
            max_det=kwargs.get("max_det", self.max_det),
            imgsz=kwargs.get("imgsz", self.imgsz),
            conf=kwargs.get("conf", self.conf),
            iou=kwargs.get("iou", self.iou),
            device=kwargs.get("device", self.device),
            verbose=kwargs.get("verbose", self.verbose),
            task="detect",
        )
        preds = []
        for result in results:
            xywh = result.boxes.xywh.detach().cpu().numpy()
            conf = result.boxes.conf.detach().cpu().numpy()
            cls = result.boxes.cls.detach().cpu().numpy()
            res = np.concatenate([xywh, conf.reshape(-1, 1), cls.reshape(-1, 1)], axis=1)
            preds.append(to_dict(res))

        return preds


class YOLOv8n(YOLOv8):
    def __init__(self, **model_config):
        model_config["model"] = model_config.get("model", "yolov8n")
        super().__init__(model_config)


class YOLOv8s(YOLOv8):
    def __init__(self, **model_config):
        model_config["model"] = model_config.get("model", "yolov8s")
        super().__init__(model_config)


class YOLOv8m(YOLOv8):
    def __init__(self, **model_config):
        model_config["model"] = model_config.get("model", "yolov8m")
        super().__init__(model_config)


class YOLOv8l(YOLOv8):
    def __init__(self, **model_config):
        model_config["model"] = model_config.get("model", "yolov8l")
        super().__init__(model_config)


class YOLOv8x(YOLOv8):
    def __init__(
        self,
        model: str = "yolov8x",
        agnostic_nms: bool = False,
        multi_label: bool = False,
        classes: Optional[list[str]] = None,
        max_det: int = 1000,
        amp: bool = False,
        imgsz: int = 640,
        conf: float = 0.25,
        iou: float = 0.45,
        device: str = "cpu",
        verbose: bool = False,
    ):
        super().__init__(model, agnostic_nms, multi_label, classes, max_det, amp, imgsz, conf, iou, device, verbose)
