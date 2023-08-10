from dataclasses import dataclass
from typing import Optional
import torch

from sportslabkit.detection_model.base import BaseDetectionModel, BaseConfig


@dataclass
class ModelConfigTemplate(BaseConfig):
    ckpt: str = ""
    repo: str = "ultralytics/yolov5"
    name: str = "yolov5s"
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
    size: int = 640
    profile: bool = False


class YOLOv5(BaseDetectionModel):
    """YOLOv5 model wrapper."""

    def load(self):
        model_ckpt = self.model_config["ckpt"]
        model_repo = self.model_config["repo"]
        model_name = self.model_config["name"]
        if model_ckpt == "":
            model = torch.hub.load(str(model_repo), model_name)
        else:
            model = torch.hub.load(str(model_repo), "custom", path=str(model_ckpt))

        if model is None:
            raise RuntimeError("Failed to load model")

        model.conf = self.model_config["conf"]
        model.iou = self.model_config["iou"]
        model.agnostic = self.model_config["agnostic"]
        model.multi_label = self.model_config["multi_label"]
        model.classes = self.model_config["classes"]
        model.max_det = self.model_config["max_det"]
        model.amp = self.model_config["amp"]
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

        size = kwargs.get("size", self.inference_config["size"])
        augment = kwargs.get("augment", self.inference_config["augment"])
        profile = kwargs.get("profile", self.inference_config["profile"])
        results = self.model(x, size, augment, profile).xywh
        results = [to_dict(r) for r in results]

        return results

    @property
    def model_config_template(self):
        return ModelConfigTemplate

    @property
    def inference_config_template(self):
        return InferenceConfigTemplate


class YOLOv5n(YOLOv5):
    def __init__(self, model_config={}, inference_config={}):
        model_config["name"] = "yolov5n"
        super().__init__(model_config, inference_config)


class YOLOv5s(YOLOv5):
    def __init__(self, model_config={}, inference_config={}):
        model_config["name"] = "yolov5s"
        super().__init__(model_config, inference_config)


class YOLOv5m(YOLOv5):
    def __init__(self, model_config={}, inference_config={}):
        model_config["name"] = "yolov5m"
        super().__init__(model_config, inference_config)


class YOLOv5l(YOLOv5):
    def __init__(self, model_config={}, inference_config={}):
        model_config["name"] = "yolov5l"
        super().__init__(model_config, inference_config)


class YOLOv5x(YOLOv5):
    def __init__(self, model_config={}, inference_config={}):
        model_config["name"] = "yolov5x"
        super().__init__(model_config, inference_config)


if __name__ == "__main__":
    import argparse

    from ..utils.utils import get_git_root

    git_root = get_git_root()
    yolov5_repo = git_root / "external" / "yolov5"
    model_ckpt = git_root / "models" / "yolov5" / "yolov5s.pt"

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_repo", type=str, default=yolov5_repo, help="model name")
    parser.add_argument("--model_ckpt", type=str, default=model_ckpt, help="model name")
    args = parser.parse_args()

    # Model
    model = YOLOv5(args.model_repo, args.model_ckpt)

    model.test()
