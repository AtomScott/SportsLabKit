import torch

from .base import BaseDetectionModel


class YOLOv5(BaseDetectionModel):
    """YOLOv5 model wrapper.

    YOLOv5 has options that can be set on instantiation and during inference. However, frror compatibility with the rest of the codebase, we only support setting options during instantiation. Therefore, if you want to set options for inference, you must pass the options as a dictionary to the `model_config` argument. For example:

    ```python
    model = YOLOv5(
        model_name="yolov5s",
        model_repo="ultralytics/yolov5",
        model_ckpt="yolov5s.pt",
        model_config={"augment": True}
    )
    ```
    """

    def __init__(self, model_name, model_repo, model_ckpt, model_config=dict(), **rkwags):
        super().__init__(model_name, model_repo, model_ckpt, model_config)

        self.model_config = model_config
        self.size = model_config.get("size", 640)
        self.augment = model_config.get("augment", False)
        self.profile = model_config.get("profile", False)

    def load(self):
        model = torch.hub.load(str(self.model_repo), "custom", path=str(self.model_ckpt), source="local")
        if model is None:
            raise RuntimeError("Failed to load model")

        model.conf = self.model_config.get("conf", 0.25)
        model.iou = self.model_config.get("iou", 0.45)
        model.agnostic = self.model_config.get("agnostic", False)
        model.multi_label = self.model_config.get("multi_label", False)
        model.classes = self.model_config.get("classes", None)
        model.max_det = self.model_config.get("max_det", 1000)
        model.amp = self.model_config.get("amp", False)
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

        results = self.model(x, self.size, self.augment, self.profile).xywh
        results = [to_dict(r) for r in results]

        return results


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
