from typing import Any, Dict

import numpy as np
from ultralytics import YOLO

from soccertrack.detection_model.base import BaseDetectionModel


class YOLOModel(BaseDetectionModel):
    """YOLO model wrapper.
    Receives the arguments controlling inference as 'inference_config' when initialized.
    """

    def __init__(
        self, model_name: str, model_ckpt: str, inference_config: Dict[str, Any] = {}
    ):
        super().__init__(model_name, model_ckpt, inference_config)

    def load(self):
        model = YOLO(self.model_ckpt)
        return model

    def forward(self, x, **kwargs):
        def to_dict(r):
            if len(r) == 0:
                return {}
            return {
                "bbox_left": r[:, 0] - r[:, 2] / 2,
                "bbox_top": r[:, 1] - r[:, 3] / 2,
                "bbox_width": r[:, 2],
                "bbox_height": r[:, 3],
                "conf": r[:, 4],
                "class": r[:, 5],
            }

        results = self.model(x, **self.inference_config)
        preds = []
        for result in results:
            xywh = result.boxes.xywh.detach().cpu().numpy()
            conf = result.boxes.conf.detach().cpu().numpy()
            cls = result.boxes.cls.detach().cpu().numpy()
            res = np.concatenate(
                [xywh, conf.reshape(-1, 1), cls.reshape(-1, 1)], axis=1
            )
            preds.append(to_dict(res))

        return results


if __name__ == "__main__":
    import argparse

    from soccertrack import detection_model
    from soccertrack.utils.utils import get_git_root

    git_root = get_git_root()
    model_ckpt = git_root / "models" / "yolov5" / "yolov5s.pt"

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name", type=str, default="YOLOModel", help="model name"
    )
    parser.add_argument("--model_ckpt", type=str, default=model_ckpt, help="model name")
    args = parser.parse_args()

    # Model
    model = YOLOModel(args.model_name, args.model_ckpt, {"half": True})
    model.test()

    # test loading
    model = detection_model.load(args.model_name, args.model_ckpt, {"half": True})
    model.test()
