import torch

from .base import BaseDetectionModel


class YOLOv5(BaseDetectionModel):
    def __init__(self, model_name, model_repo, model_ckpt, **kwargs):
        super().__init__(model_name, model_repo, model_ckpt)
        self.model.conf = 0.8

    def load(self):
        return torch.hub.load(
            str(self.model_repo), "custom", path=str(self.model_ckpt), source="local"
        )

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

        results = self.model(x, **kwargs).xywh
        results = [to_dict(r) for r in results]

        return results


if __name__ == "__main__":
    import argparse

    from ..utils.utils import get_git_root

    git_root = get_git_root()
    yolov5_repo = git_root / "external" / "yolov5"
    model_ckpt = git_root / "models" / "yolov5" / "yolov5s.pt"

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_repo", type=str, default=yolov5_repo, help="model name"
    )
    parser.add_argument("--model_ckpt", type=str, default=model_ckpt, help="model name")
    args = parser.parse_args()

    # Model
    model = YOLOv5(args.model_repo, args.model_ckpt)

    model.test()
