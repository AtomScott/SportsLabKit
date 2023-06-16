from ..logger import logger
from .base import BaseDetectionModel, Detection
from .yolov5 import YOLOv5
from .dummy import DummyDetectionModel


def inheritors(cls):
    subclasses = set()
    work = [cls]
    while work:
        parent = work.pop()
        for child in parent.__subclasses__():
            if child not in subclasses:
                subclasses.add(child)
                work.append(child)
    return subclasses


def load(model_name, model_repo, model_ckpt, model_config=None, **kwargs):
    for cls in inheritors(BaseDetectionModel):
        if model_name in [cls.__name__.lower(), cls.__name__]:
            return cls(model_name, model_ckpt, inference_config)
    logger.warning(
        f"Model {model_name} not found. Available models: {[cls.__name__ for cls in inheritors(BaseDetectionModel)]} (lowercase is allowed)"
    )


if __name__ == "__main__":
    for cls in inheritors(BaseDetectionModel):
        print(cls.__name__)
