try:
    import clip
except ImportError:
    print(
        "The clip module is not installed. Please install it using the following command:\n"
        "pip install git+https://github.com/openai/CLIP.git"
    )

from dataclasses import dataclass, field

import torch
from PIL import Image

from soccertrack.image_model.base import BaseConfig, BaseImageModel


@dataclass
class ModelConfigTemplate(BaseConfig):
    name: str = "ViT-B/32"
    device: str = "cpu"


@dataclass
class InferenceConfigTemplate(BaseConfig):
    pass


class BaseCLIP(BaseImageModel):
    def load(self):
        model_name = self.model_config["name"]
        device = self.model_config["device"]
        model, preprocess = clip.load(model_name, device=device)
        self.preprocess = preprocess
        return model

    def forward(self, x):
        ims = []
        for _x in x:
            im = Image.fromarray(_x)
            im = self.preprocess(im)
            ims.append(im)
        ims = torch.stack(ims)
        with torch.no_grad():
            image_features = self.model.encode_image(ims)
        return image_features

    @property
    def model_config_template(self):
        return ModelConfigTemplate

    @property
    def inference_config_template(self):
        return InferenceConfigTemplate


class CLIP_RN50(BaseCLIP):
    def __init__(self, model_config={}, inference_config={}):
        model_config["name"] = "RN50"
        super().__init__(model_config, inference_config)


class CLIP_RN101(BaseCLIP):
    def __init__(self, model_config={}, inference_config={}):
        model_config["name"] = "RN101"
        super().__init__(model_config, inference_config)


class CLIP_RN50x4(BaseCLIP):
    def __init__(self, model_config={}, inference_config={}):
        model_config["name"] = "RN50x4"
        super().__init__(model_config, inference_config)


class CLIP_RN50x16(BaseCLIP):
    def __init__(self, model_config={}, inference_config={}):
        model_config["name"] = "RN50x16"
        super().__init__(model_config, inference_config)


class CLIP_RN50x64(BaseCLIP):
    def __init__(self, model_config={}, inference_config={}):
        model_config["name"] = "RN50x64"
        super().__init__(model_config, inference_config)


class CLIP_ViT_B_32(BaseCLIP):
    def __init__(self, model_config={}, inference_config={}):
        model_config["name"] = "ViT-B/32"
        super().__init__(model_config, inference_config)


class CLIP_ViT_B_16(BaseCLIP):
    def __init__(self, model_config={}, inference_config={}):
        model_config["name"] = "ViT-B/16"
        super().__init__(model_config, inference_config)


class CLIP_ViT_L_14(BaseCLIP):
    def __init__(self, model_config={}, inference_config={}):
        model_config["name"] = "ViT-L/14"
        super().__init__(model_config, inference_config)


class CLIP_ViT_L_14_336px(BaseCLIP):
    def __init__(self, model_config={}, inference_config={}):
        model_config["name"] = "ViT-L/14@336px"
        super().__init__(model_config, inference_config)
