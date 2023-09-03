try:
    import clip
except ImportError:
    print(
        "The clip module is not installed. Please install it using the following command:\n"
        "pip install git+https://github.com/openai/CLIP.git"
    )


import torch
from PIL import Image

from sportslabkit.image_model.base import BaseImageModel


class BaseCLIP(BaseImageModel):
    def __init__(self, name: str = "RN50", device: str = "cpu", image_size: tuple[int, int] = (224, 224),):
        """
        Initializes the base image embedding model.

        Args:
            name (str, optional): Name of the model. Defaults to "RN50".
            device (str, optional): Device to run the model on. Defaults to "cpu".
            image_size (tuple[int, int], optional): Size of the image. Defaults to (224, 224).
        """
        super().__init__()
        self.name = name
        self.device = device
        self.image_size = image_size
        self.input_is_batched = False  # initialize the input_is_batched attribute
        self.model = self.load()

    def load(self):
        model_name = self.name
        device = self.device
        model, preprocess = clip.load(model_name, device=device)
        self.preprocess = preprocess
        return model

    def forward(self, x):
        ims = []
        for _x in x:
            im = Image.fromarray(_x)
            im = im.resize(self.image_size)
            im = self.preprocess(im)
            ims.append(im)
        ims = torch.stack(ims)
        with torch.no_grad():
            image_features = self.model.encode_image(ims)
        return image_features


class CLIP_RN50(BaseCLIP):
    def __init__(
        self,
        name: str = "RN50",
        device: str = "cpu",
        image_size: tuple[int, int] = (224, 224),
    ):
        super().__init__(name, device, image_size)



class CLIP_RN101(BaseCLIP):
    def __init__(
        self,
        name: str = "RN101",
        device: str = "cpu",
        image_size: tuple[int, int] = (224, 224),
    ):
        super().__init__(name, device, image_size)


class CLIP_RN50x4(BaseCLIP):
    def __init__(
        self,
        name: str = "RN50x4",
        device: str = "cpu",
        image_size: tuple[int, int] = (224, 224),
    ):
        super().__init__(name, device, image_size)


class CLIP_RN50x16(BaseCLIP):
    def __init__(
        self,
        name: str = "RN50x16",
        device: str = "cpu",
        image_size: tuple[int, int] = (224, 224),
    ):
        super().__init__(name, device, image_size)


class CLIP_RN50x64(BaseCLIP):
    def __init__(
        self,
        name: str = "RN50x64",
        device: str = "cpu",
        image_size: tuple[int, int] = (224, 224),
    ):
        super().__init__(name, device, image_size)


class CLIP_ViT_B_32(BaseCLIP):
    def __init__(
        self,
        name: str = "ViT-B/32",
        device: str = "cpu",
        image_size: tuple[int, int] = (224, 224),
    ):
        super().__init__(name, device, image_size)


class CLIP_ViT_B_16(BaseCLIP):
    def __init__(
        self,
        name: str = "ViT-B/16",
        device: str = "cpu",
        image_size: tuple[int, int] = (224, 224),
    ):
        super().__init__(name, device, image_size)


class CLIP_ViT_L_14(BaseCLIP):
    def __init__(
        self,
        name: str = "ViT-L/14",
        device: str = "cpu",
        image_size: tuple[int, int] = (224, 224),
    ):
        super().__init__(name, device, image_size)


class CLIP_ViT_L_14_336px(BaseCLIP):
    def __init__(
        self,
        name: str = "ViT-L/14@336px",
        device: str = "cpu",
        image_size: tuple[int, int] = (224, 224),
    ):
        super().__init__(name, device, image_size)
