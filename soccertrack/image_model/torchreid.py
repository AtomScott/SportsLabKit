from dataclasses import dataclass, field

try:
    from torchreid.utils import FeatureExtractor
except ImportError:
    print(
        "The torchreid module is not installed. Please install it using the following command:\n"
        "pip install git+https://github.com/KaiyangZhou/deep-person-reid.git"
    )

from soccertrack.image_model.base import BaseConfig, BaseImageModel
from soccertrack.logger import logger
from soccertrack.utils import (
    HiddenPrints,
    download_file_from_google_drive,
    get_git_root,
)

model_save_dir = get_git_root() / "models" / "torchreid"

model_dict = {
    "shufflenet": "https://drive.google.com/file/d/1RFnYcHK1TM-yt3yLsNecaKCoFO4Yb6a-/view?usp=sharing",
    "mobilenetv2_x1_0": "https://drive.google.com/file/d/1K7_CZE_L_Tf-BRY6_vVm0G-0ZKjVWh3R/view?usp=sharing",
    "mobilenetv2_x1_4": "https://drive.google.com/file/d/10c0ToIGIVI0QZTx284nJe8QfSJl5bIta/view?usp=sharing",
    "mlfn": "https://drive.google.com/file/d/1PP8Eygct5OF4YItYRfA3qypYY9xiqHuV/view?usp=sharing",
    "osnet_x1_0": "https://drive.google.com/file/d/1LaG1EJpHrxdAxKnSCJ_i0u-nbxSAeiFY/view?usp=sharing",
    "osnet_x0_75": "https://drive.google.com/file/d/1uwA9fElHOk3ZogwbeY5GkLI6QPTX70Hq/view?usp=sharing",
    "osnet_x0_5": "https://drive.google.com/file/d/16DGLbZukvVYgINws8u8deSaOqjybZ83i/view?usp=sharing",
    "osnet_x0_25": "https://drive.google.com/file/d/1rb8UN5ZzPKRc_xvtHlyDh-cSz88YX9hs/view?usp=sharing",
    "osnet_ibn_x1_0": "https://drive.google.com/file/d/1sr90V6irlYYDd4_4ISU2iruoRG8J__6l/view?usp=sharing",
    "osnet_ain_x1_0": "https://drive.google.com/file/d/1-CaioD9NaqbHK_kzSMW8VE4_3KcsRjEo/view?usp=sharing",
    "osnet_ain_x0_75": "https://drive.google.com/file/d/1apy0hpsMypqstfencdH-jKIUEFOW4xoM/view?usp=sharing",
    "osnet_ain_x0_5": "https://drive.google.com/file/d/1KusKvEYyKGDTUBVRxRiz55G31wkihB6l/view?usp=sharing",
    "osnet_ain_x0_25": "https://drive.google.com/file/d/1SxQt2AvmEcgWNhaRb2xC4rP6ZwVDP0Wt/view?usp=sharing",
    "resnet50_MSMT17": "https://drive.google.com/file/d/1yiBteqgIZoOeywE8AhGmEQl7FTVwrQmf/view?usp=sharing",
    "osnet_x1_0_MSMT17": "https://drive.google.com/file/d/1IosIFlLiulGIjwW3H8uMRmx3MzPwf86x/view?usp=sharing",
    "osnet_ain_x1_0_MSMT17": "https://drive.google.com/file/d/1SigwBE6mPdqiJMqhuIY4aqC7--5CsMal/view?usp=sharing",
    "resnet50_MSMT17": "https://drive.google.com/file/d/1ep7RypVDOthCRIAqDnn4_N-UhkkFHJsj/view?usp=sharing",
    "resnet50_fc512_MSMT17": "https://drive.google.com/file/d/1fDJLcz4O5wxNSUvImIIjoaIF9u1Rwaud/view?usp=sharing",
}


@dataclass
class ModelConfigTemplate(BaseConfig):
    name: str = "osnet_x1_0"
    path: str = ""
    device: str = "cpu"
    image_size: tuple[int, int] = (256, 128)
    pixel_mean: list[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    pixel_std: list[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])
    pixel_norm: bool = True
    verbose: bool = False


@dataclass
class InferenceConfigTemplate(BaseConfig):
    pass


def show_torchreid_models():
    """Print available models as a list."""
    return list(model_dict.keys())


def download_model(model_name):
    if model_name not in model_dict:
        raise ValueError(
            f"Model {model_name} not available. Available models are: {show_torchreid_models()}"
        )
    url = model_dict[model_name]
    filename = model_name + ".pth"
    file_path = model_save_dir / filename

    file_path.parent.mkdir(parents=True, exist_ok=True)

    if file_path.exists():
        logger.debug(f"Model {model_name} already exists in {model_save_dir}.")
        return file_path

    download_file_from_google_drive(url.split("/")[-2], file_path)
    logger.debug(
        f"Model {model_name} successfully downloaded and saved to {model_save_dir}."
    )
    return file_path


class BaseTorchReIDModel(BaseImageModel):
    def load(self):
        model_name = self.model_config["name"]
        model_path = self.model_config["path"]
        device = self.model_config["device"]
        verbose = self.model_config["verbose"]

        if (model_name != "") and (model_path == ""):
            model_path = download_model(model_name)
            logger.debug(model_path)
        if model_name.endswith("MSMT17"):
            model_name = model_name.replace("_MSMT17", "")
        if verbose:
            return FeatureExtractor(
                model_name=model_name,
                model_path=model_path,
                device=device,
            )
        with HiddenPrints():
            return FeatureExtractor(
                model_name=model_name,
                model_path=model_path,
                device=device,
            )

    def forward(self, x):
        return self.model(list(x))

    @property
    def model_config_template(self):
        return ModelConfigTemplate

    @property
    def inference_config_template(self):
        return InferenceConfigTemplate


class ShuffleNet(BaseTorchReIDModel):
    def __init__(self, model_config={}, inference_config={}):
        model_config["name"] = "shufflenet"
        super().__init__(model_config, inference_config)


class MobileNetV2_x1_0(BaseTorchReIDModel):
    def __init__(self, model_config={}, inference_config={}):
        model_config["name"] = "mobilenetv2_x1_0"
        super().__init__(model_config, inference_config)


class MobileNetV2_x1_4(BaseTorchReIDModel):
    def __init__(self, model_config={}, inference_config={}):
        model_config["name"] = "mobilenetv2_x1_4"
        super().__init__(model_config, inference_config)


class MLFN(BaseTorchReIDModel):
    def __init__(self, model_config={}, inference_config={}):
        model_config["name"] = "mlfn"
        super().__init__(model_config, inference_config)


class OSNet_x1_0(BaseTorchReIDModel):
    def __init__(self, model_config={}, inference_config={}):
        model_config["name"] = "osnet_x1_0"
        super().__init__(model_config, inference_config)


class OSNet_x0_75(BaseTorchReIDModel):
    def __init__(self, model_config={}, inference_config={}):
        model_config["name"] = "osnet_x0_75"
        super().__init__(model_config, inference_config)


class OSNet_x0_5(BaseTorchReIDModel):
    def __init__(self, model_config={}, inference_config={}):
        model_config["name"] = "osnet_x0_5"
        super().__init__(model_config, inference_config)


class OSNet_x0_25(BaseTorchReIDModel):
    def __init__(self, model_config={}, inference_config={}):
        model_config["name"] = "osnet_x0_25"
        super().__init__(model_config, inference_config)


class OSNet_ibn_x1_0(BaseTorchReIDModel):
    def __init__(self, model_config={}, inference_config={}):
        model_config["name"] = "osnet_ibn_x1_0"
        super().__init__(model_config, inference_config)


class OSNet_ain_x1_0(BaseTorchReIDModel):
    def __init__(self, model_config={}, inference_config={}):
        model_config["name"] = "osnet_ain_x1_0"
        super().__init__(model_config, inference_config)


class OSNet_ain_x0_75(BaseTorchReIDModel):
    def __init__(self, model_config={}, inference_config={}):
        model_config["name"] = "osnet_ain_x0_75"
        super().__init__(model_config, inference_config)


class OSNet_ain_x0_5(BaseTorchReIDModel):
    def __init__(self, model_config={}, inference_config={}):
        model_config["name"] = "osnet_ain_x0_5"
        super().__init__(model_config, inference_config)


class OSNet_ain_x0_25(BaseTorchReIDModel):
    def __init__(self, model_config={}, inference_config={}):
        model_config["name"] = "osnet_ain_x0_25"
        super().__init__(model_config, inference_config)


class ResNet50(BaseTorchReIDModel):
    def __init__(self, model_config={}, inference_config={}):
        model_config["name"] = "resnet50"
        model_config["path"] = model_dict["resnet50_MSMT17"]
        super().__init__(model_config, inference_config)


class ResNet50_fc512(BaseTorchReIDModel):
    def __init__(self, model_config={}, inference_config={}):
        model_config["name"] = "resnet50_fc512"
        model_config["path"] = model_dict["resnet50_fc512_MSMT17"]
        super().__init__(model_config, inference_config)
