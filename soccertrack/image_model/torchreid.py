import os
import sys

import requests
from torchreid.utils import FeatureExtractor

from soccertrack.logger import logger
from soccertrack.utils import download_file_from_google_drive, get_git_root

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
}


def show_torchreid_models():
    """Print available models as a list."""
    return list(model_dict.keys())


def download_model(model_name):
    url = model_dict[model_name]
    filename = model_name + ".pth"
    file_path = model_save_dir / filename

    file_path.parent.mkdir(parents=True, exist_ok=True)

    if file_path.exists():
        logger.info(f"Model {model_name} already exists in {model_save_dir}.")
        return file_path

    download_file_from_google_drive(url.split("/")[-2], file_path)
    logger.info(
        f"Model {model_name} successfully downloaded and saved to {model_save_dir}."
    )
    return file_path


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


class TorchReIDModel(FeatureExtractor):
    def __init__(
        self,
        model_name="",
        model_path="",
        image_size=(256, 128),
        pixel_mean=[0.485, 0.456, 0.406],
        pixel_std=[0.229, 0.224, 0.225],
        pixel_norm=True,
        device="cuda",
        verbose=True,
    ):
        if (model_name != "") and (model_path == ""):
            model_path = download_model(model_name)
            logger.info(model_path)
        if model_name.endswith("MSMT17"):
            model_name = model_name.replace("_MSMT17", "")
        if not verbose:
            with HiddenPrints():
                super().__init__(
                    model_name,
                    model_path,
                    image_size,
                    pixel_mean,
                    pixel_std,
                    pixel_norm,
                    device,
                    verbose,
                )
        else:
            super().__init__(
                model_name,
                model_path,
                image_size,
                pixel_mean,
                pixel_std,
                pixel_norm,
                device,
                verbose,
            )
