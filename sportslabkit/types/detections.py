from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image
from sportslabkit.utils import read_image, increment_path
from sportslabkit.logger import logger
from sportslabkit.types.detection import Detection
from sportslabkit.utils.draw import draw_bounding_boxes
import pandas as pd


class Detections:
    """SoccerTrack detections class for inference results."""

    def __init__(
        self,
        preds: Union[List[Dict], List[List], List[Detection]],
        im: Union[str, Path, Image.Image, np.ndarray],
        times: Tuple[float, float, float] = (0, 0, 0),
        names: Optional[List[str]] = None,
    ):
        self.im = self._process_im(im)
        self.preds = self._process_preds(preds)
        self.names = names
        self.times = times

    def _process_im(self, im: Union[str, Path, Image.Image, np.ndarray]) -> np.ndarray:
        return read_image(im)

    def _process_pred(self, pred: Union[Dict, List, Detection]) -> np.ndarray:
        # process predictions
        if isinstance(pred, dict):
            if len(pred.keys()) != 6:
                raise ValueError("The prediction dictionary should contain exactly 6 items")
            return np.stack(
                [
                    pred["bbox_left"],
                    pred["bbox_top"],
                    pred["bbox_width"],
                    pred["bbox_height"],
                    pred["conf"],
                    pred["class"],
                ],
                axis=0,
            )
        elif isinstance(pred, list):
            if len(pred) != 6:
                raise ValueError("The prediction list should contain exactly 6 items")
            return np.array(pred)
        elif isinstance(pred, Detection):
            return np.array(
                [
                    pred.box[0],
                    pred.box[1],
                    pred.box[2],
                    pred.box[3],
                    pred.score,
                    pred.class_id,
                ]
            )
        elif isinstance(pred, np.ndarray):
            if pred.shape != (6,):
                raise ValueError(f"pred should have the shape (6, ), but got {pred.shape}")
            return pred
        else:
            raise TypeError(f"Unsupported prediction type: {type(pred)}")

    def _process_preds(self, preds: List[Any]) -> np.ndarray:
        _processed_preds = []
        for pred in preds:
            _processed_preds.append(self._process_pred(pred))
        preds = np.array(_processed_preds)
        if not preds.size:
            preds = np.zeros((0, 6))
        return preds


    def show(self, **kwargs) -> Image.Image:
        im = self.im
        boxes = self.preds[:, :4]
        labels = [f"{int(c)} {conf:.2f}" for conf, c in self.preds[:, 4:]]
        draw_im = draw_bounding_boxes(im, boxes, labels, **kwargs)
        return Image.fromarray(draw_im)

    def save_image(self, path: Union[str, Path], **kwargs):
        image = self.show(**kwargs)
        image.save(path)

    def save_boxes(self, path: Union[str, Path]):
        with open(path, "w") as f:
            for box in self.preds[:, :4]:
                f.write(",".join(map(str, box)) + "\n")

    def crop(
        self, save: bool = True, save_dir: Union[str, Path] = "runs/detect/exp", exist_ok: bool = False
    ) -> List[Image.Image]:
        save_dir = increment_path(save_dir, exist_ok, mkdir=True) if save else None
        images = []
        for box in self.preds[:, :4]:
            cropped_im = self.im[box[1] : box[1] + box[3], box[0] : box[0] + box[2]]
            if save_dir is not None:
                Image.fromarray(cropped_im).save(Path(save_dir) / f"{box}.png")
            images.append(cropped_im)
        return images

    def to_df(self):
        # return detections as pandas DataFrames, i.e. print(results.to_df())

        preds = self.preds
        df = pd.DataFrame(
            preds,
            columns=[
                "bbox_left",
                "bbox_top",
                "bbox_width",
                "bbox_height",
                "conf",
                "class",
            ],
        )
        return df

    def to_list(self):
        # return a list of Detection objects, i.e. 'for result in results.tolist():'

        # check if empty
        if len(self.preds) == 0:
            logger.warning("No results to show.")
            return []

        dets = []
        for x, y, w, h, conf, class_id in self.preds:
            det = Detection([x, y, w, h], conf, class_id)
            dets.append(det)
        return dets

    def merge(self, other):
        # merge two Detections objects
        if isinstance(other, Detections):
            other = other.preds

        # check if other is empty
        if len(other) == 0:
            return self
        pred = np.concatenate((self.preds, other), axis=0)
        return Detections(pred, self.im, self.names, self.times)

    def __len__(self):
        return len(self.preds)
