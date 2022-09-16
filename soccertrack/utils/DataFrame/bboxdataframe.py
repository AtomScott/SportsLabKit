from __future__ import division as _division
from __future__ import print_function as _print_function
import pandas as pd
from .base import SoccerTrackMixin


import os as _os
import os.path as _path
import numpy as np
import cv2 as _cv2
# import numpy as _np
from hashlib import md5 as _md5
from PIL import ImageFont
import random

_LOC = _path.realpath(_path.join(_os.getcwd(),_path.dirname(__file__)))

#https://clrs.cc/
_COLOR_NAME_TO_RGB = dict(
    navy=((0, 38, 63), (119, 193, 250)),
    blue=((0, 120, 210), (173, 220, 252)),
    aqua=((115, 221, 252), (0, 76, 100)),
    teal=((15, 205, 202), (0, 0, 0)),
    olive=((52, 153, 114), (25, 58, 45)),
    green=((0, 204, 84), (15, 64, 31)),
    lime=((1, 255, 127), (0, 102, 53)),
    yellow=((255, 216, 70), (103, 87, 28)),
    orange=((255, 125, 57), (104, 48, 19)),
    red=((255, 47, 65), (131, 0, 17)),
    maroon=((135, 13, 75), (239, 117, 173)),
    fuchsia=((246, 0, 184), (103, 0, 78)),
    purple=((179, 17, 193), (241, 167, 244)),
    black=((24, 24, 24), (220, 220, 220)),
    gray=((168, 168, 168), (0, 0, 0)),
    silver=((220, 220, 220), (0, 0, 0))
)

_COLOR_NAMES = list(_COLOR_NAME_TO_RGB)

# _FONT_PATH = _os.path.join(_LOC, "Ubuntu-B.ttf")
# _FONT_HEIGHT = 45
# _FONT = ImageFont.truetype(None, _FONT_HEIGHT)

def _rgb_to_bgr(color):
    return list(reversed(color))

# def _color_image(image, font_color, background_color):
#     return background_color + (font_color - background_color) * image / 255

# def _get_label_image(text, font_color_tuple_bgr, background_color_tuple_bgr):
#     text_image = _FONT.getmask(text)
#     shape = list(reversed(text_image.size))
#     bw_image = np.array(text_image).reshape(shape)

    # image = [
    #     _color_image(bw_image, font_color, background_color)[None, ...]
    #     for font_color, background_color
    #     in zip(font_color_tuple_bgr, background_color_tuple_bgr)
    # ]

    # return np.concatenate(image).transpose(1, 2, 0)

X1_INDEX = 1
Y1_INDEX = 2
X2_INDEX = 3
Y2_INDEX = 4
W_INDEX = 3
H_INDEX = 4

color_list = []
i = 0
while i < 1000:
    color_list.append(_COLOR_NAMES[random.randint(0, len(_COLOR_NAMES) - 1)])
    i += 1

class BBoxDataFrame(SoccerTrackMixin, pd.DataFrame):
    @property
    def _constructor(self):
        return BBoxDataFrame

    # @property
    # def _constructor_sliced(self):
    #     raise NotImplementedError("This pandas method constructs pandas.Series object, which is not yet implemented in {self.__name__}.")

    def to_list(self, df: pd.DataFrame, xywh=True) -> list:
        """Convert a dataframe column to a 2-dim list for evaluation of object detection.
        Args:
            df (pd.DataFrame): Dataframe
            xywh (bool): If True, convert to x1y1x2y2 format. Defaults to True.
            
        Returns:
            bbox_2dim_list: 2-dim list
        """

        #Apply def append_object_id
        obj_id  = 0
        id_list = []
        for column in df.columns:
            team_id = column[0]
            player_id = column[1] 
            id_list.append((team_id, player_id))
        for id in sorted(set(id_list)):
            df.loc[:, (id[0],  id[1], 'obj_id')] = obj_id
            obj_id += 1
        df = df[df.sort_index(axis=1,level=[0,1],ascending=[True,True]).columns]

        bbox_2dim_list = []
        bbox_cols = ['bb_left', 'bb_top', 'bb_width', 'bb_height', 'conf','class_id' ,'obj_id']
        num_cols = len(bbox_cols)
        df2list = df.values.tolist()
        
        for frame_id, frame_raw in enumerate(df2list):
            for idx in range(0,len(frame_raw), num_cols):
                bbox_2dim_list.append([frame_id] + frame_raw[idx:idx+num_cols])
        
        #extract nan rows
        bbox_2dim_list = [x for x in bbox_2dim_list if not any(pd.isnull(x))]
        # print(bbox_2dim_list)
        
        for bbox in bbox_2dim_list:    
            if xywh:
                bbox[X1_INDEX] = bbox[X1_INDEX]
                bbox[Y1_INDEX] = bbox[Y1_INDEX]
                bbox[X2_INDEX] = bbox[W_INDEX] + bbox[X1_INDEX]
                bbox[Y2_INDEX] = bbox[H_INDEX] + bbox[Y1_INDEX]

        return bbox_2dim_list

    def visualize_frame(self, frame_idx, frame):

        BB_LEFT_INDEX = 0
        BB_TOP_INDEX = 1
        BB_WIDTH_INDEX = 2
        BB_HEIGHT_INDEX = 3

        # color_list =[]
        # unique_obj_ids = self.columns.get_level_values('PlayerID').unique()
        # for obj_id in unique_obj_ids:
        # color = _COLOR_NAMES[random.randint(0, len(_COLOR_NAMES) - 1)]
        # print(color_list)

        unique_team_ids = self.columns.get_level_values('TeamID').unique()
        for team_id in unique_team_ids:
            bboxdf_team = self.xs(team_id, level='TeamID', axis=1)
            # print('------team_id------', team_id)

            unique_team_ids = bboxdf_team.columns.get_level_values('PlayerID').unique()
            for idx, player_id in enumerate(unique_team_ids):
                color = color_list[idx]

                bboxdf_player = bboxdf_team.xs(player_id, level='PlayerID', axis=1)
                bboxdf_player = bboxdf_player.reset_index(drop=True)
                bbox = bboxdf_player.loc[frame_idx]
                if np.isnan(bbox[BB_LEFT_INDEX]) and np.isnan(bbox[BB_TOP_INDEX]):
                    continue

                else:
                    bb_left = bbox[BB_LEFT_INDEX]
                    bb_top = bbox[BB_TOP_INDEX]
                    bb_right = bb_left + bbox[BB_WIDTH_INDEX]
                    bb_bottom = bb_top + bbox[BB_HEIGHT_INDEX]

                    x,y,x2,y2 = list([int(bb_left), int(bb_top), int(bb_right), int(bb_bottom)])
                    frame = add(frame, x, y, x2, y2, label=f'{team_id}_{player_id}', color=color)

        return frame


def add(image, left, top, right, bottom, label=None, color=None):
    _DEFAULT_COLOR_NAME = "purple"

    if type(image) is not np.ndarray:
        raise TypeError("'image' parameter must be a numpy.ndarray")
    try:
        left, top, right, bottom = int(left), int(top), int(right), int(bottom)
    except ValueError:
        raise TypeError("'left', 'top', 'right' & 'bottom' must be a number")

    if label and type(label) is not str:
        raise TypeError("'label' must be a str")

    if label and not color:
        hex_digest = _md5(label.encode()).hexdigest()
        color_index = int(hex_digest, 16) % len(_COLOR_NAME_TO_RGB)
        color = _COLOR_NAMES[color_index]

    if not color:
        color = _DEFAULT_COLOR_NAME

    if type(color) is not str:
        raise TypeError("'color' must be a str")

    if color not in _COLOR_NAME_TO_RGB:
        msg = "'color' must be one of " + ", ".join(_COLOR_NAME_TO_RGB)
        raise ValueError(msg)

    colors = [_rgb_to_bgr(item) for item in _COLOR_NAME_TO_RGB[color]]
    # print(colors)
    color, color_text = colors
    # print((left, top), (right, bottom))

    image = _cv2.rectangle(image, (left, top), (right, bottom), color, 2)

    if label:

        _, image_width, _ = image.shape
        fontface = _cv2.FONT_HERSHEY_TRIPLEX  # フォントの種類
        fontscale = 0.5  # 文字のスケール
        thickness = 1  # 文字の太さ

        (label_width, label_height), baseline = _cv2.getTextSize(label, fontface, fontscale, thickness)

        # label_image =  _get_label_image(label, color_text, color)
        # label_height, label_width, _ = label_image.shape

        rectangle_height, rectangle_width = 1 + label_height, 1 + label_width

        rectangle_bottom = top
        rectangle_left = max(0, min(left - 1, image_width - rectangle_width))

        rectangle_top = rectangle_bottom - rectangle_height
        rectangle_right = rectangle_left + rectangle_width

        label_top = rectangle_top + 1

        if rectangle_top < 0:
            rectangle_top = top
            rectangle_bottom = rectangle_top + label_height + 1

            label_top = rectangle_top

        label_left = rectangle_left + 1
        label_bottom = label_top + label_height
        label_right = label_left + label_width

        rec_left_top = (rectangle_left, rectangle_top)
        rec_right_bottom = (rectangle_right, rectangle_bottom)

        _cv2.rectangle(image, rec_left_top, rec_right_bottom, color, -1)

        # image[label_top:label_bottom, label_left:label_right, :] = label

        _cv2.putText(image,
            text = label,
            org=(label_left, int((label_bottom))),
            fontFace = fontface,
            fontScale=fontscale,
            color=(0, 0, 0),
            thickness = thickness,
            lineType=_cv2.LINE_4)
    return image