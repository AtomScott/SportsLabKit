"""Create a camera object that can be used to read frames from a video file."""

from __future__ import annotations

from typing import (
    Dict,
    Generator,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)
from xml.etree import ElementTree

import cv2 as cv
import numpy as np
from numpy.typing import ArrayLike, NDArray

from sportslabkit.camera.calibrate import find_intrinsic_camera_parameters
from sportslabkit.camera.videoreader import VideoReader
from sportslabkit.types.types import _pathlike
from sportslabkit.utils import logger


class Camera(VideoReader):
    def __init__(
        self,
        video_path: _pathlike,
        threaded: bool = False,
        queue_size: int = 10,
        keypoint_xml: Optional[str] = None,
        x_range: Optional[Sequence[float]] = (0, 105),
        y_range: Optional[Sequence[float]] = (0, 68),
        camera_matrix: Optional[ArrayLike] = None,
        camera_matrix_path: Optional[str] = None,
        distortion_coefficients: Optional[str] = None,
        distortion_coefficients_path: Optional[str] = None,
        calibration_video_path: Optional[str] = None,
        calibration_method: str = "zhang",
        label: str = "",
        verbose: int = 0,
    ):
        """Class for handling camera calibration and undistortion.

        Args:
            video_path (str): path to video file.
            threaded (bool, optional): whether to use a threaded video reader. Defaults to False.
            queue_size (int, optional): size of queue for threaded video reader. Defaults to 10.
            keypoint_xml (str): path to file containing a mapping from pitch coordinates to video.
            x_range (Sequence[float]): pitch range to consider in x direction.
            y_range (Sequence[float]): pitch range to consider in y direction.
            camera_matrix (Optional[Union[str, np.ndarray]]): numpy array or path to file containing camera matrix.
            distortion_coefficients (Optional[Union[str, np.ndarray]]): numpy array or path to file containing distortion coefficients.
            calibration_video_path (Optional[str]): path to video file with checkerboard to use for calibration.
            label (str, optional): label for camera. Defaults to "".
            verbose (int, optional): verbosity level. Defaults to 0.
        Attributes:
            camera_matrix (np.ndarray): numpy array containing camera matrix.
            distortion_coefficients (np.ndarray): numpy array containing distortion coefficients.
            keypoint_map (Mapping): mapping from pitch coordinates to video.
            H (np.ndarray): homography matrix from image to pitch.
            w (int): width of video.
            h (int): height of video.

        """
        if threaded:
            logger.warning("Threaded video reader is buggy. Use at your own risk.")
        super().__init__(video_path, threaded, queue_size)
        self.label = label

        self.video_path = str(video_path)
        self.calibration_method = calibration_method

        self.camera_matrix = camera_matrix
        self.distortion_coefficients = distortion_coefficients
        self.camera_matrix_path = camera_matrix_path
        self.distortion_coefficients_path = distortion_coefficients_path
        self.calibration_video_path = calibration_video_path
        self.load_calibration_params()

        self.x_range = x_range
        self.y_range = y_range

        # Remove leading singleton dimension when returning single frames. Defaults to True.
        self.remove_leading_singleton = True

        if keypoint_xml is not None:
            source_keypoints, target_keypoints = read_pitch_keypoints(keypoint_xml, "video")
            self.source_keypoints = source_keypoints
            self.target_keypoints = target_keypoints

            ## TODO: add option to not undistort points maybe?
            source_keypoints = self.undistort_points(source_keypoints).squeeze()
            proj_error = np.linalg.norm(self.video2pitch(source_keypoints) - target_keypoints, axis=-1).mean()
            logger.debug(f"Camera `{self.label}`: projection error = {proj_error:.2f}m")
        else:
            self.source_keypoints = None
            self.target_keypoints = None

    def load_calibration_params(self):
        # self.mapx, self.mapy = find_intrinsic_camera_parameters(calibration_video_path, return_mappings=True)
        calibration_video_path = self.calibration_video_path

        if self.camera_matrix_path:
            np.load(self.camera_matrix_path)
        if self.distortion_coefficients_path:
            np.load(self.distortion_coefficients_path)

        if self.camera_matrix is None or self.distortion_coefficients is None:
            if calibration_video_path is not None:
                (
                    self.camera_matrix,
                    self.distortion_coefficients,
                    self.mapx,
                    self.mapy,
                ) = find_intrinsic_camera_parameters(calibration_video_path)

                self.camera_matrix_path = calibration_video_path + ".camera_matrix.npy"
                self.distortion_coefficients_path = calibration_video_path + ".distortion_coefficients.npy"

                # save this somewhere else
                # np.save(self.camera_matrix_path, self.camera_matrix)
                # np.save(self.distortion_coefficients_path, self.distortion_coefficients)
            else:
                self.camera_matrix = np.eye(3)
                self.distortion_coefficients = np.zeros(4)
                dim = (self.frame_width, self.frame_height)
                newcameramtx, _ = cv.getOptimalNewCameraMatrix(
                    self.camera_matrix, self.distortion_coefficients, dim, 1, dim
                )
                self.mapx, self.mapy = cv.initUndistortRectifyMap(
                    self.camera_matrix,
                    self.distortion_coefficients,
                    None,
                    newcameramtx,
                    dim,
                    5,
                )

    def get_frame(self, frame_idx: int) -> np.ndarray:
        """Get frame from video.

        Args:
            frame (int): frame

        Returns:
            np.ndarray: frame
        """
        return self[frame_idx]

    def iter_frames(self, calibrate: bool = False, crop: bool = False) -> Generator[NDArray, None, None]:
        """Iterate over frames of video.

        Yields:
            NDArray: frame of video.
        """
        return self

    def batch_frames(
        self, batch_size: int = 32, calibrate: bool = False, crop: bool = False
    ) -> Generator[NDArray, None, None]:
        """Iterate over frames of video.

        Yields:
            NDArray: frame of video.
        """
        frames = []
        for frame in self:
            frames.append(frame)
            if len(frames) == batch_size:
                yield np.stack(frames)
                frames = []
        if len(frames) > 0:
            yield np.stack(frames)

    def video2pitch(self, pts: ArrayLike) -> NDArray[np.float64]:
        """Convert image coordinates to pitch coordinates.

        Args:
            video_pts (np.ndarray): points in image coordinate space

        Returns:
            np.ndarray: points in pitch coordinate

        """
        if pts.ndim == 1:
            pts = pts.reshape(1, -1)

        pitch_pts = cv.perspectiveTransform(np.asarray([pts], dtype=np.float32), self.H)
        return pitch_pts

    def pitch2video(self, pitch_pts: ArrayLike) -> NDArray[np.float64]:
        """Converts pitch coordinates to image coordinates.

        Args:
            pitch_pts (ArrayLike): coordinates in pitch coordinate space.

        Raises:
            NotImplementedError: this method is not implemented.

        Returns:
            NDArray[np.float64]: ...

        """
        # TODO: implement this
        raise NotImplementedError

    def undistort_points(self, points: ArrayLike) -> NDArray[np.float64]:
        """Undistort points with the camera matrix and distortion coefficients.

        Args:
            points (ArrayLike): [description]

        Returns:
            NDArray[np.float64]: [description]

        Note:
            Not to be confused with video2pitch which uses a homography transformation.
        """

        mtx = self.camera_matrix
        dist = self.distortion_coefficients
        w = self.w
        h = self.h

        if self.calibration_method == "zhang":
            newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
            dst = cv.undistortPoints(points, mtx, dist, None, newcameramtx)
            dst = dst.reshape(-1, 2)
            x, y, w, h = roi
            dst = dst - np.asarray([x, y])
        elif self.calibration_method == "fisheye":
            mtx_new = cv.fisheye.estimateNewCameraMatrixForUndistortRectify(mtx, dist, (w, h), np.eye(3), balance=1.0)
            points = np.expand_dims(points, axis=1)
            dst = np.squeeze(cv.fisheye.undistortPoints(points, mtx, dist, P=mtx_new))
        return dst

    def pitch_contour(self, frame_num):
        pass

    def undistort_image(self, image: NDArray) -> NDArray:
        undistorted_image = cv.remap(
            image,
            self.mapx,
            self.mapy,
            interpolation=cv.INTER_LINEAR,
            borderMode=cv.BORDER_CONSTANT,
        )
        return undistorted_image

    @property
    def dtype(self):
        return np.uint8

    @property
    def shape(self):
        return (self.number_of_frames, *self.frame_shape)

    @property
    def ndim(self):
        return len(self.shape) + 1

    @property
    def size(self):
        return np.product(self.shape)

    def min(self):
        return 0

    def max(self):
        return 255

    @property
    def keypoint_map(self) -> Dict[Tuple[int, int], Tuple[int, int]]:
        """Get dictionary of pitch keypoints in pitch space to pixel space.

        Returns:
            Dict: dictionary of pitch keypoints in pitch space to pixel space.

        """
        if self.source_keypoints is None:
            return None
        return {tuple(key): value for key, value in zip(self.target_keypoints, self.source_keypoints)}

    @property
    def A(self) -> NDArray[np.float64]:
        """Calculate the affine transformation matrix from pitch to video space.

        Returns:
            NDArray[np.float64]: affine transformation matrix.

        """

        A, *_ = cv.estimateAffinePartial2D(self.source_keypoints, self.target_keypoints)
        return A

    @property
    def H(self) -> NDArray[np.float64]:
        """Calculate the homography transformation matrix from pitch to video space.

        Returns:
            NDArray[np.float64]: homography transformation matrix.

        """

        H, *_ = cv.findHomography(self.source_keypoints, self.target_keypoints, cv.RANSAC, 5.0)
        return H


def read_pitch_keypoints(xmlfile: str, annot_type: str) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Read pitch keypoints from xml file.

    Args:
        xmlfile (str): path to xml file.
        annot_type (str): type of annotation. Either 'pitch' or 'video'.

    Raises:
        ValueError: if annotation type is not 'pitch' or 'video'.

    Returns:
        Tuple[NDArray[np.float64], NDArray[np.float64]]: pitch keypoints and video keypoints.

    """
    tree = ElementTree.parse(xmlfile)
    root = tree.getroot()

    src = []
    dst = []

    if annot_type == "video":
        for child in root:
            for c in child:
                d = c.attrib
                if d != {}:
                    dst.append(eval(d["label"]))
                    src.append(eval(d["points"]))

    elif annot_type == "frame":
        for child in root:
            d = child.attrib
            if d != {}:
                dst.append(eval(d["label"]))
                src.append(eval(child[0].attrib["points"]))
    else:
        raise ValueError("Annotation type must be `video` or `frame`.")

    src = np.asarray(src)
    dst = np.asarray(dst)

    assert src.size != 0, "No keypoints found in XML file."
    return src, dst


def load_cameras(camera_info: List[Mapping]) -> List[Camera]:
    """Load cameras from a list of dictionaries containing camera information.

    Args:
        camera_info (List[Mapping]): list of dictionaries containing camera information.

    Returns:
        List[Camera]: list of cameras objects.

    """
    cameras = []
    for cam_info in camera_info:
        camera = Camera(
            video_path=cam_info.video_path,
            keypoint_xml=cam_info.keypoint_xml,
            camera_matrix=cam_info.camera_matrix,
            camera_matrix_path=cam_info.camera_matrix_path,
            distortion_coefficients=cam_info.distortion_coefficients,
            distortion_coefficients_path=cam_info.distortion_coefficients_path,
            calibration_video_path=cam_info.calibration_video_path,
            x_range=cam_info.x_range,
            y_range=cam_info.y_range,
            label=cam_info.label,
        )
        cameras.append(camera)
    return cameras
