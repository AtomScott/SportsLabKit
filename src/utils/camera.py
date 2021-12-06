import math
import os
import warnings
from functools import cached_property
from typing import List, Mapping, Optional, Sequence, Tuple, Union
from xml.etree import ElementTree

import cv2 as cv
import numpy as np
from sklearn.decomposition import PCA
from tqdm import tqdm

from src.utils import MovieIterator, cv2pil, make_video


class Camera:
    def __init__(
        self,
        video_path: str,
        keypoint_xml: str,
        x_range: Sequence[float],
        y_range: Sequence[float],
        camera_matrix: Optional[np.ndarray],
        camera_matrix_path: Optional[str],
        distortion_coefficients: Optional[str],
        distortion_coefficients_path: Optional[str],
        calibration_video_path: Optional[str],
        label: str = "",
    ):
        """Class for handling camera calibration and undistortion.

        Args:
            video_path (str): path to video file.
            keypoint_xml (str): path to file containing a mapping from pitch coordinates to video.
            x_range (Sequence[float]): pitch range to consider in x direction.
            y_range (Sequence[float]): pitch range to consider in y direction.
            camera_matrix (Optional[Union[str, np.ndarray]]): numpy array or path to file containing camera matrix.
            distortion_coefficients (Optional[Union[str, np.ndarray]]): numpy array or path to file containing distortion coefficients.
            calibration_video_path (Optional[str]): path to video file with checkerboard to use for calibration.

        Attributes:
            camera_matrix (np.ndarray): numpy array containing camera matrix.
            distortion_coefficients (np.ndarray): numpy array containing distortion coefficients.
            keypoint_map (Mapping): mapping from pitch coordinates to video.
            H (np.ndarray): homography matrix from image to pitch.
            w (int): width of video.
            h (int): height of video.
        """
        self.label = label
        source_keypoints, target_keypoints = read_pitch_keypoints(keypoint_xml, "video")

        self.source_keypoints = source_keypoints
        self.target_keypoints = target_keypoints

        self.camera_matrix_path = camera_matrix_path
        self.distortion_coefficients_path = distortion_coefficients_path

        self.video_path = video_path

        if camera_matrix is None or distortion_coefficients is None:
            if (
                calibration_video_path is not None
                and distortion_coefficients_path is not None
            ):
                assert os.path.exists(camera_matrix_path)
                assert os.path.exists(distortion_coefficients_path)
                self.camera_matrix = np.load(camera_matrix_path)
                self.distortion_coefficients = np.load(distortion_coefficients_path)

            elif calibration_video_path is not None:
                (
                    camera_matrix,
                    distortion_coefficients,
                ) = find_intrinsic_camera_parameters(calibration_video_path)
                self.camera_matrix = camera_matrix
                self.distortion_coefficients = distortion_coefficients

                self.camera_matrix_path = calibration_video_path + ".camera_matrix.npy"
                self.distortion_coefficients_path = (
                    calibration_video_path + ".distortion_coefficients.npy"
                )
                np.save(self.camera_matrix_path, camera_matrix)
                np.save(self.distortion_coefficients_path, distortion_coefficients)
            else:
                warnings.warn("Insufficient information to perform camera calibration!")

        self.x_range = x_range
        self.y_range = y_range

    def video2pitch(self, video_pts: np.ndarray) -> np.ndarray:
        """Convert image coordinates to pitch coordinates.

        Args:
            video_pts (np.ndarray): points in image coordinate space

        Returns:
            np.ndarray: points in pitch coordinate
        """
        if video_pts.ndim == 1:
            pts = video_pts.reshape(1, -1)

        pitch_pts = cv.perspectiveTransform(np.asarray([pts], dtype=np.float32), self.H)
        return pitch_pts

    def pitch_to_video(self, pitch_pts: np.ndarray) -> np.ndarray:
        # TODO: implement this
        raise NotImplementedError
        return ...

    def undistort_video(self, video_path: str, out_path: str):
        mtx = self.calibration_matrix
        dist = self.distortion_coefficients
        if mtx is None or dist is None:
            assert (
                self.checkerboard_video is not None
            ), "Checkerboard is necessary if calibration params are not available."

        movie_iterator = MovieIterator(video_path)

        w = movie_iterator.img_width
        h = movie_iterator.img_height

        newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
        undistorted_frames = []
        for i, frame in enumerate(tqdm(movie_iterator, total=len(movie_iterator))):
            dst = cv.undistort(frame, mtx, dist, None, newcameramtx)
            x, y, w, h = roi
            dst = dst[y : y + h, x : x + w]
            undistorted_frames.append(dst)

        self.src_kps = self.undistort_points(self.src_kps)
        make_video(undistorted_frames, movie_iterator.video_fps, out_path)
        return out_path

    def undistort_points(self, points, checkerboard_video=None):
        mtx = self.calibration_matrix
        dist = self.distortion_coefficients
        if mtx is None or dist is None:

            assert (
                checkerboard_video is not None
            ), "Checkerboard is necessary if calibration params are not available."
            mtx, dist = find_intrinsic_camera_parameters(checkerboard_video)
            self.mtx = mtx
            self.dist = dist

        w = self.w
        h = self.h

        newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
        dst = cv.undistortPoints(points, mtx, dist, None, newcameramtx)
        dst = dst.reshape(-1, 2)

        x, y, w, h = roi
        dst = dst - np.asarray([x, y])

        return dst

    def movie_iterator(self, calibrate: bool = False):
        movie_iterator = MovieIterator(self.video_path)
        if not calibrate:
            for i, frame in enumerate(movie_iterator):
                yield frame

        mtx = self.camera_matrix
        dist = self.distortion_coefficients
        w = self.w
        h = self.h

        newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
        for i, frame in enumerate(movie_iterator):
            dst = cv.undistort(frame, mtx, dist, None, newcameramtx)
            x, y, w, h = roi
            dst = dst[y : y + h, x : x + w]
            yield dst

    def save_calibrated_frames(self, save_path: str) -> None:
        """Save calibrated frames as a video to disk.

        Args:
            save_path (str): path to save video to.
        """
        movie_iterator = self.movie_iterator(calibrate=True)
        make_video(movie_iterator, self.video_fps, save_path)

    @cached_property
    def keypoint_map(self) -> Mapping:
        return {
            tuple(key): value
            for key, value in zip(self.target_keypoints, self.source_keypoints)
        }

    @cached_property
    def w(self) -> int:
        return MovieIterator(self.video_path).img_width

    @cached_property
    def h(self) -> int:
        return MovieIterator(self.video_path).img_height

    @cached_property
    def A(self) -> np.ndarray:
        A, *_ = cv.estimateAffinePartial2D(self.source_keypoints, self.target_keypoints)
        return A

    @cached_property
    def H(self) -> np.ndarray:
        H, *_ = cv.findHomography(
            self.source_keypoints, self.target_keypoints, cv.RANSAC, 5.0
        )
        return H


def read_pitch_keypoints(
    xmlfile: str, annot_type: str
) -> Tuple[np.ndarray, np.ndarray]:
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

    src = np.asarray(src)
    dst = np.asarray(dst)

    if src.size == 0:
        print("Array is empty. Is annot_type correct?")

    return src, dst


def find_intrinsic_camera_parameters(
    video_path: str, fps: int = 1, s: int = 4, save_path: str = None
):
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((5 * 9, 3), np.float32)
    objp[:, :2] = np.mgrid[0:9, 0:5].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.

    # list to store images with drawen corners
    imgs = []

    movie_iterator = MovieIterator(video_path)
    w = movie_iterator.img_width
    h = movie_iterator.img_height

    nskip = math.ceil(movie_iterator.video_fps / fps)

    for i, frame in enumerate(tqdm(movie_iterator, total=len(movie_iterator))):
        if i % nskip != 0:
            continue

        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # scale image for faster detection
        gray_small = cv.resize(gray, None, fx=1 / s, fy=1 / s)

        # Find the chess board corners
        ret, corners = cv.findChessboardCorners(image=gray_small, patternSize=(9, 5))

        if ret == True:
            corners *= s
            corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

            objpoints.append(objp)
            imgpoints.append(corners2)

            if save_path:
                # Draw and display the corners
                img = cv.drawChessboardCorners(frame, (9, 5), corners2, ret)
                imgs.append(img)

    # speed up
    X = np.asarray([np.ravel(x) for x in imgpoints])
    pca = PCA(n_components=1)
    Xt = np.ravel(pca.fit_transform(X))
    idxs = np.argsort(Xt)

    if len(imgpoints) > 50:
        objpoints = [objpoints[i] for i in idxs]
        imgpoints = [imgpoints[i] for i in idxs]
        if save_path:
            imgs = [imgs[i] for i in idxs]

        print(f"Too many ({len(imgpoints)}) checkerboards found. Selecting 50.")
        objpoints = [
            objpoints[int(i)]
            for i in np.linspace(0, len(imgpoints), 50, endpoint=False)
        ]
        imgpoints = [
            imgpoints[int(i)]
            for i in np.linspace(0, len(imgpoints), 50, endpoint=False)
        ]
        if save_path:
            imgs = [
                imgs[int(i)] for i in np.linspace(0, len(imgpoints), 50, endpoint=False)
            ]

    if save_path:
        os.makedirs(save_path, exist_ok=True)
        for i, img in enumerate(imgs):
            cv2pil(img).save(os.path.join(save_path, f"{i}.png"))

    print("Computing calibration parameters...")
    if len(imgpoints) > 10:
        print(
            f"(This will take time since many ({len(imgpoints)}) checkerboards were found.)"
        )

    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None
    )
    return mtx, dist


def load_cameras(camera_info: List[Mapping]) -> List[Camera]:
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
