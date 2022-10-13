"""Create a camera object that can be used to read frames from a video file."""

import math
import os
import warnings
from functools import cached_property
from typing import (
    Dict,
    Generator,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)
from xml.etree import ElementTree

import cv2 as cv
import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.decomposition import PCA

from soccertrack.utils import (
    ImageIterator,
    MovieIterator,
    cv2pil,
    logger,
    make_video,
    tqdm,
)


class Camera:
    def __init__(
        self,
        video_path: str,
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

        if keypoint_xml is not None:
            source_keypoints, target_keypoints = read_pitch_keypoints(
                keypoint_xml, "video"
            )
            self.source_keypoints = source_keypoints
            self.target_keypoints = target_keypoints

            ## TODO: add option to not undistort points maybe?
            source_keypoints = self.undistort_points(source_keypoints).squeeze()
            proj_error = np.linalg.norm(
                self.video2pitch(source_keypoints) - target_keypoints, axis=-1
            ).mean()
            logger.debug(f"Camera `{self.label}`: projection error = {proj_error:.2f}m")
        else:
            self.source_keypoints = None
            self.target_keypoints = None
            
    def load_calibration_params(self):
        # self.mapx, self.mapy = find_intrinsic_camera_parameters(calibration_video_path, return_mappings=True)
        calibration_video_path = self.calibration_video_path

        if self.camera_matrix_path:
            camera_matrix = np.load(self.camera_matrix_path)
        if self.distortion_coefficients_path:
            distortion_coefficients = np.load(self.distortion_coefficients_path)

        if self.camera_matrix is None or self.distortion_coefficients is None:
            if calibration_video_path is not None:
                (
                    self.camera_matrix,
                    self.distortion_coefficients,
                    self.mapx,
                    self.mapy,
                ) = find_intrinsic_camera_parameters(calibration_video_path)

                self.camera_matrix_path = calibration_video_path + ".camera_matrix.npy"
                self.distortion_coefficients_path = (
                    calibration_video_path + ".distortion_coefficients.npy"
                )
                
                # save this somewhere else
                # np.save(self.camera_matrix_path, self.camera_matrix)
                # np.save(self.distortion_coefficients_path, self.distortion_coefficients)
            else:
                self.camera_matrix = np.eye(3)
                self.distortion_coefficients = np.zeros(4)
                dim = (self.w, self.h)
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
        cap = cv.VideoCapture(self.video_path)
        cap.set(cv.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()
        return frame
            
    def iter_frames(self, calibrate: bool = False, crop: bool = False) -> Generator[NDArray, None, None]:
        """Iterate over frames of video.

        Yields:
            NDArray: frame of video.
        """
        for frame in MovieIterator(self.video_path):
            yield frame
            
    def movie_iterator(self, calibrate: bool = True, crop: bool = True) -> Generator[NDArray, None, None]:
        """Create a movie iterator.

        Args:
            calibrate (bool, optional): Option to calibrate frames. Defaults to False.

        Yields:
            Generator[NDArray]: frames of video.

        """
        movie_iterator = MovieIterator(self.video_path)
        if not calibrate:
            for i, frame in enumerate(movie_iterator):
                yield frame
                
        for i, frame in enumerate(movie_iterator):
            frame = self.undistort_image(frame)
            if crop:
                x1, y1, x2, y2 = self.roi
                yield frame[y1:y2, x1:x2]
            else:
                yield frame

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

    def save_calibrated_video(
        self, save_path: str, plot_pitch_keypoints: bool = True, **kwargs
    ) -> None:
        """Save a video with undistorted frames.

        Args:
            save_path (str): path to save video.

        Note:
            See utils.make_video for available kwargs.

        """
        movie_iterator = self.movie_iterator(calibrate=True)
        roi = self.roi

        make_video(
            movie_iterator, 
            outpath=save_path,
            **kwargs
        )

    def visualize_candidate_detections(
        self,
        candidate_detections: Dict[int, List],
        save_path: str,
        plot_pitch_keypoints: bool = True,
        calibrate: bool = True,
        filter_range: bool = True,
        frames: Union[int, str] = "all",
        crop: bool = False,
        **kwargs,
    ) -> None:

        """Visualize candidate detections.

        Args:
            candidate_detections (Dict[int, List[CandidateDetection]]): dictionary of candidate detections.
            save_path (str): path to save video.
            plot_pitch_keypoints (bool): Option to plot pitch keypoints. Defaults to False.
            calibrate (bool): Option to calibrate frames. Defaults to True.

        Note:
            kwargs are passed to `make_video`, so it is recommended that you refere to the documentation for `make_video`.
        """
        movie_iterator = self.movie_iterator(calibrate=calibrate, crop=crop)

        output_frames = []
        for i, frame in tqdm(
            enumerate(movie_iterator), level="DEBUG", desc="Sorting frames"
        ):
            if isinstance(frames, int):
                if i > frames:
                    break
            if i not in candidate_detections:
                continue

            for candidate_detection in candidate_detections[i]:
                if filter_range:
                    if not candidate_detection.in_range:
                        continue
                if self.label != candidate_detection.camera.label:
                    continue
                x1, y1, x2, y2 = list(
                    map(
                        int,
                        candidate_detection.get_absolute_bounding_box(BBFormat.XYX2Y2),
                    )
                )

                cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            if plot_pitch_keypoints:
                for pitch_keypoint in self.source_keypoints:
                    cv.circle(frame, pitch_keypoint.astype(int), 1, (0, 0, 255), -1)
            output_frames.append(frame)
        
        if len(output_frames) == 0:
            logger.error("No frames to save, exiting")
        else:
            make_video(output_frames, save_path, **kwargs)

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
            newcameramtx, roi = cv.getOptimalNewCameraMatrix(
                mtx, dist, (w, h), 1, (w, h)
            )
            dst = cv.undistortPoints(points, mtx, dist, None, newcameramtx)
            dst = dst.reshape(-1, 2)
            x, y, w, h = roi
            dst = dst - np.asarray([x, y])
        elif self.calibration_method == "fisheye":
            mtx_new = cv.fisheye.estimateNewCameraMatrixForUndistortRectify(
                mtx, dist, (w, h), np.eye(3), balance=1.0
            )
            points = np.expand_dims(points, axis=1)
            dst = np.squeeze(cv.fisheye.undistortPoints(points, mtx, dist, P=mtx_new))
        return dst

    def undistort_image(self, image: NDArray) -> NDArray:
        undistorted_image = cv.remap(
            image,
            self.mapx,
            self.mapy,
            interpolation=cv.INTER_LINEAR,
            borderMode=cv.BORDER_CONSTANT,
        )
        return undistorted_image

    @cached_property
    def video_fps(self) -> int:
        """Get video fps.

        Returns:
            int: video fps.

        """
        return MovieIterator(self.video_path).video_fps

    @property
    def keypoint_map(self) -> Dict[Tuple[int, int], Tuple[int, int]]:
        """Get dictionary of pitch keypoints in pitch space to pixel space.

        Returns:
            Dict: dictionary of pitch keypoints in pitch space to pixel space.

        """
        if self.source_keypoints is None:
            return None
        return {
            tuple(key): value
            for key, value in zip(self.target_keypoints, self.source_keypoints)
        }

    @property
    def w(self) -> int:
        """Width of video frames.

        Returns:
            int: width

        """
        return MovieIterator(self.video_path).img_width

    @property
    def h(self) -> int:
        """Height of video frames.

        Returns:
            int: height

        """
        return MovieIterator(self.video_path).img_height

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

        H, *_ = cv.findHomography(
            self.source_keypoints, self.target_keypoints, cv.RANSAC, 5.0
        )
        return H

    @property
    def fps(self) -> int:
        """Get video fps.

        Returns:
            int: video fps.

        """
        return MovieIterator(self.video_path).video_fps

    @property
    def roi(self) -> Tuple[int, int, int, int]:
        if self.keypoint_map is None:
            return 0, 0, self.w, self.h
        elif self.calibration_method == "zhang":
            dim = (self.w, self.h)
            K = self.camera_matrix
            D = self.distortion_coefficients
            _, (x1, y1, w, h) = cv.getOptimalNewCameraMatrix(K, D, dim, 1, dim)
            return x1, y1, x1 + w, y1 + h
        else:
            keypoint_map = self.keypoint_map
            source_keypoints = self.source_keypoints

            _cx = sum(self.x_range) / 2
            _cy = sum(self.y_range) / 2
            
            if (_cx, _cy) in keypoint_map:
                cx, cy = keypoint_map[(_cx, _cy)]
            else:
                cx, cy = self.pitch2video((_cx, _cy))
            width = source_keypoints[:, 0].max() - source_keypoints[:, 0].min()
            height = source_keypoints[:, 1].max() - source_keypoints[:, 1].min()

            width *= 1.5
            height = width * (9 / 16)  # use 16:9 aspect ratio

            x1 = cx - width / 2
            y1 = cy - height / 2
            x2 = cx + width / 2
            y2 = cy + height / 2
            return int(x1), int(y1), int(x2), int(y2)


def read_pitch_keypoints(
    xmlfile: str, annot_type: str
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
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


def find_intrinsic_camera_parameters(
    video_path: Union[str, List[str]],
    fps: int = 1,
    scale: int = 4,
    save_path: str = None,
    draw_on_save: bool = False,
    points_to_use: int = 50,
    calibration_method: str = "zhang",
    return_mappings: bool = True,
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Find intrinsic camera parameters.

    Args:
        video_path (str): path to calibration video. Must contain a checkerboard.
        fps (int, optional): fps to use. Defaults to 1.
        s (int, optional): scale Defaults to 4.
        save_path (str, optional): path to save. Defaults to None.

    Returns:
        Tuple[NDArray[np.float64], NDArray[np.float64]]: camera matrix and distortion coefficients.

    """

    def _detect_corners(iterator, scale, fps):
        assert len(iterator) > 0, "No images found in video."

        if isinstance(iterator, MovieIterator):
            nskip = math.ceil(iterator.video_fps / fps)
        else:
            nskip = 1

        # Arrays to store object points and image points from all the images.
        objpoints = []  # 3d point in real world space
        imgpoints = []  # 2d points in image plane.

        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((5 * 9, 3), np.float32)
        objp[:, :2] = np.mgrid[0:9, 0:5].T.reshape(-1, 2)

        all_corners = []
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.001)

        for i, frame in enumerate(tqdm(iterator, total=len(iterator))):
            if i % nskip != 0:
                continue

            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

            # scale image for faster detection
            gray_small = cv.resize(gray, None, fx=1 / scale, fy=1 / scale)

            # Find the chess board corners
            ret, corners = cv.findChessboardCorners(
                image=gray_small, patternSize=(9, 5)
            )

            if ret is True:
                corners *= scale
                corners = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                imgpoints.append(corners)
                objpoints.append(objp)
        return objpoints, imgpoints

    def _select_images(imgpoints, objpoints, points_to_use):
        if len(imgpoints) < points_to_use:
            return imgpoints, objpoints

        logger.info(
            f"Too many ({len(imgpoints)}) checkerboards found. Selecting {points_to_use}."
        )

        X = np.asarray([np.ravel(x) for x in imgpoints])
        pca = PCA(n_components=1)
        Xt = np.ravel(pca.fit_transform(X))

        # sort images by their distance to the origin
        idxs = np.argsort(Xt)
        objpoints = [objpoints[i] for i in idxs]
        imgpoints = [imgpoints[i] for i in idxs]

        # select points to use
        x_range = np.linspace(0, len(imgpoints), points_to_use, endpoint=False)
        objpoints = [objpoints[int(i)] for i in x_range]
        imgpoints = [imgpoints[int(i)] for i in x_range]

        return imgpoints, objpoints

    # Support multiple video files
    video_paths = [video_path] if isinstance(video_path, str) else video_path

    # Support image directories and video files
    iterators = [
        ImageIterator(video_path)
        if os.path.isdir(video_path)
        else MovieIterator(video_path)
        for video_path in video_paths
    ]

    # Find corners in each video
    objpoints, imgpoints = [], []
    for iterator in iterators:
        _objpoints, _imgpoints = _detect_corners(iterator, scale, fps)
        objpoints += _objpoints
        imgpoints += _imgpoints
    logger.debug(f"imgpoints found: {len(imgpoints)}")

    # Select frames to use based on PCA Variance
    objpoints, imgpoints = _select_images(objpoints, imgpoints, points_to_use)
    logger.debug(f"imgpoints used: {len(imgpoints)}")

    logger.info("Computing calibration parameters...")

    if isinstance(iterators[0], MovieIterator):
        dim = iterators[0].img_width, iterators[0].img_height
    else:
        dim = iterators[0].imgs[0].shape[:-1]
    if calibration_method.lower() == "zhang":
        # TODO: Abstract this into a function.
        logger.info("Using Zhang's method.")

        ret, K, D, rvecs, tvecs = cv.calibrateCamera(
            objpoints, imgpoints, dim, None, None
        )
        newcameramtx, roi = cv.getOptimalNewCameraMatrix(K, D, dim, 1, dim)
        mapx, mapy = cv.initUndistortRectifyMap(K, D, None, newcameramtx, dim, 5)
    elif calibration_method.lower() == "fisheye":
        # TODO: Abstract this into a function.
        logger.info("Using Fisheye method.")
        N_OK = len(objpoints)

        K = np.zeros((3, 3))
        D = np.zeros((4, 1))
        rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
        tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
        objpoints = np.expand_dims(np.asarray(objpoints), -2)
        # FIXME: Fisheye calibration fails with CALIB_CHECK_COND.
        # https://stackoverflow.com/q/49038464
        # calibration_flags = cv.fisheye.CALIB_RECOMPUTE_EXTRINSIC+cv.fisheye.CALIB_CHECK_COND+cv.fisheye.CALIB_FIX_SKEW
        calibration_flags = (
            cv.fisheye.CALIB_RECOMPUTE_EXTRINSIC + cv.fisheye.CALIB_FIX_SKEW
        )

        ret, K, D, rvecs, tvecs = cv.fisheye.calibrate(
            objpoints,
            imgpoints,
            dim,
            K,
            D,
            rvecs,
            tvecs,
            calibration_flags,
            (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 1e-6),
        )
        balance = 1  # 0~1 how much to crop
        new_K = cv.fisheye.estimateNewCameraMatrixForUndistortRectify(
            K, D, dim, np.eye(3), balance=balance
        )
        mapx, mapy = cv.fisheye.initUndistortRectifyMap(
            K, D, np.eye(3), new_K, dim, cv.CV_32FC1
        )
    else:
        raise ValueError("Calibration method must be `zhang` or `fisheye`.")

    logger.info("Finished computing calibration parameters.")

    return K, D, mapx, mapy


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
