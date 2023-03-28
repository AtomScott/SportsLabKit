from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray
from sklearn.decomposition import PCA
from vidgear.gears.stabilizer import Stabilizer

from soccertrack import Camera
from soccertrack.logger import logger, tqdm
from soccertrack.types import _pathlike
from soccertrack.utils import make_video


def detect_corners(
    camera: Camera,
    scale: float,
    fps: float,
    num_corners_x: int = 5,
    num_corners_y: int = 9,
):
    """Detects the corners in a set of images.

    This function detects the corners in a set of images using the cv2.findChessboardCorners function. The input images are
    downsampled using the scale parameter to increase processing speed. The function returns a tuple of two lists,
    `objpoints` and `imgpoints`, containing 3D points in real world space and 2D points in the image plane, respectively.

    Args:
        camera (Camera): A `Camera` object representing the camera from which the images were captured.
        scale (float): The scale to resize the images for faster detection. Must be greater than 0.
        fps (float): The frames per second to process. Must be greater than 0.
        num_corners_x (int, optional): The number of corners along the x-axis in the checkerboard pattern. Defaults to 5.
        num_corners_y (int, optional): The number of corners along the y-axis in the checkerboard pattern. Defaults to 9.

    Returns:
        tuple: A tuple containing two lists, `objpoints` and `imgpoints`. `objpoints` is a list of 3D points in real world space, and `imgpoints` is a list of 2D points in the image plane.

    Raises:
        ValueError: If `scale` or `fps` is less than or equal to 0.
        AssertionError: If no images are found in the video.
    """
    if scale <= 0:
        raise ValueError("The scale must be greater than 0.")
    if fps <= 0:
        raise ValueError("The fps must be greater than 0.")

    n_frames = len(camera)
    assert n_frames > 0, "No images found in video."

    nskip = np.ceil(camera.frame_rate / fps)

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.

    objp = np.zeros((num_corners_x * num_corners_y, 3), np.float32)
    objp[:, :2] = np.mgrid[0:num_corners_y, 0:num_corners_x].T.reshape(-1, 2)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)

    for i in tqdm(range(n_frames)):
        if i % nskip != 0:
            continue

        frame = camera[i]

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_small = cv2.resize(gray, None, fx=1 / scale, fy=1 / scale)

        ret, corners = cv2.findChessboardCorners(
            gray_small, (num_corners_y, num_corners_x)
        )
        if ret:
            corners *= scale
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners)
            objpoints.append(objp)

    return objpoints, imgpoints


def select_images(imgpoints, objpoints, points_to_use: int):
    """Select a subset of images based on their location in the image plane.

    This function selects a subset of images based on the location of the corners in the image plane. The images are sorted
    by their distance to the origin and a specified number of images is selected to use.

    Args:
        imgpoints (list): A list of 2D points in the image plane.
        objpoints (list): A list of 3D points in real world space.
        points_to_use (int): The number of images to use. Must be greater than 0.

    Returns:
        tuple: A tuple containing two lists, `objpoints` and `imgpoints`. `objpoints` is a list of 3D points in real world space, and `imgpoints` is a list of 2D points in the image plane.

    Raises:
        ValueError: If `points_to_use` is less than or equal to 0.
    """
    if points_to_use <= 0:
        raise ValueError("The number of points to use must be greater than 0.")

    if len(imgpoints) <= points_to_use:
        return imgpoints, objpoints

    X = np.asarray([np.ravel(x) for x in imgpoints])
    pca = PCA(n_components=1)
    Xt = np.ravel(pca.fit_transform(X))

    # sort images by their distance to the origin
    idxs = np.argsort(Xt)
    objpoints = [objpoints[i] for i in idxs]
    imgpoints = [imgpoints[i] for i in idxs]

    # select points to use
    x_range = np.linspace(
        0, len(imgpoints) - 1, points_to_use, endpoint=False, dtype=int
    )
    objpoints = [objpoints[i] for i in x_range]
    imgpoints = [imgpoints[i] for i in x_range]

    return imgpoints, objpoints


def calibrate_camera_zhang(objpoints, imgpoints, dim):
    """Compute camera matrix and distortion coefficients using Zhang's method.

    Args:
        objpoints (list): A list of 3D points in real world space.
        imgpoints (list): A list of 2D points in the image plane.
        dim (tuple): The image dimensions.

    Returns:
        tuple: A tuple containing the camera matrix, distortion coefficients, rotation vectors, and translation vectors.
    """
    ret, K, D, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, dim, None, None)
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(K, D, dim, 1, dim)
    mapx, mapy = cv2.initUndistortRectifyMap(K, D, None, newcameramtx, dim, 5)
    return K, D, mapx, mapy


def calibrate_camera_fisheye(objpoints, imgpoints, dim, balance=0.5):
    """Compute camera matrix and distortion coefficients using fisheye method.

    Args:
        objpoints (list): A list of 3D points in real world space.
        imgpoints (list): A list of 2D points in the image plane.
        dim (tuple): The image dimensions.
        balance (float): The balance factor. Must be between 0 and 1. Larger values wil

    Returns:
        tuple: A tuple containing the camera matrix, distortion coefficients, rotation vectors, and translation vectors.
    """
    N_OK = len(objpoints)

    K = np.zeros((3, 3))
    D = np.zeros((4, 1))
    rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
    tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
    objpoints = np.expand_dims(np.asarray(objpoints), -2)

    ret, K, D, rvecs, tvecs = cv2.fisheye.calibrate(
        objpoints,
        imgpoints,
        dim,
        K,
        D,
        rvecs,
        tvecs,
        cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + cv2.fisheye.CALIB_FIX_SKEW,
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6),
    )
    new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        K, D, dim, np.eye(3), balance=balance
    )
    mapx, mapy = cv2.fisheye.initUndistortRectifyMap(
        K, D, np.eye(3), new_K, dim, cv2.CV_32FC1
    )
    return K, D, mapx, mapy


def find_intrinsic_camera_parameters(
    media_path: _pathlike,
    fps: int = 1,
    scale: int = 4,
    save_path: Optional[_pathlike] = None,
    draw_on_save: bool = False,
    points_to_use: int = 50,
    calibration_method: str = "zhang",
    return_mappings: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Calculate the intrinsic parameters of a camera from a video of a checkerboard pattern.

    This function takes a video file containing a checkerboard pattern and calculates the intrinsic parameters of the camera. The video is first processed to locate the corners of the checkerboard in each frame. These corners are then used to compute the intrinsic parameters of the camera.

    Args:
        media_path (Union[str, Path]): Path to the video file or a list of video files containing the checkerboard pattern. Wildcards are supported.
        fps (int, optional): Frames per second to use when processing the video. Defaults to 1.
        scale (int, optional): Scale factor to use when processing the video. Defaults to 4.
        save_path (Optional[Union[str, Path]], optional): Path to save the computed intrinsic parameters. If not specified, the parameters are not saved. Defaults to None.
        draw_on_save (bool, optional): If `True`, the corners of the checkerboard are drawn on the frames and saved with the intrinsic parameters. Defaults to False.
        points_to_use (int, optional): Number of frames to use when calculating the intrinsic parameters. If more frames are found than this number, a subset of frames is selected based on their location in the image plane. Defaults to 50.
        calibration_method (str, optional): Calibration method to use. Must be either "zhang" or "fisheye". Defaults to "zhang".
        return_mappings (bool, optional): If `True`, the function returns the computed mapping functions along with the intrinsic parameters. Defaults to True.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: A tuple containing the camera matrix, distortion coefficients, and mapping functions (if `return_mappings` is True).

    Raises:
        ValueError: If the `calibration_method` is not "zhang" or "fisheye".
    """

    # Support multiple video files
    camera = Camera(media_path)

    # Find corners in each video
    objpoints, imgpoints = detect_corners(camera, scale, fps)
    if len(imgpoints) == 0:
        logger.error("No checkerboards found.")
    logger.info(f"imgpoints found: {len(imgpoints)}")

    # Select frames to use based on PCA Variance
    imgpoints, objpoints = select_images(imgpoints, objpoints, points_to_use)
    logger.debug(f"imgpoints used: {len(imgpoints)}")

    if 1 <= points_to_use <= len(imgpoints):
        logger.info(
            f"Too many ({len(imgpoints)}) checkerboards found. Selecting {points_to_use}."
        )

    logger.info("Computing calibration parameters...")

    dim = camera.frame_width, camera.frame_height

    # Compute camera matrix and distortion coefficients
    if calibration_method.lower() == "zhang":
        logger.info("Using Zhang's method.")
        K, D, mapx, mapy = calibrate_camera_zhang(objpoints, imgpoints, dim)
    elif calibration_method.lower() == "fisheye":
        K, D, mapx, mapy = calibrate_camera_fisheye(objpoints, imgpoints, dim)
    else:
        raise ValueError("Calibration method must be `zhang` or `fisheye`.")

    logger.info("Finished computing calibration parameters.")

    return K, D, mapx, mapy


def calibrate_video_from_mappings(
    media_path: _pathlike,
    mapx: NDArray,
    mapy: NDArray,
    save_path: _pathlike,
    stabilize: bool = True,
):
    """
    Calibrates a video using provided mapping parameters.

    Args:
    media_path (str): The path to the input video file.
    mapx (NDArray): The mapping array for x-axis.
    mapy (NDArray): The mapping array for y-axis.
    save_path (str): The path to save the calibrated video.
    stabilize (bool, optional): Whether to stabilize the video or not. Default is True.

    Returns:
    None
    """

    def generator():
        stab = Stabilizer()
        camera = Camera(media_path)
        for frame in camera:
            stab_frame = stab.stabilize(frame)

            if stab_frame is not None and stabilize:
                frame = stab_frame

            frame = cv2.remap(
                frame,
                mapx,
                mapy,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
            )
            yield frame

    make_video(generator(), save_path)
