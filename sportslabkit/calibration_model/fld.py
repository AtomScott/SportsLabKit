import cv2
import numpy as np

from .base import BaseCalibrationModel


class SimpleContourCalibrator(BaseCalibrationModel):
    def __init__(
        self,
        morph_open_size=15,
        morph_close_size=15,
        morph_dilate_size=15,
        morph_erode_size=15,
        morph_iters=1,
        dst_points=None
    ):
        """Initialize the line-based calibrator with given parameters."""
        self.morph_open_size = morph_open_size
        self.morph_close_size = morph_close_size
        self.morph_dilate_size = morph_dilate_size
        self.morph_erode_size = morph_erode_size
        self.morph_iters = morph_iters

        # If destination points are not provided, default to a standard soccer pitch
        if dst_points is None:
            # Using the dimensions of a standard soccer pitch (105m x 68m)
            self.dst_points = np.array([[0, 0], [105, 0], [105, 68], [0, 68]])

    def _preprocess_image(self, image):
        """Convert the image to grayscale and apply thresholding and morphological operations."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)[1]
        gray = gray.astype(np.uint8)
        open_kernel = np.ones((self.morph_open_size, self.morph_open_size), np.uint8)
        close_kernel = np.ones((self.morph_close_size, self.morph_close_size), np.uint8)
        dilate_kernel = np.ones((self.morph_dilate_size, self.morph_dilate_size), np.uint8)
        erode_kernel = np.ones((self.morph_erode_size, self.morph_erode_size), np.uint8)
        for i in range(self.morph_iters):
            gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, close_kernel)
            gray = cv2.morphologyEx(gray, cv2.MORPH_OPEN, open_kernel)
            gray = cv2.morphologyEx(gray, cv2.MORPH_DILATE, dilate_kernel)
            gray = cv2.morphologyEx(gray, cv2.MORPH_ERODE, erode_kernel)
        return gray

    def _get_largest_contour(self, image):
        """Extract and return the largest contour from the binary image."""
        binary = self._preprocess_image(image)
        contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        max_contour = max(contours, key=cv2.contourArea)
        return max_contour

    def _approximate_hull(self, contour):
        hull = cv2.convexHull(contour)
        return hull

    def _get_upper_left_courner(self, hull):
        """Find the nearest point from the upper left corner."""
        sorted_hull = sorted(hull, key=lambda x:x[0][0]*x[0][0] + x[0][1]*x[0][1])
        return sorted_hull[0][0]

    def _farthest_point_from(self, reference_point, point_list):
        """Find the point in 'point_list' that is farthest from 'reference_point'."""
        max_dist = 0
        farthest_point = None
        for point in point_list:
            dist = cv2.norm(reference_point - point[0])
            if dist > max_dist:
                max_dist = dist
                farthest_point = point[0]
        return farthest_point

    def _approximate_quad(self, hull):
        """Approximate a convex hull to a quadrilateral by considering most distant points."""
        first_point = self._get_upper_left_corner(hull)
        second_point = self._farthest_point_from(first_point, hull)

        max_distance = 0
        third_point = None
        for pt in np.array(hull, dtype=np.float32):
            dist = cv2.pointPolygonTest(np.array([first_point, second_point], dtype=np.float32), pt[0], True)
            if abs(dist) > max_distance:
                max_distance = abs(dist)
                third_point = pt[0]

        fourth_point = self._farthest_point_from(third_point, hull)
        quadrilateral = np.array([first_point, second_point, third_point, fourth_point])
        return quadrilateral

    def _arrange_points_clockwise(self, points):
        """Arrange the given points in clockwise order starting from top-left."""
        centroid = np.mean(points, axis=0)
        angles = np.arctan2(points[:, 1] - centroid[1], points[:, 0] - centroid[0])
        ordered_points = points[np.argsort(angles)]
        return ordered_points

    def _calculate_homography(self, src_points, dst_points):
        """Compute the transformation matrix between source and destination points."""
        ordered_src = self._arrange_points_clockwise(src_points)
        ordered_dst = self._arrange_points_clockwise(dst_points)
        H, _ = cv2.findHomography(ordered_src, ordered_dst, method=0)
        return H

    def find_quadrilateral(self, image):
        """Find the quadrilateral in the given image.

        Parameters:
        - image: numpy array
            The source image.

        Returns:
        - numpy array
            The quadrilateral in the image.
        """

        contour = self._get_largest_contour(image)
        hull = self._approximate_hull(contour)
        quadrilateral = self._approximate_quad(hull)
        return self.order_points(quadrilateral)

    def order_points(self, pts):
        """Order the points in clockwise order starting from top-left."""
        centroid = np.mean(pts, axis=0)

        # Compute the angles relative to the centroid
        angles = np.arctan2(pts[:, 1] - centroid[1], pts[:, 0] - centroid[0])

        # Sort the points based on the angles
        ordered_pts = pts[np.argsort(angles)]
        return ordered_pts

    def forward(self, image):
        """Calculate the homography matrix for the given image.

        Parameters:
        - image: numpy array
            The source image.
        - dst_points: numpy array or None
            The destination points for the transformation. If not provided,
            it defaults to the four corners of a standard soccer pitch (105m x 68m).

        Returns:
        - numpy array
            The computed homography matrix.
        """

        contour = self._get_largest_contour(image)
        quadrilateral = self._approximate_quad(contour)

        homography_matrix = self._calculate_homography(quadrilateral, self.dst_points)
        return homography_matrix


class FLDCalibrator(SimpleContourCalibrator):
    def __init__(self, length_threshold=50, distance_threshold=50, canny_th1=50, canny_th2=150, canny_aperture_size=3, do_merge=True, dst_points=None):
        """Initialize the line-based calibrator with given parameters."""
        self.fld = cv2.ximgproc.createFastLineDetector(_length_threshold=self.length_threshold,
                                                       _distance_threshold=self.distance_threshold,
                                                       _canny_th1=self.canny_th1, _canny_th2=self.canny_th2,
                                                       _canny_aperture_size=self.canny_aperture_size, _do_merge=self.do_merge)
        if dst_points is None:
            # Using the dimensions of a standard soccer pitch (105m x 68m)
            self.dst_points = np.array([[0, 0], [105, 0], [105, 68], [0, 68]], dtype=np.float32)

    def _preprocess_image(self, image):
        """Convert the image to grayscale and apply thresholding and morphological operations."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)[1]
        gray = gray.astype(np.uint8)
        kernel = np.ones((self.morph_size, self.morph_size), np.uint8)
        gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
        return gray

    def _get_lines(self, image):
        """Detect lines in the image using Fast Line Detector."""

        lines = self.fld.detect(image)
        return lines

    def _get_largest_contour(self, image):
        """Extract and return the largest contour from the binary image."""
        binary = self._preprocess_image(image)
        lines = self._get_lines(binary)

        # Creating an empty canvas to draw lines on
        line_image = np.zeros_like(binary)
        for line in lines:
            x0, y0, x1, y1 = map(int, line[0])
            cv2.line(line_image, (x0, y0), (x1, y1), 255, self.line_thickness)

        contours, _ = cv2.findContours(line_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        max_contour = max(contours, key=cv2.contourArea)
        hull = cv2.convexHull(max_contour)
        return hull
