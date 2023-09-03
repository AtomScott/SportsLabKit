import cv2
import numpy as np

from .base import BaseCalibrationModel


class LineBasedCalibrator(BaseCalibrationModel):
    def __init__(self, min_line_length=50, line_distance_threshold=50, line_thickness=15, morph_size=15, dst_points=None):
        """Initialize the line-based calibrator with given parameters."""
        self.min_line_length = min_line_length
        self.line_distance_threshold = line_distance_threshold
        self.line_thickness = line_thickness
        self.morph_size = morph_size
        # If destination points are not provided, default to a standard soccer pitch
        if dst_points is None:
            # Using the dimensions of a standard soccer pitch (105m x 68m)
            self.dst_points = np.array([
                [0, 0],
                [105, 0],
                [105, 68],
                [0, 68]
            ])

    def _preprocess_image(self, image):
        """Convert the image to grayscale and apply thresholding and morphological operations."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)[1]
        gray = gray.astype(np.uint8)
        kernel = np.ones((self.morph_size, self.morph_size), np.uint8)
        gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
        return gray

    def _get_largest_contour(self, image):
        """Extract and return the largest contour from the binary image."""
        binary = self._preprocess_image(image)
        contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        max_contour = max(contours, key=cv2.contourArea)
        hull = cv2.convexHull(max_contour)
        return hull

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

    def _approximate_contour(self, hull):
        """Approximate a convex hull to a quadrilateral by considering most distant points."""
        first_point = hull[0][0]
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
        angles = np.arctan2(points[:,1] - centroid[1], points[:,0] - centroid[0])
        ordered_points = points[np.argsort(angles)]
        return ordered_points

    def _calculate_homography(self, src_points, dst_points):
        """Compute the transformation matrix between source and destination points."""
        ordered_src = self._arrange_points_clockwise(src_points)
        ordered_dst = self._arrange_points_clockwise(dst_points)
        H, _ = cv2.findHomography(ordered_src, ordered_dst, method=0)
        return H

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
        quadrilateral = self._approximate_contour(contour)

        homography_matrix = self._calculate_homography(quadrilateral, self.dst_points)
        return homography_matrix
