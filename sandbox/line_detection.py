import cv2
import skvideo.io
from tqdm import tqdm
import numpy as np
import os
import matplotlib.pyplot as plt

import sys

# sys.path.append('../../SoccerTrack')
from soccertrack.utils.utils import make_video


def detect_corners(frame):
    # Apply blur to reduce noise
    k = 5
    blur = cv2.GaussianBlur(frame, (k, k), 0)

    # find the keypoints
    fast = cv2.FastFeatureDetector_create(threshold=10000)
    kp = fast.detect(blur, None)
    pts = cv2.KeyPoint_convert(kp)

    # draw keypoints
    dst = np.zeros_like(frame)
    dst = cv2.drawKeypoints(dst, kp, None, color=(255, 255, 255))
    cv2.imwrite("../x_ignore/corners.png", dst)

    return pts


def detect_lines(image, length_threshold=50, distance_threshold=50, thickness=15, k=15):
    # make sure that the image is in grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # threshold image so values less than 127 are set to 0
    gray = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)[1]

    # change image type for line detection
    gray = gray.astype(np.uint8)

    # dilate image to fill in holes
    kernel = np.ones((k, k), np.uint8)
    gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)

    # Create default Fast Line Detector class
    fld = cv2.ximgproc.createFastLineDetector(
        length_threshold=length_threshold,
        distance_threshold=distance_threshold,
    )
    # Get line vectors from the image
    lines = fld.detect(gray)

    # Draw lines on the image
    line_on_image = fld.drawSegments(
        np.zeros_like(gray), lines, linethickness=thickness
    )

    # image to black and white
    line_on_image = cv2.cvtColor(line_on_image, cv2.COLOR_BGR2GRAY)

    # otsu thresholding
    binary = cv2.threshold(line_on_image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[
        1
    ]

    # binary = cv2.dilate(binary, None, iterations=0)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    return binary


def FLD(
    image,
    length_threshold=50,
    distance_threshold=50,
    canny_th1=20.0,
    canny_th2=50.0,
    canny_aperture_size=5,
    do_merge=False,
):

    binary = detect_lines(
        image, length_threshold=150, distance_threshold=5, thickness=3, k=25
    )

    # save image
    # cv2.imwrite("../x_ignore/binary.png", binary)

    # find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # make mask with largest contour
    max_cnt = max(contours, key=cv2.contourArea)
    # epsilon = 0.01*cv2.arcLength(max_cnt,True)
    # approx = cv2.approxPolyDP(max_cnt,epsilon,True)

    mask = np.zeros_like(
        image
    )  # Create mask where white is what we want, black otherwise
    # x, y, w, h = cv2.boundingRect(approx)
    rect = cv2.minAreaRect(max_cnt)
    # rect = cv2.minAreaRect(approx)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    hull = cv2.convexHull(max_cnt)
    return hull

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # output = cv2.bitwise_and(image, mask)
    output = cv2.drawContours(image, [hull], 0, (255, 0, 0), 10)
    cv2.imwrite("../x_ignore/output.png", output)
    return output

    binary = detect_lines(
        output, length_threshold=80, distance_threshold=1, thickness=25, k=25
    )
    contours, hierarchy = cv2.findContours(
        binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )

    # save image

    image = np.zeros_like(image)

    red = [0, 0, 255]
    blue = [255, 0, 0]
    green = [0, 255, 0]
    for i in range(len(contours)):
        threshold = 5000
        # if contour area is larger than threshold
        cnt = contours[i]
        if cv2.contourArea(cnt) > threshold:
            epsilon = 0.02 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            color = np.random.randint(0, 255, (3)).tolist()
            cv2.drawContours(image, [approx], 0, color, -1, cv2.LINE_8, hierarchy, 0)

    out[mask == (255, 255, 255)] = image[mask == (255, 255, 255)]

    cv2.imwrite("../x_ignore/line-test.png", out)
    return out


if __name__ == "__main__":
    output = "../x_ignore/lines.mp4"
    videogen = skvideo.io.vreader("../x_ignore/drone_video.mp4")
    
    results = []
    for i, frame in enumerate(tqdm(videogen)):
        results.append(FLD(frame))
    
    # save results to file
    import pickle

    with open('hulls', 'wb') as fp:
        pickle.dump(results, fp)


    # max_frames = 300
    # results = (
    #     FLD(
    #         frame,
    #     )
    #     for i, frame in enumerate(tqdm(videogen))
    #     if i < max_frames
    # )

    make_video(results, output)
