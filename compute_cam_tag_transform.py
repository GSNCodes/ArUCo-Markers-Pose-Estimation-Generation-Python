'''
Sample Usage:-
python pose_estimation.py --K_Matrix calibration_matrix.npy --D_Coeff distortion_coefficients.npy --type DICT_5X5_100
'''
from typing import Tuple
import numpy as np
import cv2
import sys
from utils import ARUCO_DICT
import argparse
import time
from autolab_core import RigidTransform
from dvrk.vision.ZedImageCapture import ZedImageCapture

# from zed camera for 720p rectified images
CAMERA_INTRINSICS = np.array(
    [[1376.21533203125, 0.0, 1113.4146728515625], [0.0, 1376.21533203125, 612.0199584960938], [0.0, 0.0, 1.0]])
DISTORTION_COEFFS = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

TAG_SIZE = 0.07064375  # edge size in m


def rvec_tvec_to_transform(rvec, tvec):
    '''
    convert translation and rotation to pose
    '''
    if rvec is None or tvec is None:
        return None

    R = cv2.Rodrigues(rvec)[0]
    t = tvec
    return RigidTransform(R, t, from_frame='tag', to_frame='camera')


def pose_estimation(frame, aruco_dict_type, matrix_coefficients, distortion_coefficients, tag_size):
    '''
    frame - Frame from the video stream
    matrix_coefficients - Intrinsic matrix of the calibrated camera
    distortion_coefficients - Distortion coefficients associated with your camera

    return:-
    frame - The frame with the axis drawn on it
    '''

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.aruco_dict = cv2.aruco.Dictionary_get(aruco_dict_type)
    parameters = cv2.aruco.DetectorParameters_create()

    corners, ids, rejected_img_points = cv2.aruco.detectMarkers(gray, cv2.aruco_dict, parameters=parameters)

    rvec, tvec = None, None
    if len(corners) > 0:
        for i in range(0, len(ids)):
            # Estimate pose of each marker and return the values rvec and tvec---(different from those of camera coefficients)
            rvec, tvec, markerPoints = cv2.aruco.estimatePoseSingleMarkers(corners[i], tag_size, matrix_coefficients,
                                                                           distortion_coefficients)  # 0.158
            # Draw a square around the markers
            cv2.aruco.drawDetectedMarkers(frame, corners)

            # Draw Axis
            # cv2.aruco.drawAxis(frame, matrix_coefficients, distortion_coefficients, rvec, tvec, 0.01)
            # cv2.drawAxis(fr+ame, matrix_coefficients, distortion_coefficients, rvec, tvec, 0.01)
            cv2.drawFrameAxes(frame, matrix_coefficients, distortion_coefficients, rvec, tvec, 0.01)

    return frame, rvec, tvec


def get_tag_to_camera_tf(image: np.ndarray, camera_intrinsics: np.ndarray, distortion_coefficients: np.ndarray,
                         tag_size: float) -> Tuple[RigidTransform, np.ndarray]:
    aruco_dict_type = ARUCO_DICT["DICT_ARUCO_ORIGINAL"]
    output, rvec, tvec = pose_estimation(image, aruco_dict_type, camera_intrinsics, distortion_coefficients, tag_size)
    pose = rvec_tvec_to_transform(rvec, tvec)

    return pose, output


def main():
    zed = ZedImageCapture(exposure=80, resolution="2K", gain=20,
                          whitebalance_temp=4500, brightness=0)
    while True:
        # ret, frame = video.read()
        img_left, _ = zed.capture_image()
        _, output = get_tag_to_camera_tf(img_left, CAMERA_INTRINSICS, DISTORTION_COEFFS, TAG_SIZE)

        cv2.imshow('Estimated Pose. Press q to exit', output)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    zed.close()


if __name__ == '__main__':
    main()
