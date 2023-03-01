'''
Sample Usage:-
python pose_estimation.py --K_Matrix calibration_matrix.npy --D_Coeff distortion_coefficients.npy --type DICT_5X5_100
'''

import numpy as np
import cv2
import sys
from utils import ARUCO_DICT
import argparse
import time
from autolab_core import RigidTransform
from dvrk.vision.ZedImageCapture import ZedImageCapture

def rvec_tvec_to_transform(rvec, tvec):
    '''
    convert translation and rotation to pose
    '''
    if rvec is None or tvec is None:
        return None

    print(rvec.shape)
    R = cv2.Rodrigues(rvec)[0]
    t = tvec
    return RigidTransform(R, t, from_frame='tag', to_frame='world')


def pose_estimation(frame, aruco_dict_type, matrix_coefficients, distortion_coefficients):
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
    # corners, ids, rejected_img_points = cv2.aruco.detectMarkers(gray, cv2.aruco_dict, parameters=parameters,
    #                                                             cameraMatrix=matrix_coefficients,
    #                                                             distCoeff=distortion_coefficients)
    # If markers are detected
    rvec, tvec = None, None
    if len(corners) > 0:
        for i in range(0, len(ids)):
            # Estimate pose of each marker and return the values rvec and tvec---(different from those of camera coefficients)
            rvec, tvec, markerPoints = cv2.aruco.estimatePoseSingleMarkers(corners[i], 0.07064375, matrix_coefficients,
                                                                           distortion_coefficients)  # 0.158
            # Draw a square around the markers
            cv2.aruco.drawDetectedMarkers(frame, corners)

            # Draw Axis
            # cv2.aruco.drawAxis(frame, matrix_coefficients, distortion_coefficients, rvec, tvec, 0.01)
            # cv2.drawAxis(fr+ame, matrix_coefficients, distortion_coefficients, rvec, tvec, 0.01)
            cv2.drawFrameAxes(frame, matrix_coefficients, distortion_coefficients, rvec, tvec, 0.01)

    return frame, rvec, tvec


if __name__ == '__main__':

    ap = argparse.ArgumentParser()
    ap.add_argument("-k", "--K_Matrix", required=False, help="Path to calibration matrix (numpy file)")
    ap.add_argument("-d", "--D_Coeff", required=False, help="Path to distortion coefficients (numpy file)")
    ap.add_argument("-t", "--type", type=str, default="DICT_ARUCO_ORIGINAL", help="Type of ArUCo tag to detect")
    args = vars(ap.parse_args())

    if ARUCO_DICT.get(args["type"], None) is None:
        print(f"ArUCo tag type '{args['type']}' is not supported")
        sys.exit(0)

    aruco_dict_type = ARUCO_DICT[args["type"]]
    calibration_matrix_path = args["K_Matrix"]
    distortion_coefficients_path = args["D_Coeff"]

    k = np.array(
        [[1376.21533203125, 0.0, 1113.4146728515625], [0.0, 1376.21533203125, 612.0199584960938], [0.0, 0.0, 1.0]])
    d = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    # k = np.load(calibration_matrix_path)
    # d = np.load(distortion_coefficients_path)

    # video = cv2.VideoCapture(0)
    # time.sleep(2.0)

    zed = ZedImageCapture(exposure=80, resolution="2K", gain=20,
                          whitebalance_temp=4500, brightness=0)

    while True:
        # ret, frame = video.read()
        img_left, _ = zed.capture_image()

        # if not ret:
        #     break

        output, rvec, tvec = pose_estimation(img_left, aruco_dict_type, k, d)
        pose = rvec_tvec_to_transform(rvec, tvec)
        print(pose)

        cv2.imshow('Estimated Pose', output)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    zed.close()
    # video.release()
    cv2.destroyAllWindows()
