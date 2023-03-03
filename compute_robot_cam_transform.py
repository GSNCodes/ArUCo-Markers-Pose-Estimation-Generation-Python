import os
import yaml
from datetime import datetime

import numpy as np
import cv2

from autolab_core import RigidTransform
from dvrk.vision.ZedImageCapture import ZedImageCapture
from dvrk.motion.dvrkArm import dvrkArm
from compute_cam_tag_transform import get_tag_to_camera_tf

# from zed camera for 720p rectified images
CAMERA_INTRINSICS = np.array(
    [[1376.21533203125, 0.0, 1113.4146728515625], [0.0, 1376.21533203125, 612.0199584960938], [0.0, 0.0, 1.0]])
DISTORTION_COEFFS = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

TAG_SIZE = 0.07064375  # edge size in m
# Corrects for the fact that the robot touches the corners of the tag and not the center
TAG_ANGLE_OFFSET_ROBOT = np.array([TAG_SIZE / 2, -1 * TAG_SIZE / 2, 0])

JAW_CLOSED_ANGLE = [-0.2]

SAVE_DIR = "/home/davinci/aruco_camera_calibration/calibration_files"
CALIB_FNAME = "davinci_psm1_to_zed_calibration"


def get_tag_to_robot_tf(pointA: np.ndarray, pointB: np.ndarray,
                        correction_in_robot_frame: np.ndarray) -> RigidTransform:
    tag2robot_translation = pointA
    tag_x_axis_robot_frame = (pointB - pointA) / np.linalg.norm(pointB - pointA)
    tag_rot_angle_robot_frame = np.arctan2(tag_x_axis_robot_frame[1], tag_x_axis_robot_frame[0])
    R = RigidTransform.z_axis_rotation(tag_rot_angle_robot_frame)
    return RigidTransform(R, tag2robot_translation + R @ correction_in_robot_frame, from_frame='tag', to_frame='robot')


def project_robot_pt_to_img(current_end_effector_pos, robot2cam_tf, k_mat):
    current_end_effector_pos = np.append(current_end_effector_pos, 1.0)
    current_end_effector_pos = np.expand_dims(current_end_effector_pos, axis=1)
    current_end_effector_pos_in_cam_frame = np.matmul(robot2cam_tf.matrix, current_end_effector_pos)
    end_effector_cam_point_hom = np.matmul(k_mat, current_end_effector_pos_in_cam_frame[:-1, :])
    end_effector_cam_point_x = int(end_effector_cam_point_hom[0] / end_effector_cam_point_hom[2] + 0.5)
    end_effector_cam_point_y = int(end_effector_cam_point_hom[1] / end_effector_cam_point_hom[2] + 0.5)
    return end_effector_cam_point_x, end_effector_cam_point_y


def main():
    psm = dvrkArm("/PSM1")
    zed = ZedImageCapture(exposure=80, resolution="2K", gain=20,
                          whitebalance_temp=4500, brightness=0)
    # get cam tag pose
    img_left, _ = zed.capture_image()
    tag2cam_tf, output = get_tag_to_camera_tf(img_left, CAMERA_INTRINSICS, DISTORTION_COEFFS, TAG_SIZE)
    cv2.imshow("tag detection in camera", output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    if tag2cam_tf is None:
        print("Tag 2 Camera transform not detected. Returning")
        return

    # get robot corner points
    psm.set_jaw(JAW_CLOSED_ANGLE)
    input("Move robot (PSM1) end-effector tip to corner (a) of the tag and press Enter")
    pointA, _ = psm.get_current_pose()
    input("Move robot (PSM1) end-effector tip to corner (b) of the tag and press Enter")
    pointB, _ = psm.get_current_pose()

    # compute robot tag
    tag2robot_tf = get_tag_to_robot_tf(pointA, pointB, TAG_ANGLE_OFFSET_ROBOT)

    # compute robot cam
    robot2cam_tf = tag2cam_tf * tag2robot_tf.inverse()

    now = datetime.now()
    date_time_str = now.strftime("%Y-%m-%d_%H-%M-%S")
    fname = CALIB_FNAME + "_" + date_time_str + ".tf"
    fpath = os.path.join(SAVE_DIR, fname)
    robot2cam_tf.save(fpath)

    tag2robot_rot = tag2robot_tf.rotation
    point_1 = pointA
    point_2 = pointA + tag2robot_rot @ np.array([TAG_SIZE, 0.0, 0.0])
    point_3 = pointA + tag2robot_rot @ np.array([0.0, -1 * TAG_SIZE, 0.0])
    point_4 = pointA + tag2robot_rot @ np.array([TAG_SIZE, -1 * TAG_SIZE, 0.0])
    point_1_x, point_1_y = project_robot_pt_to_img(point_1, robot2cam_tf, CAMERA_INTRINSICS)
    point_2_x, point_2_y = project_robot_pt_to_img(point_2, robot2cam_tf, CAMERA_INTRINSICS)
    point_3_x, point_3_y = project_robot_pt_to_img(point_3, robot2cam_tf, CAMERA_INTRINSICS)
    point_4_x, point_4_y = project_robot_pt_to_img(point_4, robot2cam_tf, CAMERA_INTRINSICS)
    while True:
        current_end_effector_pos, _ = psm.get_current_pose()
        end_effector_cam_point_x, end_effector_cam_point_y = project_robot_pt_to_img(current_end_effector_pos,
                                                                                     robot2cam_tf, CAMERA_INTRINSICS)

        img_left, _ = zed.capture_image()
        img_left = cv2.circle(img_left, (end_effector_cam_point_x, end_effector_cam_point_y), 5, (0, 0, 255), -1)
        img_left = cv2.circle(img_left, (point_1_x, point_1_y), 5, (0, 255, 0), -1)
        img_left = cv2.circle(img_left, (point_2_x, point_2_y), 5, (0, 255, 0), -1)
        img_left = cv2.circle(img_left, (point_3_x, point_3_y), 5, (0, 255, 0), -1)
        img_left = cv2.circle(img_left, (point_4_x, point_4_y), 5, (0, 255, 0), -1)

        cv2.imshow("End Effector Tracking. Press q to exit.", img_left)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    zed.close()


if __name__ == '__main__':
    main()
