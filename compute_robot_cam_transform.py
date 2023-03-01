import numpy as np
from autolab_core import RigidTransform
from dvrk.vision.ZedImageCapture import ZedImageCapture
from dvrk.motion.dvrkArm import dvrkArm

import cv2

tag2robot_translation = np.array([-0.00527915,  0.02133142, -0.14538199])
# tag2robot_rotation = np.array([0.33442367, 0.05139208, -0.51412141, 0.78816168])

tag2cam_translation = np.array([ 0.07225525, -0.03498547,  0.29815686])
# tag2cam_rotation = np.array([0.05093597, -0.03672976, 0.97591301, -0.2089264])
tag2cam_rotation_mat = np.array([[-0.88309508,  0.4541019,  0.11804471],
 [0.36855112,  0.82705711, -0.42443681],
 [-0.29036728, -0.33131255, -0.89772983]])

pointA = tag2robot_translation
pointB = np.array([0.05046485,  0.0611812, -0.14239674])

tag_x_axis_robot_frame = (pointB - pointA) / np.linalg.norm(pointB - pointA)
tag_rot_angle_robot_frame = np.arctan2(tag_x_axis_robot_frame[1], tag_x_axis_robot_frame[0])
print('rotation angle:', tag_rot_angle_robot_frame)
R = RigidTransform.z_axis_rotation(tag_rot_angle_robot_frame)
#
# current_end_effector_pos = np.array([0.10463757, 0.08512785, -0.08360346])
# current_end_effector_orient = np.array([0.30559351, 0.21019553, -0.90507192, 0.20802708])

# correction_in_robot_frame = np.array([0, 0, 0])
correction_in_robot_frame = np.array([0.07064375 / 2, -0.07064375 / 2, 0])


def get_tag_to_robot_tf():

    # R = np.eye(3)
    return RigidTransform(R, tag2robot_translation + R@correction_in_robot_frame, from_frame='tag', to_frame='robot')


def get_tag_to_cam_tf():
    return RigidTransform(tag2cam_rotation_mat, tag2cam_translation, from_frame='tag', to_frame='camera')


def project_robot_pt_to_img(current_end_effector_pos, robot2cam_tf, k_mat):
    current_end_effector_pos = np.append(current_end_effector_pos, 1.0)
    current_end_effector_pos = np.expand_dims(current_end_effector_pos, axis=1)
    current_end_effector_pos_in_cam_frame = np.matmul(robot2cam_tf.matrix, current_end_effector_pos)
    end_effector_cam_point_hom = np.matmul(k_mat, current_end_effector_pos_in_cam_frame[:-1, :])
    end_effector_cam_point_x = int(end_effector_cam_point_hom[0] / end_effector_cam_point_hom[2] + 0.5)
    end_effector_cam_point_y = int(end_effector_cam_point_hom[1] / end_effector_cam_point_hom[2] + 0.5)
    return end_effector_cam_point_x, end_effector_cam_point_y


def main():
    tag2robot_tf = get_tag_to_robot_tf()
    print(tag2robot_tf)
    tag2cam_tf = get_tag_to_cam_tf()
    print(tag2cam_tf)
    robot2cam_tf = tag2cam_tf * tag2robot_tf.inverse()
    print(f"robot to camera tf: {robot2cam_tf}")
    print(f"camera to robot tf: {robot2cam_tf.inverse()}")

    proj_mat = np.array(
        [[1376.21533203125, 0.0, 1113.4146728515625, 0.0], [0.0, 1376.21533203125, 612.0199584960938, 0.0],
         [0.0, 0.0, 1.0, 0.0]])
    k_mat = np.array(
        [[1376.21533203125, 0.0, 1113.4146728515625], [0.0, 1376.21533203125, 612.0199584960938], [0.0, 0.0, 1.0]])

    psm = dvrkArm("/PSM1")
    zed = ZedImageCapture(exposure=80, resolution="2K", gain=20,
                          whitebalance_temp=4500, brightness=0)

    point_1 = tag2robot_translation
    point_2 = tag2robot_translation + R@np.array([0.07064375, 0.0, 0.0])
    point_3 = tag2robot_translation + R@np.array([0.0, -0.07064375, 0.0])
    point_4 = tag2robot_translation + R@np.array([0.07064375, -0.07064375, 0.0])
    point_1_x, point_1_y = project_robot_pt_to_img(point_1, robot2cam_tf, k_mat)
    point_2_x, point_2_y = project_robot_pt_to_img(point_2, robot2cam_tf, k_mat)
    point_3_x, point_3_y = project_robot_pt_to_img(point_3, robot2cam_tf, k_mat)
    point_4_x, point_4_y = project_robot_pt_to_img(point_4, robot2cam_tf, k_mat)
    print(f"{point_4}")
    while True:
        # current_end_effector_pos = np.array([0.00533202, 0.07247908, -0.14223945])
        # current_end_effector_pos = current_end_effector_pos + correction_in_robot_frame * 2.0
        current_end_effector_pos, _ = psm.get_current_pose()
        end_effector_cam_point_x, end_effector_cam_point_y = project_robot_pt_to_img(current_end_effector_pos,
                                                                                     robot2cam_tf, k_mat)

        img_left, _ = zed.capture_image()
        img_left = cv2.circle(img_left, (end_effector_cam_point_x, end_effector_cam_point_y), 5, (0, 0, 255), -1)
        img_left = cv2.circle(img_left, (point_1_x, point_1_y), 5, (0, 255, 0), -1)
        img_left = cv2.circle(img_left, (point_2_x, point_2_y), 5, (0, 255, 0), -1)
        img_left = cv2.circle(img_left, (point_3_x, point_3_y), 5, (0, 255, 0), -1)
        img_left = cv2.circle(img_left, (point_4_x, point_4_y), 5, (0, 255, 0), -1)

        cv2.imshow("img", img_left)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    zed.close()


if __name__ == '__main__':
    main()
    # psm = dvrkArm("/PSM1")
    # p, q = psm.get_current_pose()
    # p_new = p + np.array([0.0, 0.0, 0.04])
    # psm.set_pose(p_new, q)
    # p_new = p_new + np.array([-0.07064375, 0.0, 0.0])
    # # p_new = p_new + np.array([0.07064375, -0.07064375, 0.0])
    # # p_new = np.array([0.00289148, 0.07474244, -0.14473077])
    # # q = np.array([0.33442367, 0.05139208, -0.51412141, 0.78816168])
    # psm.set_pose(p_new, q)
    #
    # p_new = p_new - np.array([0.0, 0.0, 0.04])
    # psm.set_pose(p_new, q)
