import os
import sys
# import json
import time
import copy
import pickle
import numpy as np
from typing import Any
import distutils.util as util

from types import SimpleNamespace as Namespace
# from src.ha_teacher import ha_teacher
import matplotlib.pyplot as plt


def plot_robot_trajectory(filepath: str) -> None:
    with open(filepath, 'rb') as f:
        phases = pickle.load(f)

    zero_ref = []
    step = []
    desired_linear_vel = []
    desired_com_height = []
    linear_vel = []
    rpy = []
    position = []
    timestamp = []
    angular_vel = []
    ground_reaction_forces = []
    student_ddq = []
    teacher_ddq = []
    total_ddq = []
    max_ddq = []
    min_ddq = []
    action_mode = []

    tracking_err = dict({
        'p': [],
        'rpy': [],
        'v': [],
        'rpy_dot': []
    })

    n = len(phases)
    # print(f"phases: {phases}")

    for i in range(len(phases)):
        tracking_error = list(phases[i]['tracking_error'])
        if np.isnan(tracking_error).any():
            tracking_err['p'].append([0, 0, 0])
            tracking_err['rpy'].append([0, 0, 0])
            tracking_err['v'].append([0, 0, 0])
            tracking_err['rpy_dot'].append([0, 0, 0])
        else:
            tracking_err['p'].append(tracking_error[0:3])
            tracking_err['rpy'].append(tracking_error[3:6])
            tracking_err['v'].append(tracking_error[6:9])
            tracking_err['rpy_dot'].append(tracking_error[9:12])

        zero_ref.append(0)
        step.append(i)

        student_ddq.append(phases[i]['student_ddq'])
        teacher_ddq.append(phases[i]['teacher_ddq'])
        # total_ddq.append(phases[i]['stance_ddq'][2])
        min_ddq.append(phases[i]['stance_ddq_limit'][0])
        max_ddq.append(phases[i]['stance_ddq_limit'][1])

        position.append(phases[i]['base_position'])
        desired_com_height.append(phases[i]['desired_com_height'])
        timestamp.append(phases[i]['timestamp'])
        desired_linear_vel.append(phases[i]['desired_speed'][0])
        linear_vel.append(phases[i]['base_linear_vel_in_body_frame'])
        angular_vel.append(phases[i]['base_angular_vel_in_body_frame'])
        rpy.append(phases[i]['base_rpy'])

        ground_reaction_forces.append(phases[i]['desired_ground_reaction_forces'])
        # action_mode.append(phases[i]['action_mode'])
        # print(content[i]['desired_linear_vel'])
        # print(type(content[i]['desired_linear_vel']))

    step = np.array(step)
    teacher_ddq = np.array(teacher_ddq)
    student_ddq = np.array(student_ddq)
    # total_ddq = np.array(total_ddq)
    max_ddq = np.array(max_ddq)
    min_ddq = np.array(min_ddq)

    tracking_err['p'] = np.asarray(tracking_err['p'])
    tracking_err['rpy'] = np.asarray(tracking_err['rpy'])
    tracking_err['v'] = np.asarray(tracking_err['v'])
    tracking_err['rpy_dot'] = np.asarray(tracking_err['rpy_dot'])
    timestamp = np.asarray(timestamp)
    position = np.asarray(position)
    zero_ref = np.asarray(zero_ref)
    linear_vel = np.asarray(linear_vel)
    desired_linear_vel = np.asarray(desired_linear_vel)
    desired_com_height = np.asarray(desired_com_height)
    angular_vel = np.asarray(angular_vel)
    rpy = np.asarray(rpy)

    # print(f"action_mode: {action_mode}")

    step = timestamp

    # Simplex input
    simplex_step = []
    non_simplex_step = []
    simplex_input = dict({
        'p': [],
        'rpy': [],
        'v': [],
        'rpy_dot': []
    })

    linear_vel_simplex = []
    rpy_simplex = []
    angular_vel_simplex = []
    position_simplex = []

    cnt = 0
    # for i in range(n):
    #     if action_mode[i] != ha_teacher.HACActionMode.SIMPLEX:
    #         linear_vel_simplex.append(linear_vel[i])
    #         rpy_simplex.append(rpy[i])
    #         angular_vel_simplex.append(angular_vel[i])
    #         position_simplex.append(position[i])
    #         non_simplex_step.append(step[i])
    #         cnt += 1
    #     else:
    #         simplex_input['p'].append(linear_vel[i])
    #         simplex_input['rpy'].append(rpy[i])
    #         simplex_input['v'].append(angular_vel[i])
    #         simplex_input['rpy_dot'].append(position[i])
    #         simplex_step.append(step[i])
    #
    # simplex_input['p'] = np.asarray(simplex_input['p'])
    # simplex_input['rpy'] = np.asarray(simplex_input['rpy'])
    # simplex_input['v'] = np.asarray(simplex_input['v'])
    # simplex_input['rpy_dot'] = np.asarray(simplex_input['rpy_dot'])
    #
    # linear_vel_simplex = np.asarray(linear_vel_simplex)
    # rpy_simplex = np.asarray(rpy_simplex)
    # angular_vel_simplex = np.asarray(angular_vel_simplex)
    # position_simplex = np.asarray(position_simplex)
    #
    # print(f"Simplex point: {n - cnt}")
    # print(f"Non-Simplex point: {cnt}")
    # # print(f"Tracking error: {tracking_err_plot['p']}")
    # # print(tracking_err['p'] == simplex_err['p'])
    # print(len(non_simplex_step))
    # print(len(simplex_step))
    #
    # print(f"step: {step}")
    # print(f"steps: {steps}")
    # print(desired_linear_vel.shape)
    # print(linear_vel.shape)
    # print(desired_linear_vel[:, 0])
    # print(tracking_err)
    # print("............")
    # print(tracking_err['v'][:, 0])
    fig, axes = plt.subplots(4, 3)
    # for i, step in enumerate(steps):
    #     print(i)
    # Vx
    axes[0, 0].plot(step, tracking_err['v'][:, 0], zorder=4, label='error_vx')
    # axes[0, 0].plot(simplex_step, simplex_input['v'][:, 0], zorder=3, color='red', label='simplex_vx')
    axes[0, 0].plot(step, desired_linear_vel[:, 0], zorder=2, label='desire_vx')
    axes[0, 0].plot(step, linear_vel[:, 0], zorder=1, label='vx')
    axes[0, 0].set_xlabel('Time/s')
    axes[0, 0].set_ylabel('Vx')
    axes[0, 0].legend()

    # Vy
    # axes[0, 1].plot(step, simplex_err['v'][:, 1], zorder=4, label='simplex_vx')
    axes[0, 1].plot(step, tracking_err['v'][:, 1], zorder=3, label='error_vy')
    axes[0, 1].plot(step, desired_linear_vel[:, 1], zorder=2, label='desire_vy')
    axes[0, 1].plot(step, linear_vel[:, 1], zorder=1, label='vy')
    axes[0, 1].set_xlabel('Time/s')
    axes[0, 1].set_ylabel('Vy')
    axes[0, 1].legend()

    # Vz
    axes[0, 2].plot(step, tracking_err['v'][:, 2], zorder=3, label='error_vz')
    axes[0, 2].plot(step, zero_ref, zorder=2, label='desire_vz')
    axes[0, 2].plot(step, linear_vel[:, 2], zorder=1, label='vz')
    axes[0, 2].set_xlabel('Time/s')
    axes[0, 2].set_ylabel('Vz')
    axes[0, 2].legend()

    # Roll
    axes[1, 0].plot(step, tracking_err['rpy'][:, 0], zorder=3, label='error_roll')
    axes[1, 0].plot(step, zero_ref, zorder=2, label='desire_roll')
    axes[1, 0].plot(step, rpy[:, 0], zorder=1, label='roll')
    axes[1, 0].set_xlabel('Time')
    axes[1, 0].set_ylabel('Roll')
    axes[1, 0].legend()

    # Pitch
    axes[1, 1].plot(step, tracking_err['rpy'][:, 1], zorder=3, label='error_pitch')
    axes[1, 1].plot(step, zero_ref, zorder=2, label='desire_pitch')
    axes[1, 1].plot(step, rpy[:, 1], zorder=1, label='pitch')
    axes[1, 1].set_xlabel('Time')
    axes[1, 1].set_ylabel('Pitch')
    axes[1, 1].legend()

    # Yaw
    axes[1, 2].plot(step, tracking_err['rpy'][:, 2], zorder=3, label='error_yaw')
    axes[1, 2].plot(step, desired_linear_vel[:, 2], zorder=2, label='desire_yaw')
    axes[1, 2].plot(step, rpy[:, 2], zorder=1, label='yaw')
    axes[1, 2].set_xlabel('Time/s')
    axes[1, 2].set_ylabel('Yaw')
    axes[1, 2].legend()

    # Wx
    axes[2, 0].plot(step, tracking_err['rpy_dot'][:, 0], zorder=3, label='error_wx')
    axes[2, 0].plot(step, desired_linear_vel[:, 2], zorder=2, label='desire_wx')
    axes[2, 0].plot(step, angular_vel[:, 0], zorder=1, label='wx')
    axes[2, 0].set_xlabel('Time/s')
    axes[2, 0].set_ylabel('Wx')
    axes[2, 0].legend()

    # Wy
    axes[2, 1].plot(step, tracking_err['rpy_dot'][:, 1], zorder=3, label='error_wy')
    axes[2, 1].plot(step, desired_linear_vel[:, 2], zorder=2, label='desire_wy')
    axes[2, 1].plot(step, angular_vel[:, 1], zorder=1, label='wy')
    axes[2, 1].set_xlabel('Time/s')
    axes[2, 1].set_ylabel('Wy')
    axes[2, 1].legend()

    # Wz
    axes[2, 2].plot(step, tracking_err['rpy_dot'][:, 2], zorder=3, label='error_wz')
    axes[2, 2].plot(step, desired_linear_vel[:, 2], zorder=2, label='desire_wz')
    axes[2, 2].plot(step, angular_vel[:, 2], zorder=1, label='wz')
    axes[2, 2].set_xlabel('Time/s')
    axes[2, 2].set_ylabel('Wz')
    axes[2, 2].legend()

    # Px
    axes[3, 0].plot(step, tracking_err['p'][:, 0], zorder=3, label='error_px')
    axes[3, 0].plot(step, zero_ref, zorder=2, label='desire_px')
    axes[3, 0].plot(step, zero_ref, zorder=1, label='px')
    axes[3, 0].set_xlabel('Time/s')
    axes[3, 0].set_ylabel('Px')
    axes[3, 0].legend()

    # Py
    axes[3, 1].plot(step, tracking_err['p'][:, 1], zorder=3, label='error_py')
    axes[3, 1].plot(step, zero_ref, zorder=2, label='desire_py')
    axes[3, 1].plot(step, zero_ref, zorder=1, label='py')
    axes[3, 1].set_xlabel('Time/s')
    axes[3, 1].set_ylabel('Py')
    axes[3, 1].legend()

    # Pz
    axes[3, 2].plot(step, tracking_err['p'][:, 2], zorder=3, label='error_pz')
    axes[3, 2].plot(step, desired_com_height, zorder=2, label='desire_pz')
    axes[3, 2].plot(step, position[:, 2], zorder=1, label='pz')
    axes[3, 2].set_xlabel('Time/s')
    axes[3, 2].set_ylabel('Pz')
    axes[3, 2].legend()

    fig2, axes2 = plt.subplots(2, 3)

    axes2[0, 0].plot(step, student_ddq[:, 0], zorder=4, label='hp_vx')
    axes2[0, 0].plot(step, teacher_ddq[:, 0], zorder=3, label='ha_vx')
    # axes2[0, 0].plot(step, total_ddq[:, 0], zorder=2, label='total_vx')
    axes2[0, 0].plot(step, min_ddq[:, 0], zorder=1, label='ddq_min')
    axes2[0, 0].plot(step, max_ddq[:, 0], zorder=1, label='ddq_max')
    axes2[0, 0].set_xlabel('Time/s')
    axes2[0, 0].set_ylabel('ddq vx')
    axes2[0, 0].legend()

    axes2[0, 1].plot(step, student_ddq[:, 1], zorder=4, label='hp_vy')
    axes2[0, 1].plot(step, teacher_ddq[:, 1], zorder=3, label='ha_vy')
    # axes2[0, 1].plot(step, total_ddq[:, 1], zorder=2, label='total_vy')
    axes2[0, 1].plot(step, min_ddq[:, 1], zorder=1, label='ddq_min')
    axes2[0, 1].plot(step, max_ddq[:, 1], zorder=1, label='ddq_max')
    axes2[0, 1].set_xlabel('Time/s')
    axes2[0, 1].set_ylabel('ddq vy')
    axes2[0, 1].legend()

    axes2[0, 2].plot(step, student_ddq[:, 2], zorder=3, label='hp_vz')
    axes2[0, 2].plot(step, teacher_ddq[:, 2], zorder=2, label='ha_vz')
    # axes2[0, 2].plot(step, total_ddq[:, 2], zorder=1, label='total_vz')
    axes2[0, 2].plot(step, min_ddq[:, 2], zorder=1, label='ddq_min')
    axes2[0, 2].plot(step, max_ddq[:, 2], zorder=1, label='ddq_max')
    axes2[0, 2].set_xlabel('Time/s')
    axes2[0, 2].set_ylabel('ddq vz')
    axes2[0, 2].legend()

    axes2[1, 0].plot(step, student_ddq[:, 3], zorder=3, label='hp_wx')
    axes2[1, 0].plot(step, teacher_ddq[:, 3], zorder=2, label='ha_wx')
    # axes2[1, 0].plot(step, total_ddq[:, 3], zorder=1, label='total_wx')
    axes2[1, 0].plot(step, min_ddq[:, 3], zorder=1, label='ddq_min')
    axes2[1, 0].plot(step, max_ddq[:, 3], zorder=1, label='ddq_max')
    axes2[1, 0].set_xlabel('Time/s')
    axes2[1, 0].set_ylabel('ddq wx')
    axes2[1, 0].legend()

    axes2[1, 1].plot(step, student_ddq[:, 4], zorder=3, label='hp_wy')
    axes2[1, 1].plot(step, teacher_ddq[:, 4], zorder=2, label='ha_wy')
    # axes2[1, 1].plot(step, total_ddq[:, 4], zorder=1, label='total_wy')
    axes2[1, 1].plot(step, min_ddq[:, 4], zorder=1, label='ddq_min')
    axes2[1, 1].plot(step, max_ddq[:, 4], zorder=1, label='ddq_max')
    axes2[1, 1].set_xlabel('Time/s')
    axes2[1, 1].set_ylabel('ddq wy')
    axes2[1, 1].legend()

    axes2[1, 2].plot(step, student_ddq[:, 5], zorder=3, label='hp_wz')
    axes2[1, 2].plot(step, teacher_ddq[:, 5], zorder=2, label='ha_wz')
    # axes2[1, 2].plot(step, total_ddq[:, 5], zorder=1, label='total_wz')
    axes2[1, 2].plot(step, min_ddq[:, 5], zorder=1, label='ddq_min')
    axes2[1, 2].plot(step, max_ddq[:, 5], zorder=1, label='ddq_max')
    axes2[1, 2].set_xlabel('Time/s')
    axes2[1, 2].set_ylabel('ddq wz')
    axes2[1, 2].legend()

    plt.tight_layout()
    plt.show()


def plot_trajectory2(trajectory_tensor, reference_trajectory_tensor=None):
    """
   trajectory_tensor: a numpy array [n, 4], where n is the length of the trajectory,
                       5 is the dimension of each point on the trajectory, containing [x, x_dot, theta, theta_dot]
   """
    trajectory_tensor = np.array(trajectory_tensor)
    reference_trajectory_tensor = np.array(
        reference_trajectory_tensor) if reference_trajectory_tensor is not None else None
    n, c = trajectory_tensor.shape

    y_label_list = ["x", "x_dot", "theta", "theta_dot"]

    plt.figure(figsize=(9, 6))

    for i in range(c):

        plt.subplot(c, 1, i + 1)
        plt.plot(np.arange(n), trajectory_tensor[:, i], label=y_label_list[i])

        if reference_trajectory_tensor is not None:
            plt.plot(np.arange(n), reference_trajectory_tensor[:, i], label=y_label_list[i])

        plt.legend(loc='best')
        plt.grid(True)

    plt.tight_layout()
    plt.savefig("trajectory.png", dpi=300)
    # plt.show()


def rot_mat_3d(angle, axis):
    """
    Create a 3D rotation matrix for rotating points around a specified axis.

    Parameters:
        angle (float): The angle of rotation in radians.
        axis (str): The axis of rotation ('x', 'y', or 'z').

    Returns:
        numpy.ndarray: The 3D rotation matrix.
    """
    cos_theta = np.cos(angle)
    sin_theta = np.sin(angle)
    if axis == 'x':
        return np.array([[1, 0, 0],
                         [0, cos_theta, -sin_theta],
                         [0, sin_theta, cos_theta]])
    elif axis == 'y':
        return np.array([[cos_theta, 0, sin_theta],
                         [0, 1, 0],
                         [-sin_theta, 0, cos_theta]])
    elif axis == 'z':
        return np.array([[cos_theta, -sin_theta, 0],
                         [sin_theta, cos_theta, 0],
                         [0, 0, 1]])


def find_latest_file(dir):
    latest_file = None
    latest_mtime = None

    for root, dirs, files in os.walk(dir):
        for file in files:
            file_path = os.path.join(root, file)
            mtime = os.path.getmtime(file_path)
            if latest_mtime is None or mtime > latest_mtime:
                latest_file = file_path
                latest_mtime = mtime

    return latest_file


if __name__ == '__main__':

    if len(sys.argv) == 1:
        folder_name = "real_plant"
        file_order = -1
    else:
        folder_name = str(sys.argv[1])
        file_order = int(sys.argv[2])

    # dir_name = f"saved/logs/robot/{folder_name}"
    dir_name = f"saved/logs/robotr/{folder_name}"

    # dir_name = "logs/robot/real_plant"
    files = os.listdir(dir_name)
    file_list = sorted(files, key=lambda x: os.path.getmtime(os.path.join(dir_name, x)))
    print(f"filepath: {dir_name}/{file_list[file_order]}")
    plot_robot_trajectory(filepath=f"{dir_name}/{file_list[file_order]}")

    # plot_robot_trajectory("saved/logs/real_plant/updated_patch.pkl")