"""Example of running A1 robot with position control.

To run:
python -m examples.a1_exercise_example  --use_real_robot=[True|False]
"""
from absl import app
from absl import flags

import ml_collections
import numpy as np
import pybullet
from pybullet_utils import bullet_client
import time
from typing import Tuple

from src.envs.locomotion.robots.motors import MotorCommand
from config_json.locomotion.robots import a1_params
from config_json.locomotion.robots import a1_robot_params
from config_json.locomotion.robots import motor_params
from config_json.locomotion.controllers import swing_params
from config_json.locomotion.controllers import stance_params
from src.envs.locomotion.robots import a1, a1_robot

flags.DEFINE_bool('use_real_robot', False, 'whether to use real robot.')
FLAGS = flags.FLAGS


def get_dance_action(robot, t):
    # current_motor_angle = np.array([0., 0., 0.] * 4)
    # desired_motor_angle = np.array([0., 0.9, -1.8] * 4)
    # blend_ratio = np.minimum(t / 200., 1)
    # action = (1 - blend_ratio
    #           ) * current_motor_angle + blend_ratio * desired_motor_angle

    FREQ = 1
    angle_hip = 0.9 + 0.2 * np.sin(2 * np.pi * FREQ * t)
    angle_calf = -2 * angle_hip
    action = np.array([0., angle_hip, angle_calf] * 4)

    return MotorCommand(desired_position=action,
                        kp=robot.motor_group.kps,
                        desired_velocity=np.zeros(robot.num_motors),
                        kd=robot.motor_group.kds)

    # mid_action = np.array([0.0, 0.9, -1.8] * 4)
    # amplitude = np.array([0.0, 0.2, -0.4] * 4)
    # freq = 1.0
    # return MotorCommand(desired_position=mid_action +
    #                     amplitude * np.sin(2 * np.pi * freq * t),
    #                     kp=robot.motor_group.kps,
    #                     desired_velocity=np.zeros(robot.num_motors),
    #                     kd=robot.motor_group.kds)


def main(_):
    if FLAGS.use_real_robot:
        p = bullet_client.BulletClient(connection_mode=pybullet.DIRECT)
    else:
        p = bullet_client.BulletClient(connection_mode=pybullet.GUI)
    p.setAdditionalSearchPath('simulator/data')
    p.loadURDF("plane.urdf")
    p.setGravity(0.0, 0.0, -9.8)

    if FLAGS.use_real_robot:
        robot = a1_robot.A1Robot(pybullet_client=p,
                                 a1_robot_params=a1_robot_params.A1RobotParams(),
                                 motor_params=motor_params.MotorGroupParams(),
                                 swing_params=swing_params.SwingControllerParams(),
                                 stance_params=stance_params.StanceControllerParams())
    else:
        robot = a1.A1(pybullet_client=p,
                      a1_params=a1_params.A1Params(),
                      motor_params=motor_params.MotorGroupParams(),
                      swing_params=swing_params.SwingControllerParams(),
                      stance_params=stance_params.StanceControllerParams())
    robot.reset()

    for _ in range(10000):
        action = get_dance_action(robot, robot.time_since_reset)
        robot.step(action)
        # time.sleep(2)
        print(f"action is: {action}")
        # print(robot.base_orientation_rpy)


if __name__ == "__main__":
    app.run(main)
