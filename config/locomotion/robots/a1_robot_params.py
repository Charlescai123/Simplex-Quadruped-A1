"""Robot Parameters for A1 Real Plant"""

import numpy as np
from locomotion.robots.motors import MotorControlMode
from config.locomotion.robots.pose import Pose
from config.locomotion.robots.a1_params import A1Params
from config.locomotion.robots.motor_params import MotorGroupParams


# A1 Params for Real
class A1RobotParams(A1Params):

    def __init__(self):
        super().__init__()

        # Simulator settings
        self.time_step = 0.002
        self.action_repeat = 1
        self.reset_time = 3
        self.sync_gui = False  # Whether to sync simulator and real-world action in GUI

        # Motor settings
        self.motor_control_mode = MotorControlMode.HYBRID
        self.motor_init_position = Pose.LAY_DOWN_POSE
        self.motor_init_target_position = Pose.STANDING_POSE

        # Stance Leg settings
        self.mpc_body_height = 0.24
        self.mpc_body_mass = 110 / 9.8
        # self.mpc_body_mass = 108 / 9.8
        # self.mpc_body_inertia = np.array((0.017, 0, 0, 0, 0.057, 0, 0, 0, 0.064)) * 10.

        self.mpc_body_inertia = np.array((0.027, 0, 0, 0, 0.057, 0, 0, 0, 0.064)) * 5.  # from fast_and_efficient
        # self.mpc_body_inertia = np.array((0.017, 0, 0, 0, 0.057, 0, 0, 0, 0.064)) * 4 # from locomotion-simulation

        # Constants for analytical FK/IK
        self.com_offset = -np.array([0.012731, 0.002186, 0.000515])
        self.hip_offset = np.array([[0.183, -0.047, 0.], [0.183, 0.047, 0.],
                                    [-0.183, -0.047, 0.], [-0.183, 0.047, 0.]
                                    ]) + self.com_offset

        self.window_size = 60  # should be [40 - 70]

        self.safe_height = 0.12
        self.desired_vx = 0.3
        self.desired_vy = 0
        self.desired_wz = 0
