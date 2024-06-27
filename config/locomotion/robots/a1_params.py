"""Robot Parameters for A1 in Simulation"""

import numpy as np
from locomotion.robots.motors import MotorControlMode
from config.locomotion.robots.pose import Pose


# A1 Params for Sim
class A1Params:

    def __init__(self):
        # A1 Model settings
        self.urdf_path = "a1.urdf"
        self.base_joint_names = ()
        self.foot_joint_names = (
            "FR_toe_fixed",
            "FL_toe_fixed",
            "RR_toe_fixed",
            "RL_toe_fixed",
        )

        # Simulator settings
        self.time_step = 0.002
        self.action_repeat = 1
        # self.reset_time = 1
        self.reset_time = 0
        self.num_solver_iterations = 30
        self.enable_cone_friction = 0
        self.on_rack = False
        self.init_rack_position = [0, 0, 1]
        # self.init_position = [0, 0, 0.32]
        self.init_position = [0, 0, 0.26]
        self.sync_gui = False  # Whether to sync simulator and real-world action in GUI
        self.camera_fixed = False

        # Motor settings
        # self.motor_config_path = "config/jsons/a1_robot/motors.json"
        self.motor_control_mode = MotorControlMode.HYBRID
        # self.motor_init_position = Pose.LAY_DOWN_POSE
        self.motor_init_position = Pose.STANDING_POSE
        self.motor_init_target_position = Pose.STANDING_POSE

        # Gait settings
        self.init_gait_phase = np.zeros(4)  # Enter swing immediately after start
        # self.init_gait_phase = np.array([-0.2 * np.pi, 0, 0, -0.2 * np.pi])  # Stance for a while after start
        self.gait_params = [2., np.pi, np.pi, 0, 0.4]  # [freq, theta1, theta2, theta3, theta_swing_cutoff]

        # Stance Leg settings
        self.mpc_body_height = 0.24
        # self.mpc_body_mass = 110 / 9.8
        self.mpc_body_mass = 108 / 9.8
        # self.mpc_body_inertia = np.array((0.017, 0, 0, 0, 0.057, 0, 0, 0, 0.064)) * 10.

        # self.mpc_body_inertia = np.array((0.027, 0, 0, 0, 0.057, 0, 0, 0, 0.064)) * 5. # from fast_and_efficient
        self.mpc_body_inertia = np.array([0.068, 0., 0., 0., 0.228, 0., 0., 0., 0.256])  # from locomotion-simulation

        # Swing Leg settings
        self.swing_reference_positions = (
            (0.17, -0.135, 0),
            (0.17, 0.13, 0),
            (-0.195, -0.135, 0),
            (-0.195, 0.13, 0),
        )

        # State Estimator settings
        self.window_size = 20
        self.ground_normal_window_size = 10

        self.safe_height = 0.12
        self.desired_vx = 0.2
        self.desired_vy = 0
        self.desired_wz = 0
