"""Base class for all robot."""
import ml_collections
import numpy as np
from typing import Any
from typing import Sequence
from typing import Tuple, Optional
import json
import time
from locomotion import wbc_controller
from locomotion.mpc_controller import swing_leg_controller
from locomotion.gait_scheduler import offset_gait_scheduler
from locomotion.state_estimator import com_velocity_estimator
from locomotion.robots import kinematics, a1
from locomotion.robots.motors import MotorControlMode
from locomotion.robots.motors import MotorGroup
from locomotion.robots.motors import MotorModel
from locomotion.robots.motors import MotorCommand
from locomotion.robots.quadruped import QuadrupedRobot
from config.locomotion.robots.a1_params import A1Params
# from config.phydrl.env_params import TrainerEnvParams
from config.locomotion.robots.motor_params import MotorGroupParams
from config.locomotion.controllers.swing_params import SwingControllerParams
from config.locomotion.controllers.stance_params import StanceControllerParams
from config.locomotion.robots.pose import Pose



class A1RobotPhyDRL(a1.A1):
    """A1 Simulation Robot for Training."""

    def __init__(
            self,
            pybullet_client: Any = None,
            a1_params: A1Params= None,
            motor_params: MotorGroupParams = None,
            swing_params: SwingControllerParams = None,
            stance_params: StanceControllerParams = None,
    ) -> None:
        """Constructs an A1 robot and resets it to the initial states.
        Initializes a tuple with a single MotorGroup containing 12 MotorModels.
        Each MotorModel is by default configured for the robot of the A1.
        """
        # self._a1_params = a1_params
        super().__init__(
            pybullet_client=pybullet_client,
            a1_params=a1_params,
            motor_params=motor_params,
            swing_params=swing_params,
            stance_params=stance_params
        )

    def _foot_position_in_hip_frame_to_joint_angle(self,
                                                   foot_position,
                                                   l_hip_sign=1):
        """Computes foot inverse kinematics analytically."""
        l_up = 0.2
        l_low = 0.2
        l_hip = 0.08505 * l_hip_sign
        x, y, z = foot_position[0], foot_position[1], foot_position[2]
        theta_knee = -np.arccos(
            np.clip((x ** 2 + y ** 2 + z ** 2 - l_hip ** 2 - l_low ** 2 - l_up ** 2) /
                    (2 * l_low * l_up), -1, 1))
        l = np.sqrt(
            np.maximum(l_up ** 2 + l_low ** 2 + 2 * l_up * l_low * np.cos(theta_knee),
                       1e-7))
        theta_hip = np.arcsin(np.clip(-x / l, -1, 1)) - theta_knee / 2
        c1 = l_hip * y - l * np.cos(theta_hip + theta_knee / 2) * z
        s1 = l * np.cos(theta_hip + theta_knee / 2) * y + l_hip * z
        theta_ab = np.arctan2(s1, c1)
        return np.array([theta_ab, theta_hip, theta_knee])

    @property
    def _foot_positions_in_hip_frame(self):
        motor_angles = self.motor_angles.reshape((4, 3))
        foot_positions = np.zeros((4, 3))
        for i in range(4):
            foot_positions[i] = self._foot_position_in_hip_frame(
                motor_angles[i], l_hip_sign=(-1) ** (i + 1))
        return foot_positions

    def _foot_position_in_hip_frame(self, angles, l_hip_sign=1):
        """Computes foot forward kinematics analytically."""
        theta_ab, theta_hip, theta_knee = angles[0], angles[1], angles[2]
        l_up = 0.2
        l_low = 0.2
        l_hip = 0.08505 * l_hip_sign
        leg_distance = np.sqrt(l_up ** 2 + l_low ** 2 +
                               2 * l_up * l_low * np.cos(theta_knee))
        eff_swing = theta_hip + theta_knee / 2

        off_x_hip = -leg_distance * np.sin(eff_swing)
        off_z_hip = -leg_distance * np.cos(eff_swing)
        off_y_hip = l_hip

        off_x = off_x_hip
        off_y = np.cos(theta_ab) * off_y_hip - np.sin(theta_ab) * off_z_hip
        off_z = np.sin(theta_ab) * off_y_hip + np.cos(theta_ab) * off_z_hip
        return np.array([off_x, off_y, off_z])

    @property
    def foot_positions_in_body_frame(self):
        """Use analytical FK/IK/Jacobian"""
        # return self._foot_positions_in_hip_frame + HIP_OFFSETS
        return self._foot_positions_in_hip_frame + self._a1_params.hip_offset

    def compute_foot_jacobian(self, leg_id):
        """Computes foot jacobian matrix analytically."""
        motor_angles = self.motor_angles[leg_id * 3:(leg_id + 1) * 3]
        l_up = 0.2
        l_low = 0.2
        l_hip = 0.08505 * (-1) ** (leg_id + 1)

        t1, t2, t3 = motor_angles[0], motor_angles[1], motor_angles[2]
        l_eff = np.sqrt(l_up ** 2 + l_low ** 2 + 2 * l_up * l_low * np.cos(t3))
        t_eff = t2 + t3 / 2
        J = np.zeros((3, 3))
        J[0, 0] = 0
        J[0, 1] = -l_eff * np.cos(t_eff)
        J[0, 2] = l_low * l_up * np.sin(t3) * np.sin(
            t_eff) / l_eff - l_eff * np.cos(t_eff) / 2
        J[1, 0] = -l_hip * np.sin(t1) + l_eff * np.cos(t1) * np.cos(t_eff)
        J[1, 1] = -l_eff * np.sin(t1) * np.sin(t_eff)
        J[1, 2] = -l_low * l_up * np.sin(t1) * np.sin(t3) * np.cos(
            t_eff) / l_eff - l_eff * np.sin(t1) * np.sin(t_eff) / 2
        J[2, 0] = l_hip * np.cos(t1) + l_eff * np.sin(t1) * np.cos(t_eff)
        J[2, 1] = l_eff * np.sin(t_eff) * np.cos(t1)
        J[2, 2] = l_low * l_up * np.sin(t3) * np.cos(t1) * np.cos(
            t_eff) / l_eff + l_eff * np.sin(t_eff) * np.cos(t1) / 2
        return J

    def get_motor_angles_from_foot_position(self, leg_id, foot_local_position):
        motors_per_leg = self.num_motors // self.num_legs
        joint_position_idxs = list(
            range(leg_id * motors_per_leg,
                  leg_id * motors_per_leg + motors_per_leg))

        joint_angles = self._foot_position_in_hip_frame_to_joint_angle(
            foot_local_position - self._a1_params.hip_offset[leg_id],
            l_hip_sign=(-1) ** (leg_id + 1))
        return joint_position_idxs, joint_angles.tolist()
