# Lint as: python3
"""A torque based stance controller framework."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
from typing import Any, Sequence, Tuple
import pybullet as p
import numpy as np

from omegaconf import DictConfig
from src.envs.robot.unitree_a1.motors import MotorCommand
from src.envs.robot.gait_scheduler import gait_scheduler as gait_scheduler_lib
from src.envs.robot.mpc_controller import qp_torque_optimizer


# from config.variables import FORCE_DIMENSION, MAX_DDQ, MIN_DDQ, QP_KP, QP_KD
# from config.variables import QP_FRICTION_COEFF
# from utils.utils import rot_mat_3d


class TorqueStanceLegController:
    """A torque based stance leg controller framework.

    Takes in high level robot like walking speed and turning speed, and
    generates necessary the torques for stance legs.
    """

    def __init__(
            self,
            robot: Any,
            gait_scheduler: Any,
            state_estimator: Any,
            stance_params: DictConfig,
            desired_speed: Tuple[float, float] = (0, 0),
            desired_twisting_speed: float = 0,
            desired_com_height: float = 0.24,
            body_mass: float = 110 / 9.8,
            body_inertia: Tuple[float, float, float, float, float, float, float, float, float] = (
                    0.07335, 0, 0, 0, 0.25068, 0, 0, 0, 0.25447),
            num_legs: int = 4,
            # friction_coeffs: Sequence[float] = tuple([QP_FRICTION_COEFF] * 4),
    ):
        """Initializes the class.

        Tracks the desired position/velocity of the robot by computing proper joint
        torques using MPC module.

        Args:
          robot: A robot instance.
          gait_scheduler: Used to query the locomotion phase and leg states.
          state_estimator: Estimate the robot states (e.g. CoM velocity).
          desired_speed: desired CoM speed in x-y plane.
          desired_twisting_speed: desired CoM rotating speed in z direction.
          desired_body_height: The standing height of the robot.
          body_mass: The total mass of the robot.
          body_inertia: The inertia matrix in the body principle frame. We assume
            the body principle coordinate frame has x-forward and z-up.
          num_legs: The number of legs used for force planning.
          friction_coeffs: The friction coeffs on the contact surfaces.
        """
        self._robot = robot
        self._params = stance_params
        self._gait_scheduler = gait_scheduler
        self._state_estimator = state_estimator
        self.desired_speed = np.array((desired_speed[0], desired_speed[1], 0))
        self.desired_twisting_speed = desired_twisting_speed

        self._desired_body_height = desired_com_height
        self._body_mass = body_mass
        self._body_inertia = body_inertia
        self._num_legs = num_legs

        self._max_ddq = np.asarray(self._params.ddq_bound) * self._params.ddq_bound_magnitude
        self._min_ddq = -1 * self._max_ddq
        self._kp = np.asarray(self._params.ddq_kp)
        self._kd = np.asarray(self._params.ddq_kd)

        self._acc_weights = np.asarray(self._params.acc_weights)
        self._reg_weight = self._params.reg_weight
        self._friction_coeffs = np.asarray(tuple([self._params.friction_coeff] * 4))

        self._qp_torque_optimizer = qp_torque_optimizer.QPTorqueOptimizer(
            robot_mass=self._body_mass,
            robot_inertia=self._body_inertia,
            friction_coef=self._params.friction_coeff
        )

        # Variables for recording
        self._stance_action = None
        self._ground_reaction_forces = np.nan
        self._error_q = np.nan
        self._error_dq = np.nan
        self._phy_ddq = np.array([0, 0, 0, 0, 0, 0])
        self._drl_ddq = np.array([0, 0, 0, 0, 0, 0])
        self._total_ddq = self._phy_ddq + self._drl_ddq

    @property
    def stance_action(self):
        return self._stance_action

    @property
    def ground_reaction_forces(self):
        return self._ground_reaction_forces

    @property
    def tracking_error(self):
        return np.vstack((self._error_q, self._error_dq))

    @property
    def stance_ddq(self):
        return np.vstack((self._phy_ddq, self._drl_ddq, self._total_ddq))

    @property
    def stance_ddq_limit(self):
        return np.vstack((self._min_ddq, self._max_ddq))

    def reset(self, current_time):
        del current_time

    def update(self, current_time):
        del current_time

    def _estimate_robot_height(self, contacts):
        if np.sum(contacts) == 0:
            # All foot in air, no way to estimate
            return self._desired_body_height
        else:
            # base_orientation = self._robot.GetBaseOrientation()
            base_orientation = self._robot.base_orientation_quaternion
            rot_mat = self._robot.pybullet_client.getMatrixFromQuaternion(
                base_orientation)
            rot_mat = np.array(rot_mat).reshape((3, 3))

            # foot_positions = self._robot.GetFootPositionsInBaseFrame()
            foot_positions = self._robot.foot_positions_in_body_frame
            foot_positions_world_frame = (rot_mat.dot(foot_positions.T)).T
            # pylint: disable=unsubscriptable-object
            useful_heights = contacts * (-foot_positions_world_frame[:, 2])
            return np.sum(useful_heights) / np.sum(contacts)

    def get_model_action(self):

        # Robot state
        robot_com_position = self._state_estimator.com_position_in_ground_frame
        robot_com_velocity = self._state_estimator.com_velocity_in_body_frame
        robot_com_roll_pitch_yaw = np.array(
            p.getEulerFromQuaternion(self._state_estimator.com_orientation_quaternion_in_ground_frame))
        robot_com_roll_pitch_yaw_rate = self._robot.base_angular_velocity_in_body_frame

        # robot_com_roll_pitch_yaw[2] = 0  # To prevent yaw drifting

        # robot_com_roll_pitch_yaw_rate = rot_mat_3d(angle=robot_com_roll_pitch_yaw[2],
        #                                            axis='z') @ self._robot.base_angular_velocity_in_body_frame

        # Robot q and dq
        robot_q = np.hstack((robot_com_position, robot_com_roll_pitch_yaw))
        robot_dq = np.hstack((robot_com_velocity, robot_com_roll_pitch_yaw_rate))

        # Desired state
        desired_com_position = np.array((0., 0., self._desired_body_height), dtype=np.float64)
        desired_com_velocity = np.array((self.desired_speed[0], self.desired_speed[1], 0.), dtype=np.float64)
        desired_com_roll_pitch_yaw = np.array((0., 0., 0.), dtype=np.float64)
        desired_com_angular_velocity = np.array((0., 0., self.desired_twisting_speed), dtype=np.float64)

        # Desired q and dq
        desired_q = np.hstack((desired_com_position, desired_com_roll_pitch_yaw))
        desired_dq = np.hstack((desired_com_velocity, desired_com_angular_velocity))

        # Physics (model-based) ddq
        phy_ddq = self._kp * (desired_q - robot_q) + self._kd * (desired_dq - robot_dq)

        return phy_ddq

    def get_action(self, drl_action: np.ndarray = None):
        """Computes the torque for stance legs."""
        # print("----------------------------------- Stance Control Quadprog -----------------------------------")
        # s = time.time()
        self._phy_ddq = self.get_model_action()
        # print(f"phy_ddq: {self._phy_ddq}")
        # e = time.time()
        # print(f"phy_ddq time: {e - s}")
        # Residual action
        if drl_action is not None:
            self._drl_ddq = drl_action
            desired_ddq = self._phy_ddq + self._drl_ddq
        else:
            desired_ddq = self._phy_ddq

        # self._total_ddq = np.clip(desired_ddq, self._min_ddq, self._max_ddq)

        # Wrap ddq
        terminal_ddq = np.clip(desired_ddq, self._min_ddq, self._max_ddq)

        # Actual q and dq
        contacts = np.array(
            [(leg_state in (gait_scheduler_lib.LegState.STANCE,
                            gait_scheduler_lib.LegState.EARLY_CONTACT))
             for leg_state in self._gait_scheduler.desired_leg_states],
            dtype=np.int32)

        # Calculate needed contact forces
        foot_positions = self._robot.foot_positions_in_body_frame

        contact_forces = self._qp_torque_optimizer.compute_contact_force(
            foot_positions=foot_positions,
            desired_acc=terminal_ddq,
            contacts=contacts,
            acc_weights=self._acc_weights,
            reg_weight=self._reg_weight
        )

        # print(f"foot_positions_in_body_frame: {foot_positions}")
        # print(f"desired_ddq: {desired_ddq}")
        # print(f"contact_forces: {contact_forces}")
        # print("//////////////////////////////////////////////////")

        leg_action = {}
        for leg_id, force in enumerate(contact_forces):
            # While "Lose Contact" is useful in simulation, in real environment it's
            # susceptible to sensor noise. Disabling for now.
            motor_torques = self._robot.map_contact_force_to_joint_torques(leg_id, force)
            for joint_id, torque in motor_torques.items():
                leg_action[joint_id] = MotorCommand(desired_position=0,
                                                    kp=0,
                                                    desired_velocity=0,
                                                    kd=0,
                                                    desired_torque=torque)

        print("-----------------------------------------------------------------------------------------------")

        # Save values for record
        self._stance_action = leg_action
        self._ground_reaction_forces = contact_forces
        # self._error_q = robot_q - desired_q
        # self._error_dq = robot_dq - desired_dq

        return leg_action, contact_forces

    def map_ddq_to_action(self, ddq):
        # Wrap ddq
        terminal_ddq = np.clip(ddq, self._min_ddq, self._max_ddq)

        # Actual q and dq
        contacts = np.array(
            [(leg_state in (gait_scheduler_lib.LegState.STANCE,
                            gait_scheduler_lib.LegState.EARLY_CONTACT))
             for leg_state in self._gait_scheduler.desired_leg_states],
            dtype=np.int32)

        # Calculate needed contact forces
        foot_positions = self._robot.foot_positions_in_body_frame

        contact_forces = self._qp_torque_optimizer.compute_contact_force(
            foot_positions=foot_positions,
            desired_acc=terminal_ddq,
            contacts=contacts,
            acc_weights=self._acc_weights,
            reg_weight=self._reg_weight
        )

        # print(f"foot_positions_in_body_frame: {foot_positions}")
        # print(f"desired_ddq: {desired_ddq}")
        # print(f"contact_forces: {contact_forces}")
        # print("//////////////////////////////////////////////////")

        leg_action = {}
        for leg_id, force in enumerate(contact_forces):
            # While "Lose Contact" is useful in simulation, in real environment it's
            # susceptible to sensor noise. Disabling for now.
            motor_torques = self._robot.map_contact_force_to_joint_torques(leg_id, force)
            for joint_id, torque in motor_torques.items():
                leg_action[joint_id] = MotorCommand(desired_position=0,
                                                    kp=0,
                                                    desired_velocity=0,
                                                    kd=0,
                                                    desired_torque=torque)

        return leg_action, contact_forces
