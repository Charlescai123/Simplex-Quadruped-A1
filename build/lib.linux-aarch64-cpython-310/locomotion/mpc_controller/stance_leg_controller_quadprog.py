# Lint as: python3
"""A torque based stance controller framework."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Any, Sequence, Tuple
import pybullet as p
import numpy as np
import time

from locomotion.robots.motors import MotorCommand
from locomotion.gait_scheduler import gait_scheduler as gait_scheduler_lib
# from mpc_controller import leg_controller
from locomotion.mpc_controller import qp_torque_optimizer
# from config.variables import FORCE_DIMENSION, MAX_DDQ, MIN_DDQ, QP_KP, QP_KD
# from config.variables import QP_FRICTION_COEFF
from config.locomotion.controllers.stance_params import StanceControllerParams


# _FORCE_DIMENSION = 3
# KP = np.array((0., 0., 100., 100., 100., 0.))
# KD = np.array((40., 30., 10., 10., 10., 30.))
# MAX_DDQ = np.array((10., 10., 10., 20., 20., 20.))
# MIN_DDQ = -MAX_DDQ


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
            stance_params: StanceControllerParams,
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

        self._friction_coeffs = np.array(tuple([self._params.friction_coeff] * 4))

        self._qp_torque_optimizer = qp_torque_optimizer.QPTorqueOptimizer(
            robot_mass=self._body_mass,
            robot_inertia=self._body_inertia
        )

        # Variables for recording
        self._stance_action = None
        self._ground_reaction_forces = np.nan
        self._error_q = np.array([0, 0, 0, 0, 0, 0])
        self._error_dq = np.array([0, 0, 0, 0, 0, 0])
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
        return np.hstack((self._error_q, self._error_dq)).reshape(12, 1)

    @property
    def stance_ddq(self):
        print(f"phy_ddq:{self._phy_ddq}")
        print(f"drl_ddq:{self._drl_ddq}")
        print(f"total_ddq:{self._total_ddq}")
        return np.vstack((self._phy_ddq, self._drl_ddq, self._total_ddq))

    @property
    def stance_ddq_limit(self):
        return np.vstack((self._params.min_ddq, self._params.max_ddq))

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

    def get_action(self, drl_action: np.ndarray = None):
        """Computes the torque for stance legs."""
        # print("----------------------------------- Stance Control Quadprog -----------------------------------")
        s = time.time()

        # Actual q and dq
        contacts = np.array(
            [(leg_state in (gait_scheduler_lib.LegState.STANCE,
                            gait_scheduler_lib.LegState.EARLY_CONTACT))
             for leg_state in self._gait_scheduler.desired_leg_states],
            dtype=np.int32)

        robot_com_position = self._state_estimator.com_position_in_ground_frame
        robot_com_velocity = self._state_estimator.com_velocity_in_body_frame

        robot_com_roll_pitch_yaw = np.array(
            p.getEulerFromQuaternion(self._state_estimator.com_orientation_quaternion_in_ground_frame))

        robot_com_roll_pitch_yaw[2] = 0  # To prevent yaw drifting
        robot_com_roll_pitch_yaw_rate = self._robot.base_angular_velocity_in_body_frame

        # print(f"robot_com_position: {robot_com_position}")
        # print(f"robot_com_velocity: {robot_com_velocity}")
        # print(f"robot_com_roll_pitch_yaw: {robot_com_roll_pitch_yaw}")
        # print(f"robot com roll pitch yaw rate: {robot_com_roll_pitch_yaw_rate}")
        # robot_com_roll_pitch_yaw_rate = rot_mat_3d(angle=robot_com_roll_pitch_yaw[2],
        #                                            axis='z') @ self._robot.base_angular_velocity_in_body_frame

        robot_q = np.hstack((robot_com_position, robot_com_roll_pitch_yaw))
        robot_dq = np.hstack((robot_com_velocity, robot_com_roll_pitch_yaw_rate))

        e1 = time.time()

        # Desired q and dq
        desired_com_position = np.array((0., 0., self._desired_body_height),
                                        dtype=np.float64)
        desired_com_velocity = np.array(
            (self.desired_speed[0], self.desired_speed[1], 0.), dtype=np.float64)

        desired_com_roll_pitch_yaw = np.array((0., 0., 0.), dtype=np.float64)

        desired_com_angular_velocity = np.array(
            (0., 0., self.desired_twisting_speed), dtype=np.float64)

        desired_q = np.hstack((desired_com_position, desired_com_roll_pitch_yaw))
        desired_dq = np.hstack(
            (desired_com_velocity, desired_com_angular_velocity))

        # Desired ddq
        QP_KP = self._params.qp_kp
        QP_KD = self._params.qp_kd
        MIN_DDQ = self._params.min_ddq
        MAX_DDQ = self._params.max_ddq
        self._phy_ddq = QP_KP @ (desired_q - robot_q) + QP_KD @ (desired_dq - robot_dq)
        print(f"desired_q: {desired_q}")
        print(f"robot_q: {robot_q}")
        print(f"desired_dq: {desired_dq}")
        print(f"robot_dq: {robot_dq}")
        print(f"Kp: {QP_KP}")
        print(f"Kd: {QP_KD}")
        print(f"phy_ddq: {self._phy_ddq}")

        # Residual action
        if drl_action is not None:
            self._drl_ddq = drl_action
            desired_ddq = self._phy_ddq + self._drl_ddq
        else:
            desired_ddq = self._phy_ddq

        self._total_ddq = np.clip(desired_ddq, MIN_DDQ, MAX_DDQ)

        # Calculate needed contact forces
        foot_positions = self._robot.foot_positions_in_body_frame
        e2 = time.time()

        contact_forces = self._qp_torque_optimizer.compute_contact_force(
            foot_positions=foot_positions,
            desired_acc=self._total_ddq,
            contacts=contacts,
            acc_weights=self._params.acc_weights,
            reg_weight=self._params.reg_weight,
        )
        e3 = time.time()

        # print(f"foot_positions_in_body_frame: {foot_positions}")
        # print(f"desired_ddq: {desired_ddq}")
        # print(f"contact_forces: {contact_forces}")
        # print("//////////////////////////////////////////////////")

        action = {}
        for leg_id, force in enumerate(contact_forces):
            # While "Lose Contact" is useful in simulation, in real environment it's
            # susceptible to sensor noise. Disabling for now.
            motor_torques = self._robot.map_contact_force_to_joint_torques(leg_id, force)
            for joint_id, torque in motor_torques.items():
                action[joint_id] = MotorCommand(desired_position=0,
                                                kp=0,
                                                desired_velocity=0,
                                                kd=0,
                                                desired_torque=torque)
        # print("After IK: {}".format(time.time() - start_time))
        e4 = time.time()
        # print("-----------------------------------------------------------------------------------------------")
        # print(f"stance part 1 time: {e1 - s}")
        # print(f"stance part 2 time: {e2 - e1}")
        # print(f"stance part 3 time: {e3 - e2}")
        # print(f"stance part 4 time: {e4 - e3}")

        # Save values for record
        self._stance_action = action
        self._ground_reaction_forces = contact_forces
        self._error_q = desired_q - robot_q
        self._error_dq = desired_dq - robot_dq

        return action, contact_forces

    def get_hac_action(self, chi, state_trig, F_kp: np.ndarray = None, F_kd: np.ndarray = None):
        """Computes the torque for stance legs."""
        # print("------------------------------- Simplex Stance Control Quadprog -------------------------------")
        s = time.time()

        # Actual q and dq
        contacts = np.array(
            [(leg_state in (gait_scheduler_lib.LegState.STANCE,
                            gait_scheduler_lib.LegState.EARLY_CONTACT))
             for leg_state in self._gait_scheduler.desired_leg_states],
            dtype=np.int32)

        robot_com_position = self._state_estimator.com_position_in_ground_frame
        robot_com_velocity = self._state_estimator.com_velocity_in_body_frame
        robot_com_roll_pitch_yaw = np.array(
            p.getEulerFromQuaternion(self._state_estimator.com_orientation_quaternion_in_ground_frame))

        robot_com_roll_pitch_yaw[2] = 0  # To prevent yaw drifting
        robot_com_roll_pitch_yaw_rate = self._robot.base_angular_velocity_in_body_frame
        # robot_com_roll_pitch_yaw_rate = rot_mat_3d(angle=robot_com_roll_pitch_yaw[2],
        #                                            axis='z') @ self._robot.base_angular_velocity_in_body_frame

        robot_q = np.hstack((robot_com_position, robot_com_roll_pitch_yaw))
        robot_dq = np.hstack((robot_com_velocity, robot_com_roll_pitch_yaw_rate))

        e1 = time.time()

        # Desired q and dq
        desired_com_position = np.array((0., 0., self._desired_body_height),
                                        dtype=np.float64)
        desired_com_velocity = np.array(
            (self.desired_speed[0], self.desired_speed[1], 0.), dtype=np.float64)

        desired_com_roll_pitch_yaw = np.array((0., 0., 0.), dtype=np.float64)

        desired_com_angular_velocity = np.array(
            (0., 0., self.desired_twisting_speed), dtype=np.float64)

        desired_q = np.hstack((desired_com_position, desired_com_roll_pitch_yaw))
        desired_dq = np.hstack(
            (desired_com_velocity, desired_com_angular_velocity))

        # Desired ddq
        # self._phy_ddq = self._kp * (desired_q - robot_q) + self._kd * (desired_dq - robot_dq)

        err_q = (desired_q - robot_q).reshape(6, 1)
        err_dq = (desired_dq - robot_dq).reshape(6, 1)
        # err_state = np.hstack((err_q, err_dq)).T

        # kp = np.diag(F_hat[6:12, 0:6])
        # kd = np.diag(F_hat[6:12, 6:12])

        # kp = np.diag([0, 0, 63, 33, 33, 31])
        # kd = np.diag([24, 20, 20, 22, 22, 22])

        kp = F_kp
        kd = F_kd

        # kp = np.array([0.1, 0.1, 100., 100., 100., 0.1])
        # kd = np.array([40., 30., 10., 10., 10., 30.])
        # print("!!!!!!!!!!!!!!!!!!!!!!!")
        # print(state_trig[:6].shape)
        # print(err_q.shape)
        # time.sleep(0.5)
        self._phy_ddq = (kp @ (err_q - chi * state_trig[:6])
                         + kd @ (err_dq - chi * state_trig[6:])).squeeze()
        print(f"simplex ddq: {self._phy_ddq}")
        desired_ddq = self._phy_ddq

        MIN_DDQ = self._params.min_ddq
        MAX_DDQ = self._params.max_ddq
        self._total_ddq = np.clip(desired_ddq, MIN_DDQ, MAX_DDQ)

        # Calculate needed contact forces
        foot_positions = self._robot.foot_positions_in_body_frame
        e2 = time.time()

        contact_forces = self._qp_torque_optimizer.compute_contact_force(
            foot_positions=foot_positions,
            desired_acc=self._total_ddq,
            contacts=contacts,
            acc_weights=self._params.acc_weights,
            reg_weight=self._params.reg_weight,
        )
        e3 = time.time()

        # print(f"foot_positions_in_body_frame: {foot_positions}")
        # print(f"desired_ddq: {desired_ddq}")
        # print(f"contact_forces: {contact_forces}")
        # print("//////////////////////////////////////////////////")

        action = {}
        for leg_id, force in enumerate(contact_forces):
            # While "Lose Contact" is useful in simulation, in real environment it's
            # susceptible to sensor noise. Disabling for now.
            motor_torques = self._robot.map_contact_force_to_joint_torques(leg_id, force)
            for joint_id, torque in motor_torques.items():
                action[joint_id] = MotorCommand(desired_position=0,
                                                kp=0,
                                                desired_velocity=0,
                                                kd=0,
                                                desired_torque=torque)
        # print("After IK: {}".format(time.time() - start_time))
        e4 = time.time()
        # print("-----------------------------------------------------------------------------------------------")
        # print(f"stance part 1 time: {e1 - s}")
        # print(f"stance part 2 time: {e2 - e1}")
        # print(f"stance part 3 time: {e3 - e2}")
        # print(f"stance part 4 time: {e4 - e3}")

        # Save values for record
        self._stance_action = action
        self._ground_reaction_forces = contact_forces
        self._error_q = robot_q - desired_q
        self._error_dq = robot_dq - desired_dq

        return action, contact_forces
