# Lint as: python3
"""A torque based stance controller framework."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import logging
from omegaconf import DictConfig
from typing import Any, Sequence, Tuple

import numpy as np
import pybullet as p  # pytype: disable=import-error
import sys
import time

from src.envs.robot.unitree_a1.motors import MotorCommand
from src.envs.robot.gait_scheduler import gait_scheduler as gait_scheduler_lib

try:
    import mpc_osqp as convex_mpc  # pytype: disable=import-error
except:  # pylint: disable=W0702
    print("You need to install PhyDRL-Locomotion")
    print("Run python3 setup.py install --user in this repo")
    sys.exit()

_FORCE_DIMENSION = 3


# The QP weights in the convex MPC formulation. See the MIT paper for details:
#   https://ieeexplore.ieee.org/document/8594448/
# Intuitively, this is the weights of each state dimension when tracking a
# desired CoM trajectory. The full CoM state is represented by
# (roll_pitch_yaw, position, angular_velocity, velocity, gravity_place_holder).

# Best results
# _MPC_WEIGHTS = (1., 1., 0, 0, 0, 10, 0., 0., .1, .1, .1, .0, 0)
#
# # These weights also give good results.
# # _MPC_WEIGHTS = (1., 1., 0, 0, 0, 20, 0., 0., .1, 1., 1., .0, 0.)
#
# PLANNING_HORIZON_STEPS = 10
# PLANNING_TIMESTEP = 0.025


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
            desired_speed: Tuple[float, float] = (0, 0),
            desired_twisting_speed: float = 0,
            desired_com_height: float = 0.24,
            body_mass: float = 110 / 9.8,
            body_inertia: Tuple[float, float, float, float, float, float, float, float, float] = (
                    0.07335, 0, 0, 0, 0.25068, 0, 0, 0, 0.25447),
            stance_params: DictConfig = None,
            num_legs: int = 4,
    ):
        """Initializes the class.

        Tracks the desired position/velocity of the robot by computing proper joint
        torques using MPC module.

        Args:
          robot: A robot instance.
          gait_scheduler: Used to query the robot phase and leg states.
          state_estimator: Estimate the robot states (e.g. CoM velocity).
          desired_speed: desired CoM speed in x-y plane.
          desired_twisting_speed: desired CoM rotating speed in z direction.
          desired_com_height: The standing height of CoM of the robot.
          body_mass: The total mass of the robot.
          body_inertia: The inertia matrix in the body principle frame. We assume
            the body principle coordinate frame has x-forward and z-up.
          num_legs: The number of legs used for force planning.
          friction_coeffs: The friction coeffs on the contact surfaces.
        """
        # TODO: add acc_weight support for mpc torque stance controller.
        self._params = stance_params
        self._robot = robot
        self._gait_scheduler = gait_scheduler
        self._state_estimator = state_estimator
        self.desired_speed = np.array((desired_speed[0], desired_speed[1], 0))
        self.desired_twisting_speed = desired_twisting_speed

        self._desired_body_height = desired_com_height
        self._body_mass = body_mass
        self._body_inertia_list = list(body_inertia)

        self._num_legs = num_legs
        self._friction_coeffs = np.array(tuple([self._params.friction_coeff] * 4))

        self._planning_horizon_steps = stance_params.planning_horizon_steps
        self._planning_timestep = stance_params.planning_timestep

        if np.any(np.isclose(self._friction_coeffs, 1.)):
            raise ValueError("self._cpp_mpc.compute_contact_forces seg faults when "
                             "a friction coefficient is equal to 1.")

        self._weights_list = list(self._params.mpc_weights)
        self._cpp_mpc = convex_mpc.ConvexMpc(self._body_mass,
                                             self._body_inertia_list,
                                             self._num_legs,
                                             self._planning_horizon_steps,
                                             self._planning_timestep,
                                             self._weights_list,
                                             1e-5,
                                             convex_mpc.QPOASES)

        self._future_contact_estimate = np.ones((self._planning_horizon_steps, 4))

        self._stance_action = None
        self._ground_reaction_forces = np.nan

        # Variables for recording
        self._error_q = np.array([0, 0, 0, 0, 0, 0])
        self._error_dq = np.array([0, 0, 0, 0, 0, 0])


    @property
    def tracking_error(self):
        return np.hstack((self._error_q, self._error_dq)).reshape(12, 1)

    @property
    def stance_action(self):
        return self._stance_action

    @property
    def ground_reaction_forces(self):
        return self._ground_reaction_forces

    @property
    def stance_ddq(self):
        phy_ddq = np.array([0, 0, 0, 0, 0, 0])
        drl_ddq = np.array([0, 0, 0, 0, 0, 0])
        total_ddq = np.array([0, 0, 0, 0, 0, 0])
        return np.vstack((phy_ddq, drl_ddq, total_ddq))

    @property
    def stance_ddq_limit(self):
        min_ddq = np.array([0, 0, 0, 0, 0, 0])
        max_ddq = np.array([0, 0, 0, 0, 0, 0])
        return np.vstack((min_ddq, max_ddq))

    def reset(self, current_time):
        del current_time
        # Re-construct CPP solver to remove stochasticity due to warm-start
        self._cpp_mpc = convex_mpc.ConvexMpc(self._body_mass,
                                             self._body_inertia_list,
                                             self._num_legs,
                                             self._planning_horizon_steps,
                                             self._planning_timestep,
                                             self._weights_list,
                                             1e-5,
                                             convex_mpc.QPOASES)

    def update(self, current_time, future_contact_estimate=None):
        del current_time
        self._future_contact_estimate = future_contact_estimate

    def get_action(self, drl_action=None):
        """Computes the torque for stance legs."""

        ############################################## Part 1 ##############################################
        s1 = time.time()
        desired_com_position = np.array(
            (0., 0., self._desired_body_height), dtype=np.float64)

        desired_com_velocity = np.array(
            (self.desired_speed[0], self.desired_speed[1], 0.), dtype=np.float64)

        # Walk parallel to the ground
        desired_com_roll_pitch_yaw = np.zeros(3)
        desired_com_angular_velocity = np.array(
            (0., 0., self.desired_twisting_speed), dtype=np.float64)

        # Desired q and dq
        desired_q = np.hstack((desired_com_position, desired_com_roll_pitch_yaw))
        desired_dq = np.hstack((desired_com_velocity, desired_com_angular_velocity))

        foot_contact_states = np.array(
            [(leg_state in (gait_scheduler_lib.LegState.STANCE,
                            gait_scheduler_lib.LegState.EARLY_CONTACT,
                            gait_scheduler_lib.LegState.LOSE_CONTACT))
             for leg_state in self._gait_scheduler.leg_states],
            dtype=np.int32)

        if not foot_contact_states.any():
            logging.info("No foot in contact...")
            return {}, None

        e1 = time.time()
        print(f".....................................part 1 time: {e1 - s1}")

        ############################################## Part 2 ##############################################
        s2 = time.time()
        if self._future_contact_estimate is not None:
            contact_estimates = self._future_contact_estimate.copy()
            contact_estimates[0] = foot_contact_states
            # print(contact_estimates)
            # input("Any Key...")
        else:
            PLANNING_HORIZON_STEPS = self._planning_horizon_steps
            contact_estimates = np.array([foot_contact_states] * PLANNING_HORIZON_STEPS)
            # print(f"foot_contact_state is: {foot_contact_state}")
            # print(f"contact_estimates is: {contact_estimates}")

        e2 = time.time()
        print(f".....................................part 2 time: {e2 - s2}")

        ############################################## Part 3 ##############################################
        s3 = time.time()
        # com_position = np.array(self._robot.base_position)
        robot_com_position = np.array(self._state_estimator.com_position_in_ground_frame)

        # We use the body yaw aligned world frame for MPC computation.
        # com_roll_pitch_yaw = np.array(self._robot.base_orientation_rpy,
        #                               dtype=np.float64)
        robot_com_roll_pitch_yaw = np.array(p.getEulerFromQuaternion(
            self._state_estimator.com_orientation_quaternion_in_ground_frame))

        # print("Com Position: {}".format(com_position))
        robot_com_roll_pitch_yaw[2] = 0
        # gravity_projection_vec = np.array([0., 0., 1.])

        robot_com_velocity = self._state_estimator.com_velocity_in_body_frame
        robot_com_roll_pitch_yaw_rate = self._robot.base_angular_velocity_in_body_frame

        gravity_projection_vec = np.array(
            self._state_estimator.gravity_projection_vector)
        predicted_contact_forces = [0] * self._num_legs * _FORCE_DIMENSION

        # print("Com RPY: {}".format(com_roll_pitch_yaw))
        # print("Com pos: {}".format(com_position))
        # print("Com Vel: {}".format(
        #         self._state_estimator.com_velocity_ground_frame))
        # print("Ground orientation_world_frame: {}".format(
        #     p.getEulerFromQuaternion(
        #         self._state_estimator.ground_orientation_world_frame)))
        # print("Gravity projection: {}".format(gravity_projection_vec))
        # print("Com RPY Rate: {}".format(self._robot.base_rpy_rate))
        p.submitProfileTiming("predicted_contact_forces")
        e3 = time.time()
        print(f".....................................part 3 time: {e3 - s3}")

        robot_q = np.hstack((robot_com_position, robot_com_roll_pitch_yaw))
        robot_dq = np.hstack((robot_com_velocity, robot_com_roll_pitch_yaw_rate))

        ############################################## Part 4 ##############################################
        s4 = time.time()

        # All computations are conducted under the body ground frame
        predicted_contact_forces = self._cpp_mpc.compute_contact_forces(
            robot_com_position,  # com_position
            np.asarray(robot_com_velocity),  # com_velocity
            np.array(robot_com_roll_pitch_yaw, dtype=np.float64),  # com_roll_pitch_yaw
            gravity_projection_vec,  # Normal Vector of ground
            # Angular velocity in the yaw aligned world frame is actually different
            # from rpy rate. We use it here as a simple approximation.
            # np.asarray(self._state_estimator.com_rpy_rate_ground_frame,
            #            dtype=np.float64),  #com_angular_velocity
            robot_com_roll_pitch_yaw_rate,
            np.asarray(contact_estimates,
                       dtype=np.float64).flatten(),  # Foot contact states
            np.array(self._robot.foot_positions_in_body_frame.flatten(),
                     dtype=np.float64),  # foot_positions_base_frame
            self._friction_coeffs,  # foot_friction_coeffs
            desired_com_position,  # desired_com_position
            desired_com_velocity,  # desired_com_velocity
            desired_com_roll_pitch_yaw,  # desired_com_roll_pitch_yaw
            desired_com_angular_velocity  # desired_com_angular_velocity
        )
        # print("................................................................")
        # print("Parameters for solution:")
        # print(f"desired_com_position: {desired_com_position}")
        # print(f"desired_com_velocity: {desired_com_velocity}")
        # print(f"desired_com_roll_pitch_yaw: {desired_com_roll_pitch_yaw}")
        # print(f"desired_com_angular_velocity: {desired_com_angular_velocity}")
        # print(f"com_position: {com_position}")
        # print(f"com_velocity_in_ground_frame: {self._state_estimator.com_velocity_in_ground_frame}")
        # print(f"com_roll_pitch_yaw: {com_roll_pitch_yaw}")
        # print(f"base_angular_velocity_in_body_frame: {self._robot.base_angular_velocity_in_body_frame}")
        # print(f"contact_estimates: {contact_estimates}")
        # print(f"foot_positions_in_body_frame: {self._robot.foot_positions_in_body_frame}")
        # print(f"friction_coeff: {self._friction_coeffs}")
        # print(f"predicted_contact_forces: {predicted_contact_forces}")

        p.submitProfileTiming()

        # sol = np.array(predicted_contact_forces).reshape((-1, 12))
        # x_dim = np.array([0, 3, 6, 9])
        # y_dim = x_dim + 1
        # z_dim = y_dim + 1

        # logging.info("X_forces: {}".format(-sol[:5, x_dim]))
        # logging.info("Y_forces: {}".format(-sol[:5, y_dim]))
        # logging.info("Z_forces: {}".format(-sol[:5, z_dim]))
        # import pdb
        # pdb.set_trace()
        # input("Any Key...")

        e4 = time.time()
        print(f".....................................part 4 time: {e4 - s4}")

        ############################################## Part 5 ##############################################
        s5 = time.time()

        contact_forces = {}
        contact_forces_record = []
        for i in range(self._num_legs):
            forces = predicted_contact_forces[
                     i * _FORCE_DIMENSION: (i + 1) * _FORCE_DIMENSION]
            contact_forces[i] = np.array(forces)
            contact_forces_record.append(forces)
        print(f"contact_forces: {contact_forces}")
        # input("Any Key...")

        action = {}
        for leg_id, force in contact_forces.items():
            # While "Lose Contact" is useful in simulation, in real environment it's
            # susceptible to sensor noise. Disabling for now.
            motor_torques = self._robot.map_contact_force_to_joint_torques(
                leg_id, force)
            for joint_id, torque in motor_torques.items():
                action[joint_id] = MotorCommand(desired_position=0,
                                                kp=0,
                                                desired_velocity=0,
                                                kd=0,
                                                desired_torque=torque)
        # print("After IK: {}".format(time.time() - start_time))
        e5 = time.time()
        print(f".....................................part 5 time: {e5 - s5}")

        # Save values for record
        self._stance_action = action
        self._ground_reaction_forces = contact_forces_record
        self._error_q = robot_q - desired_q
        self._error_dq = robot_dq - desired_dq

        return action, contact_forces
