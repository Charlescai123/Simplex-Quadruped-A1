"""State estimator."""
import time
import numpy as np
import pybullet as p
from typing import Any, Sequence

from src.envs.robot.gait_scheduler import gait_scheduler as gait_generator_lib
from src.envs.robot.state_estimator import moving_window_filter

# from src.envs.robot.unitree_a1 import a1_robot

_DEFAULT_WINDOW_SIZE = 20


class COMVelocityEstimator(object):
    """Estimate the CoM velocity using on board sensors.

    Requires knowledge about the base velocity in world frame, which for example
    can be obtained from a MoCap system. This estimator will filter out the high
    frequency noises in the velocity so the results can be used with controllers
    reliably.

    """

    def __init__(self,
                 robot: Any,
                 velocity_window_size: int = _DEFAULT_WINDOW_SIZE,
                 ground_normal_window_size: int = _DEFAULT_WINDOW_SIZE):

        from src.envs.robot.unitree_a1 import a1_robot

        self._robot = robot
        self._com_velocity_in_body_frame = None
        self._com_velocity_in_world_frame = None
        self._ground_normal_filter = None
        self._velocity_filter = None
        self._velocity_window_size = velocity_window_size
        self._ground_normal_window_size = ground_normal_window_size
        self._last_desired_leg_states = [gait_generator_lib.LegState.STANCE] * 4
        self._swing_force_history = [[], [], [], []]
        self._ground_normal = np.array([0., 0., 1.])
        self.reset(0)

    def reset(self, current_time):
        del current_time
        # We use a moving window filter to reduce the noise in velocity estimation.
        self._velocity_filter = moving_window_filter.MovingWindowFilter(
            window_size=self._velocity_window_size)
        self._ground_normal_filter = moving_window_filter.MovingWindowFilter(
            window_size=self._ground_normal_window_size)

        self._com_velocity_in_world_frame = np.array((0, 0, 0))
        self._com_velocity_in_body_frame = np.array((0, 0, 0))

    def _compute_ground_normal(self, contact_foot_positions):
        """Computes the surface orientation in robot frame based on foot positions.
        Solves a least-squares problem, see the following paper for details:
        https://ieeexplore.ieee.org/document/7354099
        """
        contact_foot_positions = np.array(contact_foot_positions)
        normal_vec = np.linalg.lstsq(contact_foot_positions, np.ones(4))[0]
        normal_vec /= np.linalg.norm(normal_vec)
        if normal_vec[2] < 0:
            normal_vec = -normal_vec
        return normal_vec

    def update(self, desired_leg_states, is_real_robot=False):
        # print(f"desired_leg_states: {desired_leg_states}")

        # Update foot force calibration
        if self._robot.a1_config.model == 'a1_robot':
        # if isinstance(self._robot, a1_robot.A1Robot):
            print("An real robot instance")

            for leg_id in range(4):
                if desired_leg_states[leg_id] == gait_generator_lib.LegState.SWING:
                    self._swing_force_history[leg_id].append(
                        self._robot.foot_forces[leg_id])
                if (desired_leg_states[leg_id] == gait_generator_lib.LegState.STANCE
                        and self._last_desired_leg_states[leg_id] == gait_generator_lib.LegState.SWING):
                    # Transition from swing to stance, update sensor calibration
                    avg_swing_force = np.mean(self._swing_force_history[leg_id])
                    self._robot.update_foot_contact_force_threshold(
                        leg_id, avg_swing_force + 10)
                    self._swing_force_history[leg_id] = []
            self._last_desired_leg_states = desired_leg_states

        # Update CoM velocity in body frame
        velocity = np.array(self._robot.base_linear_velocity)
        self._com_velocity_in_world_frame = self._velocity_filter.calculate_average(
            velocity)

        base_orientation = self._robot.base_orientation_quaternion
        _, inverse_rotation = self._robot.pybullet_client.invertTransform(
            (0, 0, 0), base_orientation)

        self._com_velocity_in_body_frame, _ = (
            self._robot.pybullet_client.multiplyTransforms(
                (0, 0, 0), inverse_rotation, self._com_velocity_in_world_frame,
                (0, 0, 0, 1)))

        ground_normal_vector = self._compute_ground_normal(
            self._robot.foot_contact_history)

        self._ground_normal = self._ground_normal_filter.calculate_average(
            ground_normal_vector)
        self._ground_normal /= np.linalg.norm(self._ground_normal)

    @property
    def com_position_in_ground_frame(self):
        foot_contacts = self._robot.foot_contacts.copy()

        if np.sum(foot_contacts) == 0:  # No feet on the ground
            return np.array((0, 0, self._robot.mpc_body_height))
        else:
            foot_positions_robot_frame = self._robot.foot_positions_in_body_frame

            ground_orientation_matrix_robot_frame = p.getMatrixFromQuaternion(
                self.ground_orientation_in_robot_frame)

            # Reshape
            ground_orientation_matrix_robot_frame = np.array(
                ground_orientation_matrix_robot_frame).reshape((3, 3))

            foot_positions_ground_frame = (foot_positions_robot_frame.dot(
                ground_orientation_matrix_robot_frame.T))

            foot_heights = -foot_positions_ground_frame[:, 2]

            # print(f"foot_positions_robot_frame: {foot_positions_robot_frame}")
            # print(f"foot_positions_ground_frame: {foot_positions_ground_frame}")

            return np.array((
                0,
                0,
                np.sum(foot_heights * foot_contacts) / np.sum(foot_contacts),
            ))

    @property
    def com_orientation_quaternion_in_ground_frame(self):
        _, orientation = p.invertTransform([0., 0., 0.],
                                           self.ground_orientation_in_robot_frame)
        return np.array(orientation)

    @property
    def com_velocity_in_ground_frame(self):
        _, world_orientation_ground_frame = p.invertTransform(
            [0., 0., 0.], self.ground_orientation_in_world_frame)
        return np.array(
            p.multiplyTransforms([0., 0., 0.], world_orientation_ground_frame,
                                 self._com_velocity_in_world_frame,
                                 [0., 0., 0., 1.])[0])

    @property
    def com_rpy_rate_in_ground_frame(self):
        com_quat_world_frame = p.getQuaternionFromEuler(self._robot.base_rpy_rate)
        _, world_orientation_ground_frame = p.invertTransform(
            [0., 0., 0.], self.ground_orientation_world_frame)
        _, com_quat_ground_frame = p.multiplyTransforms(
            [0., 0., 0.], world_orientation_ground_frame, [0., 0., 0.],
            com_quat_world_frame)
        return np.array(p.getEulerFromQuaternion(com_quat_ground_frame))

    @property
    def com_velocity_in_body_frame(self) -> Sequence[float]:
        """The base velocity projected in the body aligned inertial frame.

        The body aligned frame is an inertia frame that coincides with the body
        frame, but has a zero relative velocity/angular velocity to the world frame.

        Returns:
          The com velocity in body aligned frame.
        """
        return self._com_velocity_in_body_frame

    @property
    def ground_normal(self):
        return self._ground_normal

    @property
    def gravity_projection_vector(self):
        _, world_orientation_ground_frame = p.invertTransform(
            [0., 0., 0.], self.ground_orientation_in_world_frame)
        return np.array(
            p.multiplyTransforms([0., 0., 0.], world_orientation_ground_frame,
                                 [0., 0., 1.], [0., 0., 0., 1.])[0])

    @property
    def ground_orientation_in_robot_frame(self):
        normal_vec = self.ground_normal
        axis = np.array([-normal_vec[1], normal_vec[0], 0])
        axis /= np.linalg.norm(axis)
        angle = np.arccos(normal_vec[2])
        return np.array(p.getQuaternionFromAxisAngle(axis, angle))

    @property
    def ground_orientation_in_world_frame(self) -> Sequence[float]:
        return np.array(
            p.multiplyTransforms([0., 0., 0.], self._robot.base_orientation_quaternion,
                                 [0., 0., 0.],
                                 self.ground_orientation_in_robot_frame)[1])

    # @property
    def estimate_robot_x_y_z(self):
        contacts = np.array(
            [(leg_state in (gait_generator_lib.LegState.STANCE,
                            gait_generator_lib.LegState.EARLY_CONTACT))
             for leg_state in self._robot.controller.gait_scheduler.desired_leg_states],
            dtype=np.float64)
        # foot_positions = self._robot.GetFootPositionsInBaseFrame()  # this is relative positions of the leg to the base
        foot_positions = self._robot.foot_positions_in_body_frame  # this is relative positions of the leg to the base

        # base_orientation = self._robot.GetBaseOrientation()
        base_orientation_quat = self._robot.base_orientation_quaternion
        rot_mat = self._robot.pybullet_client.getMatrixFromQuaternion(
            base_orientation_quat)
        rot_mat = np.array(rot_mat).reshape((3, 3))
        foot_positions_world_frame = (rot_mat.dot(foot_positions.T)).T

        # pylint: disable=unsubscriptable-object
        x = contacts * (foot_positions_world_frame[:, 0])
        y = contacts * (foot_positions_world_frame[:, 1])
        z = contacts * (-foot_positions_world_frame[:, 2])

        # print(f"foot_positions: {foot_positions}")
        # print(f"base_orientation_quat: {base_orientation_quat}")
        # print(f"rot_mat: {rot_mat}")
        # print(f"foot_positions_world_frame: {foot_positions_world_frame}")
        # print(f"x: {x}")
        # print(f"y: {y}")
        # print(f"z: {z}")
        # print(f"contacts: {contacts}")
        # print(f"self._robot.gait_scheduler.desired_leg_states: {self._robot.gait_scheduler.desired_leg_states}")

        return np.sum(x) / np.sum(contacts), np.sum(y) / np.sum(contacts), np.sum(z) / np.sum(contacts)
