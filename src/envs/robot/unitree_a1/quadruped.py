"""Base class for all robot."""
import ml_collections
import numpy as np
from abc import ABCMeta, abstractmethod
from typing import Any
from typing import Optional
from typing import Tuple

from src.envs.robot.unitree_a1.motors import MotorCommand
from src.envs.robot.unitree_a1.motors import MotorControlMode
from src.envs.robot.unitree_a1.motors import MotorGroup


class QuadrupedRobot(metaclass=ABCMeta):
    """QuadrupedRobot Base (Abstract Class)

    A `Robot` is composed of joints which correspond to motors. For the most
    flexibility, we choose to pass motor objects to the robot when it is
    constructed. This allows for easy config_json-driven instantiation of
    different robot morphologies.

    Motors are passed as a collection of another collection of motors. A
    collection of motors at the lowest level is implemented in a `MotorGroup`.
    This means a robot can support the following configurations:
    1 Motor Robot: [ [ Arm Motor ] ]
    1 MotorGroup Robot: [ [ Arm Motor1, Arm Motor2 ] ]
    2 MotorGroup Robot: [ [ Leg Motor1, Leg Motor2 ],
      [ Arm Motor1, Arm Motor2 ] ]
    """

    def __init__(
            self,
            motors: MotorGroup,
            sensors: Optional[None] = None,
            base_joint_names: Tuple[str, ...] = None,
            foot_joint_names: Tuple[str, ...] = None,
    ) -> None:
        """Constructs a base robot and resets it to the initial states.
        TODO
        """
        # Robot hardware config_json
        self._base_joint_names = base_joint_names
        self._foot_joint_names = foot_joint_names
        self._motor_group = motors
        self._sensors = sensors
        self._num_motors = self._motor_group.num_motors if self._motor_group else 0
        self._motor_torques = None
        self._foot_contact_history = None
        self._step_counter = 0
        self._action_counter = 0

    # Position Interpolation
    @staticmethod
    def joint_linear_interpolation(init_pos, target_pos, rate):
        """Interpolation for the motor joint"""
        rate = max(0.0, min(rate, 1.0))
        p = init_pos * (1 - rate) + target_pos * rate
        return p

    @abstractmethod
    def _apply_action(self):
        raise NotImplementedError()

    @abstractmethod
    def step(self) -> None:
        raise NotImplementedError()

    @abstractmethod
    def _update_contact_history(self):
        raise NotImplementedError()

    @abstractmethod
    def base_position(self):
        raise NotImplementedError()

    # @abstractmethod
    # def base_velocity(self):
    #     raise NotImplementedError()

    @abstractmethod
    def base_orientation_rpy(self):
        raise NotImplementedError()

    @abstractmethod
    def base_orientation_quaternion(self):
        raise NotImplementedError()

    @abstractmethod
    def base_linear_velocity(self):
        raise NotImplementedError()

    @abstractmethod
    def base_angular_velocity(self):
        raise NotImplementedError()

    @abstractmethod
    def base_angular_velocity_in_body_frame(self):
        raise NotImplementedError()

    @abstractmethod
    def foot_contacts(self):
        raise NotImplementedError()

    @abstractmethod
    def foot_contact_history(self):
        raise NotImplementedError()

    @abstractmethod
    def foot_positions_in_body_frame(self):
        raise NotImplementedError()

    @abstractmethod
    def compute_foot_jacobian(self):
        """Compute the Jacobian for a given leg."""
        raise NotImplementedError()

    @abstractmethod
    def get_motor_angles_from_foot_position(self):
        raise NotImplementedError()

    @abstractmethod
    def motor_angles(self):
        """Motor positions"""
        raise NotImplementedError()

    @abstractmethod
    def motor_velocities(self):
        """Motor velocities"""
        raise NotImplementedError()

    @abstractmethod
    def motor_torques(self):
        """Motor torques."""
        raise NotImplementedError()

    @property
    def motor_group(self):
        return self._motor_group

    @abstractmethod
    def map_contact_force_to_joint_torques(self):
        """Maps the foot contact force to the leg joint torques."""
        raise NotImplementedError()

    @abstractmethod
    def control_timestep(self):
        """The frequency for the running controller"""
        raise NotImplementedError()

    @abstractmethod
    def time_since_reset(self):
        raise NotImplementedError()

    @abstractmethod
    def swing_reference_positions(self):
        raise NotImplementedError()

    @property
    def num_legs(self):
        return 4

    @property
    def num_motors(self):
        raise self._num_motors

    @property
    def mpc_body_height(self):
        raise NotImplementedError()

    @property
    def mpc_body_mass(self):
        raise NotImplementedError()

    @property
    def mpc_body_inertia(self):
        raise NotImplementedError()
