"""Real A1 robot class."""
import ml_collections
import numpy as np
import robot_interface
import time
from typing import Any
from typing import Tuple

from config.locomotion.robots.pose import Pose
from config.locomotion.robots.a1_robot_params import A1RobotParams
from config.a1_real_params import A1RealParams
from config.locomotion.robots.motor_params import MotorGroupParams
from config.locomotion.controllers.swing_params import SwingControllerParams
from config.locomotion.controllers.stance_params import StanceControllerParams
from agents.phydrl.policies.ddpg import DDPGAgent
from locomotion.state_estimator import a1_robot_state_estimator
from locomotion.robots.motors import MotorControlMode
from locomotion.robots.motors import MotorCommand
from locomotion.robots import a1

np.set_printoptions(suppress=True)


class A1Robot(a1.A1):
    """Class for interfacing with A1 hardware."""

    def __init__(
            self,
            pybullet_client: Any = None,
            ddpg_agent: DDPGAgent = None,
            mat_engine: Any = None,
            a1_robot_params: A1RobotParams = None,
            motor_params: MotorGroupParams = None,
            swing_params: SwingControllerParams = None,
            stance_params: StanceControllerParams = None,
            logdir='./logs'
    ) -> None:

        self._params = a1_robot_params

        self._raw_state = robot_interface.LowState()
        self._contact_force_threshold = np.zeros(4)

        # Send an initial zero command in order to receive state information.
        self._robot_interface = robot_interface.RobotInterface(0xff)
        self._state_estimator = a1_robot_state_estimator.A1RobotStateEstimator(self)
        self._last_reset_time = time.time()

        super(A1Robot, self).__init__(
            pybullet_client=pybullet_client,
            ddpg_agent=ddpg_agent,
            mat_engine=mat_engine,
            a1_params=a1_robot_params,
            motor_params=motor_params,
            swing_params=swing_params,
            stance_params=stance_params,
            logdir=logdir
        )

    def _receive_observation(self) -> None:
        """Receives observation from robot and saves the state.

    Note that the returned state from robot's receive_observation() function
    is mutable. So we need to copy the value out.
    """
        self._raw_state = self._robot_interface.receive_observation()
        # self._state_safe_check(self._raw_state)  # State safety check

    def step(self,
             action: MotorCommand,
             motor_control_mode: MotorControlMode = None) -> None:

        self._step_counter += 1
        for _ in range(self._params.action_repeat):
            self._apply_action(action, motor_control_mode)
            self._action_counter += 1

            self._receive_observation()
            self._state_estimator.update(self._raw_state)
            self._update_contact_history()
        # time.sleep(self.control_timestep)

    def _apply_action(self,
                      action: MotorCommand,
                      motor_control_mode: MotorControlMode = None) -> None:
        """Clips and then apply the motor commands using the motor phydrl.
    Args:
      action: np.array. Can be motor angles, torques, or hybrid commands.
      motor_control_mode: A MotorControlMode enum.
    """
        if motor_control_mode is None:
            motor_control_mode = self._motor_group.motor_control_mode
        command = np.zeros(60, dtype=np.float32)

        print(f"The motor control is in {motor_control_mode}")
        # print(f"action is: \n{action}")
        # motor action safety check
        # self._motor_action_safe_check(action, motor_control_mode)

        # Position Mode
        if motor_control_mode == MotorControlMode.POSITION:
            for motor_id in range(self.num_motors):
                command[motor_id * 5] = action.desired_position[motor_id]
                command[motor_id * 5 + 1] = action.kp[motor_id]
                command[motor_id * 5 + 3] = action.kd[motor_id]

        # Torque Mode
        elif motor_control_mode == MotorControlMode.TORQUE:
            for motor_id in range(self.num_motors):
                command[motor_id * 5 + 4] = action.desired_torque[motor_id]

        # Hybrid Mode
        elif motor_control_mode == MotorControlMode.HYBRID:
            command[0::5] = action.desired_position
            command[1::5] = action.kp
            command[2::5] = action.desired_velocity
            command[3::5] = action.kd
            command[4::5] = action.desired_torque

        # Unknown Mode
        else:
            raise ValueError('Unknown motor control mode for A1 robot: {}.'.format(
                motor_control_mode))

        # print generated motor actions
        # self._motor_action_print(command)

        # Constrain the torques
        print("Clipping the Generated torques")
        print(f"The command is: {command}")
        # applied_torque = self._clip_torques(desired_torque, current_velocity)

        self._robot_interface.send_command(command)

    def reset(self, hard_reset: bool = False, reset_time=1.5):
        """Reset the robot to default motor angles."""
        # super(A1Robot, self).reset(hard_reset, num_reset_steps=0)
        # self.motor_group.init_positions = Pose.STANDING_POSE
        self.motor_group.init_positions = self._params.motor_init_target_position
        # print(self.motor_angles)
        # print(self.motor_group.init_positions)

        for _ in range(10):
            self._robot_interface.send_command(np.zeros(60, dtype=np.float32))
            time.sleep(0.001)
            self._receive_observation()
        print("About to reset the robot.")

        initial_motor_position = self.motor_angles
        end_motor_position = self.motor_group.init_positions

        # Stand up in 1.5 seconds, and fix the standing pose afterward.
        standup_time = min(reset_time, 1.)
        stand_foot_forces = []
        for t in np.arange(0, reset_time, self.control_timestep):
            blend_ratio = min(t / standup_time, 1)
            desired_motor_position = blend_ratio * end_motor_position + (
                    1 - blend_ratio) * initial_motor_position
            action = MotorCommand(desired_position=desired_motor_position,
                                  kp=self.motor_group.kps,
                                  desired_velocity=np.zeros(self.num_motors),
                                  kd=self.motor_group.kds)
            self.step(action, MotorControlMode.POSITION)
            time.sleep(self.control_timestep)
            if t > standup_time:
                stand_foot_forces.append(self.foot_forces)

        # Calibrate foot force sensors
        stand_foot_forces = np.mean(stand_foot_forces, axis=0)
        for leg_id in range(4):
            self.update_foot_contact_force_threshold(leg_id,
                                                     stand_foot_forces[leg_id] * 0.8)

        self._last_reset_time = time.time()
        self._state_estimator.reset()

    # @property
    # def sim_conf(self):
    #     return self._sim_conf

    # @property
    # def robot_config(self):
    #     return self._robot_config

    @property
    def robot_params(self):
        return self._params

    @property
    def foot_forces(self):
        return np.array(self._raw_state.footForce)

    def update_foot_contact_force_threshold(self, leg_id, threshold):
        self._contact_force_threshold[leg_id] = threshold

    @property
    def foot_contacts(self):
        return np.array(self._raw_state.footForce) > self._contact_force_threshold

    @property
    def base_position(self):
        contacts = np.array(self.foot_contacts)
        if not np.sum(contacts):
            return np.array([0., 0., self.mpc_body_height])
        foot_positions_base_frame = self.foot_positions_in_body_frame
        foot_heights = -foot_positions_base_frame[:, 2]
        base_height = np.sum(foot_heights * contacts) / np.sum(contacts)
        return np.array([0., 0., base_height])

    @property
    def base_linear_velocity(self):
        return self._state_estimator.estimated_velocity.copy()

    @property
    def base_orientation_rpy(self):
        return np.array(self._raw_state.imu.rpy)

    @property
    def base_orientation_quaternion(self):
        q = self._raw_state.imu.quaternion
        return np.array([q[1], q[2], q[3], q[0]])

    @property
    def motor_angles(self):
        return np.array([motor.q for motor in self._raw_state.motorState[:12]])

    @property
    def motor_velocities(self):
        return np.array([motor.dq for motor in self._raw_state.motorState[:12]])

    @property
    def motor_torques(self):
        return np.array(
            [motor.tauEst for motor in self._raw_state.motorState[:12]])

    @property
    def base_angular_velocity_in_body_frame(self):
        return np.array(self._raw_state.imu.gyroscope)

    @property
    def foot_positions_in_body_frame(self):
        """Use analytical FK/IK/Jacobian"""
        # return self._foot_positions_in_hip_frame + HIP_OFFSETS
        return self._foot_positions_in_hip_frame + self._params.hip_offset

    @property
    def time_since_reset(self):
        return time.time() - self._last_reset_time

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
            foot_local_position - self._params.hip_offset[leg_id],
            l_hip_sign=(-1) ** (leg_id + 1))
        return joint_position_idxs, joint_angles.tolist()

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

    # Checkout the robot state to prevent zero scenarios
    def _state_safe_check(self, state):
        # Check IMU States
        q = state.imu.quaternion
        rpy = state.imu.rpy

        # Check Motor Angles States
        motor_angles = [motor.q for motor in state.motorState[:12]]

        try:
            assert 0. not in q and 0. not in rpy
            assert 0. not in motor_angles
        except:
            print("Exception data in received observation")
            raise RuntimeError

    # print the generated motor hybrid actions
    def _motor_action_print(self, actions):
        print("Generated Actions for Motors are:")
        assert len(actions) == 60
        cnt = 0
        for i in range(len(LEGS)):
            leg = LEGS[i]
            print(f"{leg} motors:")
            for j in range(len(LEG_MOTORS)):
                motor_idx = i * 3 + j
                print(f"{LEG_MOTORS[j]}".rjust(5) + f" motor [{motor_idx}]:", end=' ')
                for _ in range(5):
                    value = round(actions[cnt], 3)
                    print(f"{value}", end=' ', sep='\t')
                    cnt += 1
                print()
            print()
        assert cnt == 60

    # Motor Safety Examination
    def _motor_action_safe_check(self, action, motor_control_mode=None):
        if motor_control_mode is None:
            motor_control_mode = self.motor_group.motor_control_mode

        print("Conducting Safe Check for the Motor's Action...")

        # Position Mode
        # if motor_control_mode == MotorControlMode.POSITION:
        #     for motor_id in range(self.num_motors):
        #         desired_p = action.desired_position[motor_id]
        #         if ((desired_p > self.motor_group.motor_max_positions[motor_id]) or
        #                 (desired_p < self.motor_group.motor_min_positions[motor_id])):
        #             print(f"The position action for the motor [{motor_id}] is: {desired_p}, "
        #                   f"which is out of range!!!")
        #             print("For safe consideration, cutting off the program now.")
        #             exit(0)

        # Torque Mode
        if motor_control_mode == MotorControlMode.TORQUE:
            for motor_id in range(self.num_motors):
                desired_torque = action.desired_torque[motor_id]
                if ((desired_torque > self.motor_group.motor_max_torques[motor_id]) or
                        (desired_torque < self.motor_group.motor_min_torques[motor_id])):
                    print(f"The torque action for the motor [{motor_id}] is: {desired_torque}, "
                          f"which is out of range!!!")
                    print("For safe consideration, cutting off the program now.")
                    exit(0)


        # Hybrid Mode
        elif motor_control_mode == MotorControlMode.HYBRID:
            desired_pos = action.desired_position
            kp = action.kp
            desired_vel = action.desired_velocity
            kd = action.kd
            torque = action.desired_torque
            desired_torque = (kp * (desired_pos - self.motor_angles)
                              + kd * (desired_vel - self.motor_velocities)
                              + torque)

            for motor_id in range(self.num_motors):
                applied_torque = desired_torque[motor_id]
                if (applied_torque > self.motor_group.motor_max_torques[motor_id]) or (
                        applied_torque < self.motor_group.motor_min_torques[motor_id]):
                    print(f"The hybrid action for the motor [{motor_id}] is: {applied_torque}, "
                          f"which is out of range!!!")
                    print("For safe consideration, cutting off the program now.")
                    exit(0)

        print("motor action safety check is done.")
