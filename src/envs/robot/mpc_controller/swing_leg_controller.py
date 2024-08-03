"""The swing leg controller class."""

import copy
import math
import time
import numpy as np
from absl import logging
from omegaconf import DictConfig
from typing import Any, Mapping, Sequence, Tuple

from src.envs.robot.unitree_a1.motors import MotorCommand
from src.envs.robot.gait_scheduler import gait_scheduler as gait_scheduler_lib


def _gen_parabola(phase: float, start: float, mid: float, end: float) -> float:
    """Gets a point on a parabola y = a*x^2 + b*x + c.

    The Parabola is determined by three points (0, start), (0.5, mid), (1, end) in
    the plane.

    Args:
      phase: Normalized to [0, 1]. A point on the x-axis of the parabola.
      start: The y value at x == 0.
      mid: The y value at x == 0.5.
      end: The y value at x == 1.

    Returns:
      The y value at x == phase.
    """
    mid_phase = 0.5
    delta_1 = mid - start
    delta_2 = end - start
    delta_3 = mid_phase ** 2 - mid_phase
    coef_a = (delta_1 - delta_2 * mid_phase) / delta_3
    coef_b = (delta_2 * mid_phase ** 2 - delta_1) / delta_3
    coef_c = start

    return coef_a * phase ** 2 + coef_b * phase + coef_c


def _gen_swing_foot_trajectory(input_phase: float,
                               start_pos: Sequence[float],
                               end_pos: Sequence[float],
                               foot_lift_height: float) -> Tuple[float]:
    """Generates the swing trajectory using a parabola.

    Args:
      input_phase: the swing/stance phase value between [0, 1].
      start_pos: The foot's position at the beginning of swing cycle.
      end_pos: The foot's desired position at the end of swing cycle.

    Returns:
      The desired foot position at the current phase.
    """
    # We augment the swing speed using the below formula. For the first half of
    # the swing cycle, the swing leg moves faster and finishes 80% of the full
    # swing trajectory. The rest 20% of trajectory takes another half swing
    # cycle. Intuitively, we want to move the swing foot quickly to the target
    # landing location and stay above the ground, in this way the control is more
    # robust to perturbations to the body that may cause the swing foot to drop
    # onto the ground earlier than expected. This is a common practice similar
    # to the MIT cheetah and Marc Raibert's original controllers.

    phase = input_phase
    if input_phase <= 0.5:
        phase = 0.8 * math.sin(input_phase * math.pi)
    else:
        phase = 0.8 + (input_phase - 0.5) * 0.4

    x = (1 - phase) * start_pos[0] + phase * end_pos[0]
    y = (1 - phase) * start_pos[1] + phase * end_pos[1]
    mid = max(end_pos[2], start_pos[2]) + foot_lift_height
    z = _gen_parabola(phase, start_pos[2], mid, end_pos[2])

    # PyType detects the wrong return type here.
    return (x, y, z)  # pytype: disable=bad-return-type


# def cubic_bezier(x0: Sequence[float], x1: Sequence[float],
#                  t: float) -> Sequence[float]:
#   progress = t**3 + 3 * t**2 * (1 - t)
#   return x0 + progress * (x1 - x0)

# def _gen_swing_foot_trajectory(input_phase: float, start_pos: Sequence[float],
#                                end_pos: Sequence[float]) -> Tuple[float]:
#   max_clearance = 0.10
#   mid_z = max(end_pos[2], start_pos[2]) + max_clearance
#   mid_pos = (start_pos + end_pos) / 2
#   mid_pos[2] = mid_z
#   if input_phase < 0.5:
#     t = input_phase * 2
#     foot_pos = cubic_bezier(start_pos, mid_pos, t)
#   else:
#     t = input_phase * 2 - 1
#     foot_pos = cubic_bezier(mid_pos, end_pos, t)
#   return foot_pos


class RaibertSwingLegController:
    """Controls the swing leg position using Raibert's formula.

    For details, please refer to chapter 2 in "Legged robot that balance" by
    Marc Raibert. The key idea is to stabilize the swing foot's location based on
    the CoM moving speed.

    """

    def __init__(self,
                 robot: Any,
                 gait_scheduler: Any,
                 state_estimator: Any,
                 desired_speed: Tuple[float, float],
                 desired_twisting_speed: float,
                 desired_com_height: float,
                 swing_params: DictConfig
                 ):

        """Initializes the class.

        Args:
          robot: A robot instance.
          gait_scheduler: Generates the stance/swing pattern.
          state_estimator: Estimates the CoM speeds.
          desired_speed: Behavior robot. X-Y speed.
          desired_twisting_speed: Behavior control robot.
          desired_com_height: Desired standing height.
          foot_landing_clearance: The foot clearance on the ground at the end of
            the swing cycle.
        """
        self._robot = robot
        self._state_estimator = state_estimator
        self._gait_scheduler = gait_scheduler
        self._last_leg_states = gait_scheduler.desired_leg_states

        # self._swing_config = _load_controller_config(config_path)

        self.desired_speed = np.array((desired_speed[0], desired_speed[1], 0))
        self.desired_twisting_speed = desired_twisting_speed
        self._desired_com_height = desired_com_height
        self._desired_landing_height = np.array(
            (0, 0, desired_com_height - swing_params.foot_landing_clearance))

        self._phase_switch_foot_local_position = None
        self.foot_placement_position = np.zeros(12)  # Scheduled foot placement position
        self.use_raibert_heuristic = swing_params.use_raibert_heuristic
        self._foot_lift_height = swing_params.foot_lift_height
        self._foot_placement_interval = np.asarray(swing_params.foot_placement_interval)
        self._raibert_kp = np.asarray(swing_params.raibert_kp)

        self._swing_action = None

        self.reset(0)

    def reset(self, current_time: float) -> None:
        """Called during the start of a swing cycle.

        Args:
          current_time: The wall time in seconds.
        """
        del current_time

        self._last_leg_states = self._gait_scheduler.desired_leg_states
        self._phase_switch_foot_local_position = \
            self._robot.foot_positions_in_body_frame.copy()

    def update(self, current_time: float) -> None:
        """Called at each control step.
        Args:
          current_time: The wall time in seconds.
        """
        del current_time
        new_leg_states = self._gait_scheduler.desired_leg_states

        # print(f"last_leg_states: {self._last_leg_states}")
        # print(f"phase_switch_foot_local_position: {self._phase_switch_foot_local_position}")

        # Detects phase switch for each leg, so we can remember the feet position at
        # the beginning of the swing phase.
        for leg_id, state in enumerate(new_leg_states):
            if (state == gait_scheduler_lib.LegState.SWING
                    and state != self._last_leg_states[leg_id]):
                self._phase_switch_foot_local_position[leg_id] = (
                    self._robot.foot_positions_in_body_frame[leg_id])

        self._last_leg_states = copy.deepcopy(new_leg_states)

    @property
    def swing_action(self):
        return self._swing_action

    @property
    def foot_lift_height(self):
        return self._foot_lift_height

    @foot_lift_height.setter
    def foot_lift_height(self, foot_lift_height: float) -> None:
        self._foot_lift_height = foot_lift_height

    @property
    def foot_landing_clearance(self):
        return self._desired_com_height - self._desired_landing_height[2]

    @foot_landing_clearance.setter
    def foot_landing_clearance(self, landing_clearance: float) -> None:
        self._desired_landing_height = np.array(
            (0., 0., self._desired_com_height - landing_clearance))

    def get_action(self) -> Mapping[Any, Any]:

        # ------------------------- swing get_action part 1 ------------------------
        # s = time.time()

        com_velocity = self._state_estimator.com_velocity_in_body_frame
        # s1 = time.time()

        com_velocity = np.array((com_velocity[0], com_velocity[1], 0))  # set z-axis velocity to be 0
        # s2 = time.time()
        _, _, yaw_dot = self._robot.base_angular_velocity_in_body_frame
        # s3 = time.time()
        hip_positions = self._robot.swing_reference_positions
        # s4 = time.time()
        all_joint_angles = {}

        # e1 = time.time()
        # print(f"swing get_action part 1 time: {e1 - s}")
        # print(f"part 1-1: {s1 - s}")
        # print(f"part 1-2: {s2 - s1}")
        # print(f"part 1-3: {s3 - s2}")
        # print(f"part 1-4: {s4 - s3}")

        all_joint_angles = {}

        # e1 = time.time()
        # print(f"swing get_action part 1 time: {e1 - s}")

        for leg_id, leg_state in enumerate(self._gait_scheduler.leg_states):

            # print(f"leg_state: {self._gait_scheduler.leg_states}")

            # Skip other leg states which is not SWING
            if leg_state in (gait_scheduler_lib.LegState.STANCE,
                             gait_scheduler_lib.LegState.EARLY_CONTACT,
                             gait_scheduler_lib.LegState.LOSE_CONTACT):
                continue

            # Not consider the body pitch/roll and all calculation is in the body frame.
            # Since the robot body has angular velocity in z-axis, we use v = w * r to
            # calculate the hip velocity in the horizontal plane w.r.t the body frame

            hip_offset = hip_positions[leg_id]
            twisting_vector = np.array((-hip_offset[1], hip_offset[0], 0))

            # v_hip = v_linear + v_rot = v_linear + v_angular * distance
            hip_horizontal_velocity = com_velocity + yaw_dot * twisting_vector

            target_hip_horizontal_velocity = (
                    self.desired_speed + self.desired_twisting_speed * twisting_vector)

            # If not generated the foot placement position
            if self.use_raibert_heuristic or (not self.foot_placement_position.any()):

                # Use raibert heuristic to determine foot landing position in hip_ground frame
                foot_horizontal_landing_position = (
                        hip_horizontal_velocity *
                        self._gait_scheduler.stance_duration[leg_id] / 2
                        - self._raibert_kp * (target_hip_horizontal_velocity - hip_horizontal_velocity)
                )

                max_foot_landing = self._foot_placement_interval
                min_foot_landing = -1 * self._foot_placement_interval

                # Use clip to restrict the target position
                foot_horizontal_landing_position = np.clip(foot_horizontal_landing_position,
                                                           min_foot_landing, max_foot_landing)

                # Calculated foot landing position in body frame
                foot_landing_position = (foot_horizontal_landing_position
                                         - self._desired_landing_height
                                         + np.array((hip_offset[0], hip_offset[1], 0)))

            else:
                foot_landing_position = (self.foot_placement_position[leg_id]
                                         - self._desired_landing_height
                                         + np.array((hip_offset[0], hip_offset[1], 0)))

            # Compute target position compensation due to slope
            gravity_projection_vector = self._state_estimator.gravity_projection_vector

            multiplier = -self._desired_landing_height[2] / gravity_projection_vector[2]
            foot_landing_position[:2] += gravity_projection_vector[:2] * multiplier
            # logging.info("Compsenation: {}".format(gravity_projection_vector[:2] *
            #                                        multiplier))

            # foot target position in body frame
            foot_target_position = _gen_swing_foot_trajectory(
                input_phase=self._gait_scheduler.normalized_phase[leg_id],
                start_pos=self._phase_switch_foot_local_position[leg_id],
                end_pos=foot_landing_position,
                foot_lift_height=self._foot_lift_height)

            # print(f"foot target position: {foot_target_position}")
            # print(f"phase is: {self._gait_scheduler.normalized_phase[leg_id]}")
            # print(f"start_pos: {self._phase_switch_foot_local_position[leg_id]}")

            joint_ids, joint_angles = (
                self._robot.get_motor_angles_from_foot_position(leg_id, foot_target_position))

            # Update the stored joint angles as needed.
            for joint_id, joint_angle in zip(joint_ids, joint_angles):
                all_joint_angles[joint_id] = (joint_angle, leg_id)

        # e2 = time.time()
        # print(f"swing get_action part 2 time: {e2 - e1}")

        action = {}
        kps = self._robot.motor_group.kps
        kds = self._robot.motor_group.kds

        for joint_id, joint_angle_leg_id in all_joint_angles.items():
            leg_id = joint_angle_leg_id[1]
            action[joint_id] = MotorCommand(desired_position=joint_angle_leg_id[0],
                                            kp=kps[joint_id],
                                            desired_velocity=0,
                                            kd=kds[joint_id],
                                            desired_torque=0)
            # if self._gait_scheduler.desired_leg_states[
            #     leg_id] == gait_scheduler_lib.LegState.SWING:
            #     # This is a hybrid action for PD control.
            #     action[joint_id] = (joint_angle_leg_id[0], kps[joint_id], 0,
            #                         kds[joint_id], 0)

        # e3 = time.time()
        # print(f"swing get_action part 3 time: {e3 - e2}")
        return action
