"""
A gait scheduler that computes leg states based on phases.
"""
from absl import logging
import numpy as np
from typing import Any, Sequence
import copy

from src.envs.robot.gait_scheduler import gait_scheduler

LegState = gait_scheduler.LegState

_INIT_LEG_STATES = (
    gait_scheduler.LegState.SWING,
    gait_scheduler.LegState.STANCE,
    gait_scheduler.LegState.STANCE,
    gait_scheduler.LegState.SWING,
)


class OffsetGaitScheduler(gait_scheduler.GaitScheduler):
    """Phase-variable based gait generator."""

    def __init__(self,
                 robot: Any,
                 init_phase: Sequence[float] = np.zeros(4),
                 gait_parameters: Sequence[float] = [2., np.pi, np.pi, 0, 0.4],
                 early_touchdown_phase_threshold: float = 0.5,
                 lose_contact_phase_threshold: float = 0.1):
        """Initializes the gait generator."""
        self.robot = robot
        self.stance_duration = None
        self.last_action_time = None
        self.steps_since_reset = None
        # del init_phase  # unused
        # self.init_phase = np.array([0, gait_params[1], gait_params[2], gait_params[3]])
        self.init_phase = copy.deepcopy(init_phase)
        self.current_phase = None

        self._early_touchdown_phase_threshold = early_touchdown_phase_threshold
        self._lose_contact_phase_threshold = lose_contact_phase_threshold
        self.gait_params = tuple(gait_parameters)  # [freq, theta1, theta2, theta3, theta_swing_cutoff]
        self.prev_frame_robot_time = 0
        self.swing_cutoff = None
        self.reset()

    def reset(self):
        # self.current_phase = np.zeros(4)        # Enter swing immediately after start
        self.current_phase = self.init_phase
        # print(f"current_phase: {self.current_phase}")
        # self.current_phase = np.array([0, 1.8 * np.pi, 1.8 * np.pi, 0])

        self.steps_since_reset = 0
        self.prev_frame_robot_time = self.robot.time_since_reset

        self.last_action_time = self.robot.time_since_reset
        # self.swing_cutoff = 2 * np.pi * 0.5
        self.swing_cutoff = 2 * np.pi * self.gait_params[4]

    def update(self):
        # Calculate the amount of time passed
        current_robot_time = self.robot.time_since_reset
        frame_duration = self.robot.time_since_reset - self.prev_frame_robot_time
        self.prev_frame_robot_time = current_robot_time

        # Propagate phase for front-right leg
        self.current_phase[0] += 2 * np.pi * frame_duration * self.gait_params[0]

        # Offset for remaining legs
        self.current_phase[1:4] = self.current_phase[0] + self.gait_params[1:4]
        # self.swing_cutoff = 2 * np.pi * self.gait_params[4]
        self.stance_duration = 1 / self.gait_params[0] * (1 - self.gait_params[4]) * np.ones(4)

        # print("----------------------------------------------------------")
        # print(f"gait_params: {self.gait_params}")
        # print(f"desired_leg_states: {self.desired_leg_states}")
        # print(f"stance_duration: {self.stance_duration}")
        # print(f"swing_cutoff: {self.swing_cutoff}")
        # print(f"current_phase: {self.current_phase}")
        # print("----------------------------------------------------------")


    def get_estimated_contact_states(self, num_steps, dt):
        current_phase = self.current_phase.copy()
        future_phases = np.repeat(np.arange(num_steps)[:, None], 4,
                                  axis=-1) * 2 * np.pi * self.gait_params[0] * dt
        all_phases = np.fmod(current_phase + future_phases + 2 * np.pi, 2 * np.pi)
        ans = np.where(all_phases < self.swing_cutoff, False, True)
        return ans

    def get_observation(self):
        return np.concatenate(
            (np.cos(self.normalized_phase * np.pi),
             np.sin(self.normalized_phase * np.pi),
             np.where(np.fmod(self.current_phase, 2 * np.pi) < self.swing_cutoff, 0, 1),
             np.where(np.fmod(self.current_phase, 2 * np.pi) >= self.swing_cutoff, 0, 1)
             )
        )

    @property
    def desired_leg_states(self):
        # !!!!!!!!!!!!!!!!!!!!!!!!!! warning !!!!!!!!!!!!!!!!!!!!!!!!
        # np.fmod cannot really wrap to [0, 2*pi] for the negatives
        # wrapped_phase = np.fmod(self.current_phase + 2 * np.pi, 2 * np.pi)

        # Wrap phase to [0, 2 * pi]
        wrapped_phase = np.mod(self.current_phase + 2 * np.pi, 2 * np.pi)
        # print(f"wrapped_phase: {wrapped_phase}")

        # desired_leg_states = np.array([
        #     LegState.SWING if phase < self.swing_cutoff else LegState.STANCE
        #     for phase in wrapped_phase
        # ])
        # print(f"desired_leg_states: {desired_leg_states}")
        # import time
        # time.sleep(1)
        desired_states = np.array([
            LegState.SWING if phase < self.swing_cutoff else LegState.STANCE
            for phase in wrapped_phase
        ])
        assert np.count_nonzero(desired_states == LegState.SWING) <= 2, "The robot leg state is abnormal"
        return desired_states

    @property
    def normalized_phase(self):
        # Wrap phase to [0, 2 * pi]
        wrapped_phase = np.fmod(self.current_phase + 2 * np.pi, 2 * np.pi)
        phase_in_swing_ratio = wrapped_phase / self.swing_cutoff
        phase_in_stance_ratio = (wrapped_phase - self.swing_cutoff) / (2 * np.pi - self.swing_cutoff)

        return np.where(wrapped_phase < self.swing_cutoff,
                        phase_in_swing_ratio,
                        phase_in_stance_ratio)

    @property
    def leg_states(self):

        _leg_states = self.desired_leg_states.copy()
        contact_states = self.robot.foot_contacts

        for leg_id in range(self.robot.num_legs):

            # Detect lost contact status
            if (_leg_states[leg_id] == gait_scheduler.LegState.STANCE
                    and not contact_states[leg_id] and self.normalized_phase[leg_id]
                    > self._lose_contact_phase_threshold):
                logging.info("lost contact detected.")
                _leg_states[leg_id] = gait_scheduler.LegState.LOSE_CONTACT

            # Detect early touch down status
            if (_leg_states[leg_id] == gait_scheduler.LegState.SWING
                    and contact_states[leg_id] and self.normalized_phase[leg_id]
                    > self._early_touchdown_phase_threshold):
                logging.info("early touch down detected.")
                _leg_states[leg_id] = gait_scheduler.LegState.EARLY_CONTACT
                # print(f"leg_id: {leg_id}")
                # print(self.normalized_phase[leg_id])

        # print(f"...leg_states: {_leg_states}")
        return _leg_states
