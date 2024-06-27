"""A1 Trainer PhyDRL"""
import scipy.interpolate
import numpy as np
import pybullet_data
from pybullet_utils import bullet_client
import pybullet  # pytype:disable=import-error
import math
import time

from config.locomotion.robots.a1_params import A1Params
from config.locomotion.robots.motor_params import MotorGroupParams
from config.locomotion.controllers.swing_params import SwingControllerParams
from config.locomotion.controllers.stance_params import StanceControllerParams
from config.phydrl.env_params import TrainerEnvParams
from locomotion.robots import a1
from locomotion.robots import a1_robot_phydrl


class A1TrainerEnv:
    def __init__(self, params: TrainerEnvParams):
        self.previous_tracking_error = None
        self._params = params
        # print("+++++++++++++++++++++++++++++++++++++++++++++")
        # print("+++++++++++++++++++++++++++++++++++++++++++++")
        # print("+++++++++++++++++++++++++++++++++++++++++++++")
        # print("+++++++++++++++++++++++++++++++++++++++++++++")
        # print("+++++++++++++++++++++++++++++++++++++++++++++")
        # print("+++++++++++++++++++++++++++++++++++++++++++++")
        # print(self._params.a1_params.init_gait_phase)
        # print(self._params.a1_params.gait_params)

        if params.show_gui:
            self.p = bullet_client.BulletClient(connection_mode=pybullet.GUI)
        else:
            self.p = bullet_client.BulletClient(connection_mode=pybullet.DIRECT)

        self.robot = None
        self.mpc_controller = None

        # command_function = _generate_example_linear_angular_speed # higher level goal
        self.states_observations_dim = 12
        self.action_dim = 6

        self.termination = None
        self.states = None
        self.observation = None
        # self.brv = 1  # velocity reference
        self.target_lin_speed = [params.ref_vx, 0, 0]
        self.target_ang_speed = 0.0

        self.diff_q = None
        self.diff_dq = None
        self.current_step = 0
        # self.previous_tracking_error = None

        # reference set point
        self.ref_vx = params.ref_vx
        self.ref_px = params.ref_px
        self.ref_pz = params.ref_pz
        self.set_point = np.array(
            [self.ref_px, 0, self.ref_pz,
             0., 0., 0.,
             self.ref_vx, 0., 0.,
             0., 0., 0.]
        )

        self.fall_threshold = params.fall_threshold

        self.reset(step=0)

    def random_reset(self):
        pass

    def reset(self, step, reset_status=None):
        self.p.resetSimulation()
        self.p.setPhysicsEngineParameter(numSolverIterations=30)
        self.p.setTimeStep(0.001)
        self.p.setGravity(0, 0, -9.8)
        self.p.setPhysicsEngineParameter(enableConeFriction=0)
        # plane = self.p.loadURDF("plane.urdf")

        plane = self.p.loadURDF("envs/meshes/plane.urdf")

        if self._params.use_real_urdf:  # whether to use a more realistic urdf file for training
            self.p.setAdditionalSearchPath("envs/sim_envs_v2")
        else:
            self.p.setAdditionalSearchPath(pybullet_data.getDataPath())

        # self.p.changeDynamics(plane, -1, lateralFriction=0.44)  # change friction from higher to lower 0.44,
        # self.p.changeDynamics(plane, -1, lateralFriction=1.0)  # change friction from higher to lower
        self.p.changeDynamics(plane, -1, lateralFriction=self._params.friction)  # change friction from higher to lower

        if self._params.if_record_video:
            self.p.startStateLogging(self.p.STATE_LOGGING_VIDEO_MP4, f"{step}_record.mp4")

        # self.robot = a1.A1(
        #     pybullet_client=self.p,
        #     a1_params=self._params.a1_params,
        #     motor_params=self._params.motor_params,
        #     swing_params=self._params.swing_params,
        #     stance_params=self._params.stance_params
        # )

        self.robot = a1_robot_phydrl.A1RobotPhyDRL(
            pybullet_client=self.p,
            a1_params=self._params.a1_params,
            motor_params=self._params.motor_params,
            swing_params=self._params.swing_params,
            stance_params=self._params.stance_params
        )

        # pp = self.p.getPhysicsEngineParameters()
        # print(f"pp is: {pp}")
        # import time
        # time.sleep(123)

        if self._params.if_add_terrain:
            self.add_terrain()

        self.add_lane()

        self.mpc_controller = self.robot.controller  # WBC controller
        # self.mpc_controller.start_thread()           # Start controller

        # self.mpc_control = _setup_controller(self.robot)  # MPC controller for low-level control
        # self.mpc_control.reset()

        self.states = self.get_state()
        # print(f"self.states: {self.states}")
        self.previous_tracking_error = self.get_tracking_error()
        # print(f"tracking error: {self.previous_tracking_error}")
        self.observation, self.termination, _ = self.get_observations(self.states)
        self.current_step = 0

    def get_observations(self, state):
        observation = []  # 16 dims
        roll, pitch, _ = state['base_rpy']

        termination = False
        abort = False

        angle_threshold = 30 * (math.pi / 180)

        # observation of root orientation,  3
        robot_orientation = state['base_rpy']
        observation.extend(robot_orientation)

        # root angular velocity 3
        robot_angular_velocity = state['base_rpy_rate']
        observation.extend(robot_angular_velocity)

        # linear_velocity 3
        robot_linear_velocity = state['base_linear_vel']
        observation.extend(robot_linear_velocity)

        # # motion_angle 12
        # motor_angle = state['motor_angles']
        # observation.extend(motor_angle)
        #
        # # motion angle rate 12
        # motor_angle_rate = state['motor_vels']
        # observation.extend(motor_angle_rate)

        # foot_contact 4
        foot_contact = state['contacts']
        observation.extend(foot_contact)

        # velocity in body frame 3
        velocity_in_body_frame = state["base_vels_body_frame"]
        observation.extend(velocity_in_body_frame)

        com_fall = self.mpc_controller.state_estimator.estimate_robot_x_y_z()

        # if abs(roll) > angle_threshold or abs(pitch) > angle_threshold:

        if abs(com_fall[2]) < self.fall_threshold:
            print("Fall: height:", com_fall[2])
            termination = True

        if math.isnan(float(robot_linear_velocity[0])):
            abort = True
            termination = True  # new add
            print("ABORT_DUE_TO_SIMULATION_ERROR")

        return observation, termination, abort

    def get_state(self):

        states = dict(timestamp=self.robot.time_since_reset,
                      base_rpy=self.robot.base_orientation_rpy,
                      motor_angles=self.robot.motor_angles,
                      base_linear_vel=self.robot.base_linear_velocity,
                      base_vels_body_frame=self.mpc_controller.state_estimator.com_velocity_in_body_frame,
                      # base_rpy_rate=self.robot.GetBaseRollPitchYawRate(), todo: rpy rate or angular vel ???
                      base_rpy_rate=self.robot.base_angular_velocity,
                      motor_vels=self.robot.motor_velocities,
                      contacts=self.robot.foot_contacts)
        # print(f"states: {states}")
        return states

    def get_states_vector(self):
        angle = self.states['base_rpy']

        com_position_xyz = self.mpc_controller.state_estimator.estimate_robot_x_y_z()

        base_rpy_rate = self.states['base_rpy_rate']
        com_velocity = self.states['base_vels_body_frame']

        states_vector = np.hstack((com_position_xyz, angle, com_velocity, base_rpy_rate))

        # print(f"states_vector: {states_vector}")

        # states_vector = np.hstack((angle, com_position_xyz, base_rpy_rate, com_velocity))
        return states_vector

    def get_tracking_error(self):  # this is used for computing reward
        current_time = self.current_step * 0.002
        # current_time = self.current_step * self.params.time_step

        states_vector_robot = self.get_states_vector()
        tracking_error = states_vector_robot - self.set_point
        return tracking_error

    def get_run_reward(self, x_velocity: float, move_speed: float, cos_pitch: float, dyaw: float):
        # reward = rewards.tolerance(cos_pitch * x_velocity,
        #                            bounds=(move_speed, 2 * move_speed),
        #                            margin=2 * move_speed,
        #                            value_at_margin=0,
        #                            sigmoid='linear')
        v_diff = (x_velocity * cos_pitch - move_speed)
        reward = math.exp(-2 * abs(v_diff))
        # print("velocity_reward", reward)
        reward -= 0.1 * np.abs(dyaw)

        return 0 * reward  # [0, 1] => [0, 10]

        # return 10 * reward  # [0, 1] => [0, 10]

    def get_drl_reward(self):  # todo change the reward to be consistent as MPC, get rid of the first 2 terms
        x_velocity = self.states['base_vels_body_frame'][0]
        move_speed = self.target_lin_speed[0]
        cos_pitch = math.cos(self.states['base_rpy'][1])
        dyaw = self.states['base_rpy'][2]

        # reward = self.get_run_reward(x_velocity, move_speed, cos_pitch, dyaw)

        reward = 0 * self.get_run_reward(x_velocity, move_speed, cos_pitch, dyaw)

        return reward

    def get_mpc_reward(self):  # todo this is the reward that follows the MPC
        reward = - 1 * np.linalg.norm(self.diff_q) - 1 * np.linalg.norm(self.diff_dq)
        return reward

    def get_reward(self):
        return self.get_ly_reward()

    # def get_ly_reward(self):
    #
    #     # p_vector = [3.47460562e+00, 1.24762541e+00, 3.20913563e-02, 0,
    #     #             4.95019320e-03, 3.18336706e-04, 1.50271376e-03, 1.42167852e-03,
    #     #             5.38414459e-04, 7.06091611e-04, 9.79127975e-04, 1.03626752e-03]
    #     #
    #     # p_vector = [7.98096939e-0, 5.87765318e-01, 1.16942752e-04, 1, 1.72725536e-04, 1, 1.81666331e-04, 1.81892955e-04,
    #     #             1.87235756e-04, 1, 1.88585412e-04, 1.88390268e-04]
    #
    #     p_vector = [0, 0, 0, 0, 0, 0, 1.81666331e-04, 1.81892955e-04,
    #                 1.87235756e-04, 1, 1.88585412e-04, 1.88390268e-04]
    #
    #     p_matrix = np.diag(p_vector)
    #     tracking_error_current = self.get_tracking_error()
    #     tracking_error_current = np.expand_dims(tracking_error_current, axis=-1)
    #     # tracking_error_pre = self.previous_tracking_error
    #     # tracking_error_pre = np.expand_dims(tracking_error_pre, axis=-1)
    #
    #     ly_reward_cur = np.transpose(tracking_error_current, axes=(1, 0)) @ p_matrix @ tracking_error_current
    #     # ly_reward_pre = np.transpose(tracking_error_pre, axes=(1, 0)) @ p_matrix @ tracking_error_pre
    #
    #     # return ly_reward_pre - ly_reward_cur
    #     return -1 * ly_reward_cur
    #
    def get_ly_reward(self):

        # p_vector = [0.001, 0.000, 0.000, 0.000, 0.000, 0.000,
        #            1.000, 0.000, 0.000, 0.000, 0.000, 0.000]

        p_vector = [0, 0, 0,
                    1, 0, 0,
                    1, 1.88585412e-04, 0,
                    1.88585412e-04, 1.81892955e-04, 1.87235756e-04]  # 2

        # p_vector = [0, 0, 0, 0, 0, 0, 1.81666331e-04, 1.81892955e-04,
        #            1.87235756e-04, 2, 2, 0.1]

        p_matrix1 = np.array([[6.3394, 0, 0, 0, 0, 0, 0.4188, 0, 0, 0, 0, 0],
                              [0, 1.4053, 0, 0, 0, 0, 0, 0.3018, 0, 0, 0, 0],
                              [0, 0, 94.0914, 0, 0, 0, 0, 0, 9.1062, 0, 0, 0],
                              [0, 0, 0, 95.1081, 0, 0, 0, 0, 0, 9.2016, 0, 0],
                              [0, 0, 0, 0, 95.1081, 0, 0, 0, 0, 0, 9.2016, 0],
                              [0, 0, 0, 0, 0, 1.4053, 0, 0, 0, 0, 0, 0.3018],
                              [0.4188, 0, 0, 0, 0, 0, 106.1137, 0, 0, 0, 0, 0],
                              [0, 0.3018, 0, 0, 0, 0, 0, 77.1735, 0, 0, 0, 0],
                              [0, 0, 9.1062, 0, 0, 0, 0, 0, 1.8594, 0, 0, 0],
                              [0, 0, 0, 9.2016, 0, 0, 0, 0, 0, 1.8783, 0, 0],
                              [0, 0, 0, 0, 9.2016, 0, 0, 0, 0, 0, 1.8783, 0],
                              [0, 0, 0, 0, 0, 0.3018, 0, 0, 0, 0, 0, 77.1735]]) * 1

        M_matrix = np.array([[6.33931716274651, 0, 0, 0, 0, 0, 0.39824214223179, 0, 0, 0, 0, 0],
                             [0, 1.40521824475728, 0, 0, 0, 0, 0, 0.286679284833682, 0, 0, 0, 0],
                             [0, 0, 92.2887010538464, 0, 0, 0, 0, 0, 8.92428326269013, 0, 0, 0],
                             [0, 0, 0, 93.2865880895433, 0, 0, 0, 0, 0, 9.01777538552449, 0, 0],
                             [0, 0, 0, 0, 93.2865880895433, 0, 0, 0, 0, 0, 9.01777538552449, 0],
                             [0, 0, 0, 0, 0, 1.40521824475728, 0, 0, 0, 0, 0, 0.286679284833682],
                             [0.39824214223179, 0, 0, 0, 0, 0, 97.7952232108596, 0, 0, 0, 0, 0],
                             [0, 0.286679284833682, 0, 0, 0, 0, 0, 72.6131010296885, 0, 0, 0, 0],
                             [0, 0, 8.92428326269013, 0, 0, 0, 0, 0, 1.84054305542176, 0, 0, 0],
                             [0, 0, 0, 9.01777538552449, 0, 0, 0, 0, 0, 1.8592311555769, 0, 0],
                             [0, 0, 0, 0, 9.01777538552449, 0, 0, 0, 0, 0, 1.8592311555769, 0],
                             [0, 0, 0, 0, 0, 0.286679284833682, 0, 0, 0, 0, 0, 72.6131010296885]]) * 1

        # p_matrix1 = np.array([[ 1.48523774e+02,  3.55592139e-01, -4.06874090e+00, -6.72600786e-03,
        #                       -3.16661037e-02,  3.78221596e-04, -5.92627115e+00,  3.55355531e+01,
        #                       -1.37930809e-01, -9.03477959e-01, -5.04845953e-01,  2.89392271e-02],
        #                      [ 3.55592139e-01,  4.54276424e-03,  9.01439341e-02, -4.58477379e-05,
        #                       -9.99696001e-05,  2.85500510e-06, -2.50784284e-02,  2.71733301e-01,
        #                       -1.61160698e-03, -6.92347027e-03, -2.19570986e-03,  2.21580297e-04],
        #                      [-4.06874090e+00,  9.01439341e-02,  1.08296216e+02, -1.28412645e-03,
        #                       -5.74624984e-03,  8.05491443e-05, -9.85719812e+00,  8.99331944e+00,
        #                        2.92440942e-01, -2.41624495e-01, -9.04092893e-01,  7.65495824e-03],
        #                      [-6.72600786e-03, -4.58477379e-05, -1.28412645e-03,  2.27318063e-03,
        #                        3.96923554e-06, -1.30115622e-07,  8.27576779e-04, -4.97300831e-03,
        #                        1.66950006e-05,  1.72470961e-04,  6.84584002e-05, -5.46133315e-06],
        #                      [-3.16661037e-02, -9.99696001e-05, -5.74624984e-03,  3.96923554e-06,
        #                        3.20581479e-03, -4.97661153e-08,  3.62241364e-01, -9.09232554e-03,
        #                       -1.14899368e-02,  2.30325439e-04,  3.09120957e-02, -6.85302039e-06],
        #                      [ 3.78221596e-04,  2.85500510e-06,  8.05491443e-05, -1.30115622e-07,
        #                       -4.97661153e-08,  2.27094996e-03, -9.41011953e-07,  2.84313953e-04,
        #                       -2.46895207e-06, -8.25599358e-06, -1.65870793e-08,  2.80350560e-05],
        #                      [-5.92627115e+00, -2.50784284e-02, -9.85719812e+00,  8.27576779e-04,
        #                        3.62241364e-01, -9.41011953e-07,  1.11810292e+02, -2.22939973e+00,
        #                       -3.53599393e+00,  5.95966949e-02,  9.44632060e+00, -1.73451365e-03],
        #                      [ 3.55355531e+01,  2.71733301e-01,  8.99331944e+00, -4.97300831e-03,
        #                       -9.09232554e-03,  2.84313953e-04, -2.22939973e+00,  2.71676632e+01,
        #                       -1.69953501e-01, -6.92217163e-01, -1.96084758e-01,  2.21560221e-02],
        #                      [-1.37930809e-01, -1.61160698e-03,  2.92440942e-01,  1.66950006e-05,
        #                       -1.14899368e-02, -2.46895207e-06, -3.53599393e+00, -1.69953501e-01,
        #                        1.16666837e-01,  4.14454501e-03, -3.00479647e-01, -1.38107741e-04],
        #                      [-9.03477959e-01, -6.92347027e-03, -2.41624495e-01,  1.72470961e-04,
        #                        2.30325439e-04, -8.25599358e-06,  5.95966949e-02, -6.92217163e-01,
        #                        4.14454501e-03,  1.93599779e-02,  5.12011067e-03, -5.53209808e-04],
        #                      [-5.04845953e-01, -2.19570986e-03, -9.04092893e-01,  6.84584002e-05,
        #                        3.09120957e-02, -1.65870793e-08,  9.44632060e+00, -1.96084758e-01,
        #                       -3.00479647e-01,  5.12011067e-03,  8.05850092e-01, -1.48692679e-04],
        #                      [ 2.89392271e-02,  2.21580297e-04,  7.65495824e-03, -5.46133315e-06,
        #                       -6.85302039e-06,  2.80350560e-05, -1.73451365e-03,  2.21560221e-02,
        #                       -1.38107741e-04, -5.53209808e-04, -1.48692679e-04,  2.28066593e-03]])*1

        p_matrix2 = np.diag(p_vector)

        # p_matrix = 0.001*p_matrix1 + p_matrix2

        tracking_error_current = self.get_tracking_error()
        tracking_error_current = np.expand_dims(tracking_error_current, axis=-1)
        tracking_error_pre = self.previous_tracking_error
        tracking_error_pre = np.expand_dims(tracking_error_pre, axis=-1)

        # print(tracking_error_current)
        # print(tracking_error_pre)

        ly_reward_cur = np.transpose(tracking_error_current, axes=(1, 0)) @ p_matrix1 @ tracking_error_current

        ly_reward_pre = np.transpose(tracking_error_pre, axes=(1, 0)) @ p_matrix1 @ tracking_error_pre

        # ly_reward_pre = np.transpose(tracking_error_pre, axes=(1, 0)) @ M_matrix @ tracking_error_pre

        ly_reward2 = np.transpose(tracking_error_current, axes=(1, 0)) @ p_matrix2 @ tracking_error_current

        # print(ly_reward_pre - ly_reward_cur)
        # print(ly_reward2)

        # print(self.termination)

        penl = self.termination * 0  # 50

        # reward = (ly_reward_pre - ly_reward_cur)*1 + (ly_reward2)*-10

        # print(penl)

        # reward = ((ly_reward_pre - ly_reward_cur) * .01) + ((ly_reward2) * -0) - penl

        # reward = ((ly_reward_pre - ly_reward_cur) * .01) + ((ly_reward2) * -1) - penl

        reward = ((ly_reward_pre - ly_reward_cur) * 0.01) + ((ly_reward2) * 0.0) - penl

        # print(ly_reward_pre - ly_reward_cur)

        return reward

    # def initialize_env(self):
    #     self.reset(step=0)

    def step(self, action, action_mode='mpc'):
        """
        Here the action is generated from DRL agent, that controls ground reaction force (GRF).
        dim: 12, 3 dims (motors) for each leg action is in [-1,1]
        """
        self.previous_tracking_error = self.get_tracking_error()

        if action_mode == 'residual':
            self.mpc_controller.set_desired_speed(self.target_lin_speed, self.target_ang_speed)
            # _update_controller_params(self.mpc_control, self.target_lin_speed, self.target_ang_speed)

            self.mpc_controller.update()  # update the clock

            # rescale the action to be [0.5, 1.5], this will be a multiplier to scale up/down the mpc action
            action *= self._params.action_magnitude
            s = time.time()
            # applied_action, _, diff_q, diff_dq = self.mpc_controller.get_drl_action(self.current_step,
            #
            #                                                                         self.get_states_vector(), action)
            if self.mpc_controller._ha_teacher.teacher_enable:
                curr_state = self.mpc_controller.stance_leg_controller.tracking_error  # Current state
                applied_action, qp_sol = self.mpc_controller._ha_teacher.get_hac_action(states=curr_state)
            else:
                applied_action, _ = self.mpc_controller.get_action(drl_action=action)

            # self.diff_q = diff_q
            # self.diff_dq = diff_dq
            e = time.time()
            # print(f"applied_action: {applied_action}")
            # print(f"diff_q: {diff_q}")
            # print(f"diff_dq: {diff_dq}")
            print(f"residual control get action time: {e - s}")

        elif action_mode == 'mpc':
            self.mpc_controller.set_desired_speed(self.target_lin_speed, self.target_ang_speed)
            # _update_controller_params(self.mpc_control, self.target_lin_speed, self.target_ang_speed)

            self.mpc_controller.update()  # update the clock
            # applied_action, _, diff_q, diff_dq = self.mpc_control.get_action()
            applied_action, _, diff_q, diff_dq = self.mpc_controller.get_drl_action(self.current_step,
                                                                                    self.get_states_vector())
        else:
            action *= 100  # to check the dim and magnitude of the action
            applied_action = action

        self.robot.step(applied_action)

        state = self.get_state()

        observation, termination, abort = self.get_observations(state)

        self.states = state  # update the states buffer
        self.observation = observation
        self.termination = termination

        self.current_step += 1

        return observation, termination, abort

    def get_performance_score(self):
        pass

    def add_terrain(self):
        boxHalfLength = 0.2
        boxHalfWidth = 2.5
        boxHalfHeight = 0.05
        sh_colBox = self.p.createCollisionShape(self.p.GEOM_BOX, halfExtents=[0.5, boxHalfWidth, 0.05])
        sh_final_col = self.p.createCollisionShape(self.p.GEOM_BOX, halfExtents=[0.5, boxHalfWidth, 0.05])
        boxOrigin = 0.8 + boxHalfLength
        step1 = self.p.createMultiBody(baseMass=0, baseCollisionShapeIndex=sh_colBox,
                                       basePosition=[boxOrigin, 1, boxHalfHeight],
                                       baseOrientation=[0.0, 0.0, 0.0, 1])

        step2 = self.p.createMultiBody(baseMass=0, baseCollisionShapeIndex=sh_final_col,
                                       basePosition=[boxOrigin + 0.5 + boxHalfLength, 1, 0.05 + 2 * boxHalfHeight],
                                       baseOrientation=[0.0, 0.0, 0.0, 1])

        self.p.changeDynamics(step1, -1, lateralFriction=0.85)
        self.p.changeDynamics(step2, -1, lateralFriction=0.85)

    def add_lane(self):

        # all units are in meters
        track_length = 15
        track_width = 0.03
        track_height = 0.0005
        lane_half_width = 0.6
        track_left = self.p.createVisualShape(self.p.GEOM_BOX, halfExtents=[track_length, track_width, track_height],
                                              rgbaColor=[1, 0, 0, 0.7])
        track_middle = self.p.createVisualShape(self.p.GEOM_BOX, halfExtents=[track_length, track_width, track_height],
                                                rgbaColor=[0, 0, 1, 0.7])
        track_right = self.p.createVisualShape(self.p.GEOM_BOX, halfExtents=[track_length, track_width, track_height],
                                               rgbaColor=[1, 0, 0, 0.7])

        boxOrigin_x = 0
        self.p.createMultiBody(baseMass=0, baseVisualShapeIndex=track_left,
                               basePosition=[boxOrigin_x, lane_half_width, 0.0005],
                               baseOrientation=[0.0, 0.0, 0.0, 1])

        self.p.createMultiBody(baseMass=0, baseVisualShapeIndex=track_middle,
                               basePosition=[boxOrigin_x, 0, 0.0005],
                               baseOrientation=[0.0, 0.0, 0.0, 1])

        self.p.createMultiBody(baseMass=0, baseVisualShapeIndex=track_right,
                               basePosition=[boxOrigin_x, -lane_half_width, 0.0005],
                               baseOrientation=[0.0, 0.0, 0.0, 1])
