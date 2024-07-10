"""Training Environment for A1"""
import math
import time
import numpy as np
import pybullet_data
import scipy.interpolate
from omegaconf import DictConfig
from pybullet_utils import bullet_client
import pybullet  # pytype:disable=import-error

from src.envs.robot.unitree_a1 import a1
from src.envs.robot.unitree_a1 import a1_robot
from src.envs.simulator.utils import add_terrain, add_lane


class A1Envs:
    def __init__(self, a1_envs_cfg: DictConfig, agent=None) -> None:

        self._robot_cfg = a1_envs_cfg.robot
        self._sim_cfg = a1_envs_cfg.simulator

        # if self._sim_cfg.show_gui:
        if self._robot_cfg.interface.model == 'a1':
            self.p = bullet_client.BulletClient(connection_mode=pybullet.GUI)
        else:
            self.p = bullet_client.BulletClient(connection_mode=pybullet.DIRECT)

        self.robot = None
        self.robot_controller = None
        self.agent = agent

        self.states_observations_dim = 12
        self.action_dim = 6

        self.termination = None
        self.state = None
        self.observation = None

        self.diff_q = None
        self.diff_dq = None
        self.current_step = 0
        self.previous_tracking_error = None

        # reference set point
        self.ref_vx = self._robot_cfg.command.desired_vx
        self.ref_vy = self._robot_cfg.command.desired_vy
        self.ref_wz = self._robot_cfg.command.desired_wz
        self.ref_pz = self._robot_cfg.command.mpc_body_height

        self.set_point = np.array(
            [0, 0, self.ref_pz,  # p
             0., 0., 0.,  # rpy
             self.ref_vx, self.ref_vy, 0,  # v
             0., 0., self.ref_wz]  # w
        )

        self.target_lin_speed = [self.ref_vx, self.ref_vy, 0]
        self.target_ang_speed = self.ref_wz
        self.fall_threshold = self._robot_cfg.command.safe_height

        self.reset(step=0)

    def random_reset(self):
        pass

    def reset_move_back(self, step, random_reset=False):
        self.p.resetSimulation()
        self.p.setGravity(0, 0, -9.8)
        self.p.setTimeStep(self._envs_cfg.fixed_time_step)
        self.p.setAdditionalSearchPath(self._envs_cfg.envs_path)
        self.p.setPhysicsEngineParameter(numSolverIterations=self._envs_cfg.num_solver_iterations)
        self.p.setPhysicsEngineParameter(enableConeFriction=self._envs_cfg.enable_cone_friction)
        plane = self.p.loadURDF("envs/training_env/meshes/plane.urdf")

        if random_reset:
            l, r = self._envs_cfg.random_reset.friction
            friction_coeff = np.random.uniform(l, r)
            self.p.changeDynamics(plane, -1, lateralFriction=friction_coeff)
        else:
            self.p.changeDynamics(plane, -1, lateralFriction=self._envs_cfg.friction)

        # if self._params.use_real_urdf:  # whether to use a more realistic urdf file for training
        #     self.p.setAdditionalSearchPath("envs/sim_envs_v2")
        # else:
        #     self.p.setAdditionalSearchPath(pybullet_data.getDataPath())

        # Add terrain
        if self._envs_cfg.add_terrain:
            add_terrain(self.p)

        # Add lane
        if self._envs_cfg.add_lane:
            add_lane(self.p)

        if self._envs_cfg.record_video:
            self.p.startStateLogging(self.p.STATE_LOGGING_VIDEO_MP4, f"{step}_record.mp4")

        # pp = self.p.getPhysicsEngineParameters()
        # print(f"pp is: {pp}")
        # time.sleep(123)

        if self._robot_cfg.robot_model.model_name == 'a1_model':
            self.robot = a1.A1(
                pybullet_client=self.p,
                cmd_params=self._robot_cfg.command,
                a1_params=self._robot_cfg.robot_model,
                gait_params=self._robot_cfg.gait_scheduler,
                swing_params=self._robot_cfg.swing_controller,
                stance_params=self._robot_cfg.stance_controller,
                motor_params=self._robot_cfg.motor_group,
                vel_estimator_params=self._robot_cfg.com_velocity_estimator,
                logdir='saved/logs/robot/training'
            )
        elif self._robot_cfg.robot_model.model_name == 'a1_robot_model':
            self.robot = a1_robot.A1Robot(
                pybullet_client=p,
                ddpg_agent=self.agent,
                cmd_params=cfg.robot.command,
                a1_robot_params=cfg.robot.robot_model.a1_robot_model,
                gait_params=cfg.robot.gait_scheduler,
                swing_params=cfg.robot.swing_controller,
                stance_params=cfg.robot.stance_controller,
                motor_params=cfg.robot.motor_group,
                vel_estimator_params=cfg.robot.com_velocity_estimator
            )
        else:
            raise RuntimeError("Cannot find predefined robot model")

        if random_reset:
            lv, hv = self._envs_cfg.random_reset.velocity
            initial_vx = np.random.uniform(lv, hv)
            self.p.resetBaseVelocity(self.robot.quadruped, linearVelocity=[initial_vx, 0, 0])

        self.locomotion_controller = self.robot.controller  # Robot Locomotion Controller
        self.locomotion_controller.update()

        # self.state = self.get_state()
        self.state = self.locomotion_controller.robot_state
        # print(f"self.state: {self.state}")
        self.previous_tracking_error = self.get_tracking_error()
        # print(f"tracking error: {self.previous_tracking_error}")
        self.observation, self.termination, _ = self.get_observation(self.state)
        self.current_step = 0

    def reset(self, step, random_reset=False):

        self.p.resetSimulation()
        self.p.setGravity(0, 0, -9.8)
        self.p.setTimeStep(self._sim_cfg.fixed_time_step)
        self.p.setAdditionalSearchPath(self._sim_cfg.envs_path)
        self.p.setPhysicsEngineParameter(numSolverIterations=self._sim_cfg.num_solver_iterations)
        self.p.setPhysicsEngineParameter(enableConeFriction=self._sim_cfg.enable_cone_friction)
        plane = self.p.loadURDF(self._sim_cfg.plane_urdf_path)

        if random_reset:
            l, r = self._sim_cfg.random_reset.friction
            friction_coeff = np.random.uniform(l, r)
            self.p.changeDynamics(plane, -1, lateralFriction=friction_coeff)
        else:
            self.p.changeDynamics(plane, -1, lateralFriction=self._sim_cfg.friction)

        # if self._params.use_real_urdf:  # whether to use a more realistic urdf file for training
        #     self.p.setAdditionalSearchPath("envs/sim_envs_v2")
        # else:
        #     self.p.setAdditionalSearchPath(pybullet_data.getDataPath())

        # Add terrain
        if self._sim_cfg.add_terrain:
            add_terrain(self.p)

        # Add lane
        if self._sim_cfg.add_lane:
            add_lane(self.p)

        if self._sim_cfg.record_video:
            self.p.startStateLogging(self.p.STATE_LOGGING_VIDEO_MP4, f"{step}_record.mp4")

        # pp = self.p.getPhysicsEngineParameters()
        # print(f"pp is: {pp}")
        # time.sleep(123)

        if self._robot_cfg.interface.model == 'a1':
            self.robot = a1.A1(
                pybullet_client=self.p,
                cmd_params=self._robot_cfg.command,
                a1_params=self._robot_cfg.interface,
                gait_params=self._robot_cfg.gait_scheduler,
                swing_params=self._robot_cfg.swing_controller,
                stance_params=self._robot_cfg.stance_controller,
                motor_params=self._robot_cfg.motor_group,
                vel_estimator_params=self._robot_cfg.com_velocity_estimator,
                logdir='saved/logs/robot/training'
            )
        elif self._robot_cfg.interface.model == 'a1_robot':
            self.robot = a1_robot.A1Robot(
                ddpg_agent=self.agent,
                pybullet_client=self.p,
                cmd_params=self._robot_cfg.command,
                a1_robot_params=self._robot_cfg.interface,
                gait_params=self._robot_cfg.gait_scheduler,
                swing_params=self._robot_cfg.swing_controller,
                stance_params=self._robot_cfg.stance_controller,
                motor_params=self._robot_cfg.motor_group,
                vel_estimator_params=self._robot_cfg.com_velocity_estimator,
            )
        else:
            raise RuntimeError(f"Cannot find robot model: {self._robot_cfg.interface.model}")

        if random_reset:
            lv, hv = self._envs_cfg.random_reset.velocity
            initial_vx = np.random.uniform(lv, hv)
            self.p.resetBaseVelocity(self.robot.quadruped, linearVelocity=[initial_vx, 0, 0])

        self.locomotion_controller = self.robot.controller  # Robot Locomotion Controller
        self.locomotion_controller.update()

        # self.state = self.get_state()
        self.state = self.locomotion_controller.robot_state
        # print(f"self.state: {self.state}")
        self.previous_tracking_error = self.tracking_error
        # print(f"tracking error: {self.previous_tracking_error}")
        self.observation, self.termination, _ = self.get_observation(self.state)
        self.current_step = 0

    def get_observation(self, state_vector):
        observation = []  # 16 dims

        termination = False
        abort = False

        observation.extend(state_vector)  # 12 dims
        # observation.extend(self.robot.foot_contacts)  # 4 dims

        com_height = self.locomotion_controller.state_estimator.com_position_in_ground_frame[2]

        if com_height < self.fall_threshold:
            print(
                f"The height of robot is: {com_height}, which exceeds the safety boundary, the robot may fall")
            termination = True

        if math.isnan(float(self.robot.base_linear_velocity[0])):
            abort = True
            termination = True  # new add
            print("ABORT_DUE_TO_SIMULATION_ERROR")

        return observation, termination, abort

    def get_state(self):
        states = dict(timestamp=self.robot.time_since_reset,
                      base_rpy=self.robot.base_orientation_rpy,
                      motor_angles=self.robot.motor_angles,
                      base_linear_vel=self.robot.base_linear_velocity,
                      base_vels_body_frame=self.locomotion_controller.state_estimator.com_velocity_in_body_frame,
                      # base_rpy_rate=self.robot.GetBaseRollPitchYawRate(), todo: rpy rate or angular vel ???
                      base_rpy_rate=self.robot.base_angular_velocity,
                      motor_vels=self.robot.motor_velocities,
                      contacts=self.robot.foot_contacts)
        # print(f"states: {states}")
        return states

    def get_states_vector(self):
        angle = self.state['base_rpy']

        com_position_xyz = self.locomotion_controller.state_estimator.estimate_robot_x_y_z()

        base_rpy_rate = self.state['base_rpy_rate']
        com_velocity = self.state['base_vels_body_frame']

        states_vector = np.hstack((com_position_xyz, angle, com_velocity, base_rpy_rate))

        # print(f"states_vector: {states_vector}")

        # states_vector = np.hstack((angle, com_position_xyz, base_rpy_rate, com_velocity))
        return states_vector

    @property
    def tracking_error(self):  # this is used for computing reward
        current_state = self.locomotion_controller.robot_state
        reference_state = self.locomotion_controller.ref_point
        # print(f"current_state: {current_state}")
        # print(f"reference_state: {reference_state}")
        error_state = current_state - reference_state
        return error_state

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
        x_velocity = self.state['base_vels_body_frame'][0]
        move_speed = self.target_lin_speed[0]
        cos_pitch = math.cos(self.state['base_rpy'][1])
        dyaw = self.state['base_rpy'][2]

        # reward = self.get_run_reward(x_velocity, move_speed, cos_pitch, dyaw)

        reward = 0 * self.get_run_reward(x_velocity, move_speed, cos_pitch, dyaw)

        return reward

    def get_mpc_reward(self):  # todo this is the reward that follows the MPC
        reward = - 1 * np.linalg.norm(self.diff_q) - 1 * np.linalg.norm(self.diff_dq)
        return reward

    def get_reward(self, s, s_next):
        # return self.get_ly_reward()
        return self.get_lyapunov_reward(s, s_next)

    def get_lyapunov_reward(self, s, s_next):

        p_matrix = np.array([[6.3394, 0, 0, 0, 0, 0, 0.4188, 0, 0, 0, 0, 0],
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

        p_matrix2 = np.array([[0., -0., 0.0006, 0., 0., 0.0003, 0.0011, 0., -0., -0., -0., 0.],
                              [-0., 0., -0., -0., -0., -0., 0., 0., 0., 0., 0., -0.],
                              [0.0006, -0., 14.9873, 0., 0.0038, 0.0116, -0.0263, 0., -0.0021, -0., -0.0005, 0.0001],
                              [0., -0., 0., 0., 0., -0., 0., -0., -0., 0., -0., -0.],
                              [0., -0., 0.0038, 0., 0.0003, 0.0014, 0.0004, 0., 0.0002, -0., 0.0001, -0.],
                              [0.0003, -0., 0.0116, -0., 0.0014, 0.6157, 0.0881, 0., 0.0058, -0., 0.0001, 0.007],
                              [0.0011, 0., -0.0263, 0., 0.0004, 0.0881, 0.4242, -0., -0.0025, 0., -0.0008, 0.0022],
                              [0., 0., 0., -0., 0., 0., -0., 0., 0., -0., 0., -0.],
                              [-0., 0., -0.0021, -0., 0.0002, 0.0058, -0.0025, 0., 0.0028, -0., 0.0002, 0.],
                              [-0., 0., -0., 0., -0., -0., 0., -0., -0., 0., -0., 0.],
                              [-0., 0., -0.0005, -0., 0.0001, 0.0001, -0.0008, 0., 0.0002, -0., 0., -0.],
                              [0., -0., 0.0001, -0., -0., 0.007, 0.0022, -0., 0., 0., -0., 0.0001]])

        p_matrix3 = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 120.0539, 0, 0, 0, 0, 0, 3.3359, 0, 0, 0],
                              [0, 0, 0, 0.0014, 0, 0, 0, 0, 0, 0.0001, 0, 0],
                              [0, 0, 0, 0, 0.0014, 0, 0, 0, 0, 0, 0.0001, 0],
                              [0, 0, 0, 0, 0, 137.6301, 0, 0, 0, 0, 0, 4.5877],
                              [0, 0, 0, 0, 0, 0, 1.5004, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0.0061, 0, 0, 0, 0],
                              [0, 0, 3.3359, 0, 0, 0, 0, 0, 0.6706, 0, 0, 0],
                              [0, 0, 0, 0.0001, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0.0001, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 4.5877, 0, 0, 0, 0, 0, 0.2303]])

        # p_matrix = 0.001*p_matrix1 + p_matrix2

        # tracking_error_current = self.get_tracking_error()
        # tracking_error_current = np.expand_dims(tracking_error_current, axis=-1)
        # tracking_error_pre = self.previous_tracking_error
        # tracking_error_pre = np.expand_dims(tracking_error_pre, axis=-1)

        # print(tracking_error_current)
        # print(tracking_error_pre)
        # p_matrix1 = self._dynamics.P  # New P Matrix
        # p_quadratic = self._dynamics.A_bar.T @ self._dynamics.P @ self._dynamics.A_bar

        # p_bar = self._dynamics.A_bar.T @ p_matrix3 @ self._dynamics.A_bar
        ly_reward_curr = s.T @ p_matrix3 @ s
        # ly_reward_curr = s.T @ p_bar @ s

        ly_reward_next = s_next.T @ p_matrix3 @ s_next
        # ly_reward_pre = np.transpose(tracking_error_pre, axes=(1, 0)) @ p_quadratic @ tracking_error_pre

        # ly_reward_pre = np.transpose(tracking_error_pre, axes=(1, 0)) @ M_matrix @ tracking_error_pre

        # ly_reward2 = np.transpose(tracking_error_current, axes=(1, 0)) @ p_matrix2 @ tracking_error_current

        # print(ly_reward_pre - ly_reward_cur)
        # print(ly_reward2)

        # print(self.termination)

        # penl = self.termination * 0  # 50

        # reward = (ly_reward_pre - ly_reward_cur)*1 + (ly_reward2)*-10

        reward = ly_reward_curr - ly_reward_next

        # print(ly_reward_pre - ly_reward_cur)

        return reward

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

        p_matrix2 = np.diag(p_vector)

        # p_matrix = 0.001*p_matrix1 + p_matrix2

        tracking_error_current = self.get_tracking_error()
        tracking_error_current = np.expand_dims(tracking_error_current, axis=-1)
        tracking_error_pre = self.previous_tracking_error
        tracking_error_pre = np.expand_dims(tracking_error_pre, axis=-1)

        # print(tracking_error_current)
        # print(tracking_error_pre)
        # p_matrix1 = self._dynamics.P  # New P Matrix
        # p_quadratic = self._dynamics.A_bar.T @ self._dynamics.P @ self._dynamics.A_bar

        ly_reward_cur = np.transpose(tracking_error_current, axes=(1, 0)) @ p_matrix1 @ tracking_error_current

        ly_reward_pre = np.transpose(tracking_error_pre, axes=(1, 0)) @ p_matrix1 @ tracking_error_pre
        # ly_reward_pre = np.transpose(tracking_error_pre, axes=(1, 0)) @ p_quadratic @ tracking_error_pre

        # ly_reward_pre = np.transpose(tracking_error_pre, axes=(1, 0)) @ M_matrix @ tracking_error_pre

        ly_reward2 = np.transpose(tracking_error_current, axes=(1, 0)) @ p_matrix2 @ tracking_error_current

        # print(ly_reward_pre - ly_reward_cur)
        # print(ly_reward2)

        # print(self.termination)

        penl = self.termination * 0  # 50

        # reward = (ly_reward_pre - ly_reward_cur)*1 + (ly_reward2)*-10

        # print(penl)

        # reward = ((ly_reward_pre - ly_reward_cur) * 0.01) + ((ly_reward2) * 0.0) - penl
        reward = ly_reward_pre - ly_reward_cur

        # print(ly_reward_pre - ly_reward_cur)

        return reward

    # def initialize_env(self):
    #     self.reset(step=0)

    def env_step(self, applied_action):
        """
        Here the action is generated from DRL agents, that controls ground reaction force (GRF).
        dim: 12, 3 dims (motors) for each leg action is in [-1,1]
        """
        # applied_action = None
        self.previous_tracking_error = self.tracking_error
        self.locomotion_controller.set_desired_speed(self.target_lin_speed, self.target_ang_speed)

        # print(f"mode: {self.locomotion_controller.mode}")
        # print(f"swing desired speed: {self.locomotion_controller.swing_leg_controller.desired_speed}")
        # print(f"stance desired speed: {self.locomotion_controller.stance_leg_controller.desired_speed}")

        # Robot step to get next state
        self.robot.step(applied_action)
        self.locomotion_controller.update()  # update the clock

        self.state = self.locomotion_controller.robot_state  # Update the states buffer

        observation, termination, abort = self.get_observation(self.state)

        self.observation = observation
        self.termination = termination

        self.current_step += 1

        return observation, termination, abort

    def get_performance_score(self):
        pass
