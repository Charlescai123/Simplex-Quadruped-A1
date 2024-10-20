"""A phydrl based controller framework."""

import os
import enum
import copy
import time
import pickle
import pybullet
import threading
import numpy as np
import ml_collections
import multiprocessing
from datetime import datetime
from pybullet_utils import bullet_client
from typing import Tuple, Any
from typing import Mapping
from absl import logging
from omegaconf import DictConfig

# from robot.robot import a1
from src.hp_student.agents.ddpg import DDPGAgent
from src.envs.simulator.worlds import plane_world, abstract_world
from src.envs.robot.unitree_a1.motors import MotorCommand
from src.envs.robot.unitree_a1.motors import MotorControlMode
from src.envs.robot.gait_scheduler import offset_gait_scheduler
from src.envs.robot.state_estimator import com_velocity_estimator
from src.envs.robot.mpc_controller import swing_leg_controller
from src.envs.robot.mpc_controller import stance_leg_controller_mpc
from src.envs.robot.mpc_controller import stance_leg_controller_quadprog
import tensorflow as tf
from src.hp_student.agents.replay_mem import ReplayMemory


class ControllerMode(enum.Enum):
    DOWN = 1
    STAND = 2
    WALK = 3
    TERMINATE = 4


class GaitType(enum.Enum):
    CRAWL = 1
    TROT = 2
    FLYTROT = 3


class LocomotionController(object):
    """Robot's Locomotion Controller.

    The actual effect of this controller depends on the composition of each
    individual subcomponent.

    """

    def __init__(
            self,
            robot: Any = None,
            ddpg_agent: DDPGAgent = None,
            desired_speed: Tuple[float, float] = [0., 0.],
            desired_twisting_speed: float = 0.,
            desired_com_height: float = 0.,
            mpc_body_mass: float = 110 / 9.8,
            mpc_body_inertia: Tuple[float, float, float, float, float, float, float, float, float] = (
                    0.07335, 0, 0, 0, 0.25068, 0, 0, 0, 0.25447),
            gait_config: DictConfig = None,
            vel_estimator_config: DictConfig = None,
            swing_config: DictConfig = None,
            stance_config: DictConfig = None,
            logdir: str = 'logs/',
    ):
        """Initializes the class.

        Args:
          robot: A robot instance. (Provides sensor input and kinematics)
          ddpg_agent: An agents instance used to get PhyDRL action (WBC will get mpc action if None)
          desired_speed: desired CoM speed in x-y plane.
          desired_twisting_speed: desired CoM rotating speed in z direction.
          desired_com_height: The standing height of CoM of the robot.
          mpc_body_mass: The total mass of the robot.
          mpc_body_inertia: The inertia matrix in the body principle frame. We assume the body principle
                            coordinate frame has x-forward and z-up.
          swing_params: Parameters for swing leg controller.
          stance_params: Parameters for stance leg controller.
        """

        self._robot = robot
        self._ddpg_agent = ddpg_agent
        self._gait_scheduler = None
        self._velocity_estimator = None

        # Parameter config
        self._gait_config = gait_config
        self._swing_config = swing_config
        self._stance_config = stance_config
        self._vel_estimator_config = vel_estimator_config

        from src.ha_teacher.ha_teacher import HATeacher
        from src.coordinator.coordinator import Coordinator
        from omegaconf import DictConfig

        teacher_config = DictConfig(
            {"chi": 0.3, "teacher_learn": True, "teacher_enable": True, "epsilon": 0.6, "tau": 10,
             "cvxpy_solver": "solver"})
        self.ha_teacher = HATeacher(robot=self._robot, teacher_cfg=teacher_config)
        self.coordinator = Coordinator()

        # Uncomment for online CL
        # self._initial_online_cl()
        #
        # self._replay_buffer = ReplayMemory(size=1e6)
        # self._prefill_size = 512
        # self._minibatch_size = 512

        # Logs
        self._logs = []
        self._logdir = logdir

        # Desired v/w
        self._desired_speed = desired_speed
        self._desired_twisting_speed = desired_twisting_speed

        # MPC parameters
        self._desired_com_height = desired_com_height
        self._mpc_body_mass = mpc_body_mass
        self._mpc_body_inertia = np.asarray(mpc_body_inertia)

        self._desired_mode = None
        self._is_control = False

        self._reset_time = None
        self._setup_controllers()

        # self._reset_time = self._clock()
        self._time_since_reset = 0

        self._mode = ControllerMode.WALK
        self.set_controller_mode(ControllerMode.WALK)
        self._gait = None
        self._desired_gait = GaitType.TROT

        # self._handle_gait_switch()

        self._control_thread = None
        # self._control_thread = threading.Thread(target=self.run)
        # self.run_thread.start()
        # self.run_thread = multiprocessing.Process(target=self.run)
        # self.run_thread.start()

        self._robot_state = np.zeros(12)
        vx = self._desired_speed[0]
        vy = self._desired_speed[1]
        wz = self._desired_twisting_speed
        pz = self._desired_com_height
        self._ref_point = np.array([0., 0., pz,  # p
                                    0., 0., 0.,  # rpy
                                    vx, vy, 0.,  # v
                                    0., 0., wz])  # rpy_dot

        # Cached variables for real time efficiency when indexing
        if self._ddpg_agent is not None:
            self._action_magnitude = np.array(self._ddpg_agent.params.action.magnitude)

        TFLITE_FILE_PATH = 'tf_models/tf_lite/model.tflite'
        # Load the TFLite model and allocate tensors.
        self.actor_lite = tf.lite.Interpreter(model_path=TFLITE_FILE_PATH)
        self.actor_lite.allocate_tensors()
        # Get input and output tensors.
        self.lite_input_details = self.actor_lite.get_input_details()
        self.lite_output_details = self.actor_lite.get_output_details()

        self.hp_action = np.array([0., 0., 0., 0., 0., 0.])
        self.ha_action = np.array([0., 0., 0., 0., 0., 0.])
        self.reset_controllers()
        self.beta_distribution_noise = np.random.beta(a=0.5, b=0.5, size=6) * 0.5

    def _initial_online_cl(self):
        ################################### For Online Continual Learning ###################################
        # self._replay_buffer = ReplayMemory(size=1e6)
        self._max_episode_steps = 10000
        self._total_steps = 50000
        self._prefill_size = 512
        self._minibatch_size = 512

        # self._prefill_size = 300
        # self._minibatch_size = 300

        def count_dirs_in_directory(directory):
            files_and_dirs = os.listdir(directory)
            dirs = [d for d in files_and_dirs if os.path.isdir(os.path.join(directory, d))]
            return len(dirs)

        self._continual_dir = 'continual_learn'
        self._replay_buffer_path = self._continual_dir + '/buffer.pickle'
        self._statistic_file_path = self._continual_dir + '/statistic.txt'
        # self._replay_buffer_path = None
        self._online_cl_model_dir = self._continual_dir + '/online_cl_model/'

        self._model_ep_cnt = count_dirs_in_directory(self._online_cl_model_dir)

        if self._model_ep_cnt == 0:
            # self._pretrained_weights_path = self._continual_dir + f'/pre-trained/friction_0.7_delay_yaw_penalty'
            self._pretrained_weights_path = self._continual_dir + f'/pre-trained/friction_0.7_delay_vel_0.6_sub_optimal1_best'
        else:
            self._pretrained_weights_path = self._online_cl_model_dir + f'ep{self._model_ep_cnt}'
        self._weight_save_path = self._online_cl_model_dir + f'ep{self._model_ep_cnt + 1}'

        # Load weights
        if os.path.isdir(self._pretrained_weights_path):
            self.ddpg_agent.load_weights(self._pretrained_weights_path)
            print(f"Loading pretrained model {self._pretrained_weights_path} from the last training")

        # Load replay buffer
        try:
            with open(self._replay_buffer_path, 'rb') as f:
                self._replay_buffer = pickle.load(f)
                print(f"Loading replay buffer {self._replay_buffer_path} from the last training")
                # self.ddpg_agent.change_learning_rate(lr_rate_actor=0.0001, lr_rate_critic=0.0001)
                # self.ddpg_agent.get_learning_rate()
                self.ddpg_agent.change_learning_rate(lr_rate_actor=0., lr_rate_critic=0.)
                self.ddpg_agent.get_learning_rate()
                t0 = time.time()
                for _ in range(1):
                    minibatch = self._replay_buffer.sample(self._minibatch_size)
                    _ = self.ddpg_agent.optimize(minibatch)
                t1 = time.time()
                print(f"initial optimize time: {t1 - t0}")
        except:
            self._replay_buffer = ReplayMemory(size=1e6)
            for _ in range(self._prefill_size):
                virtual_experience = (np.zeros(12), np.zeros(6), np.zeros(1), np.zeros(12), False)
                self._replay_buffer.add(virtual_experience)
            print(f"replay buffer size: {self._replay_buffer.get_size()}")
            print(f"ddpg agent: {self.ddpg_agent}")
            minibatch = self._replay_buffer.sample(self._minibatch_size)
            # self._ddpg_agent.change_learning_rate(lr_rate_actor=0., lr_rate_critic=0.)
            # _ = self.ddpg_agent.optimize(minibatch)
            self._replay_buffer = ReplayMemory(size=1e6)

        # try:
        #     with open(self._replay_buffer_path, 'rb') as f:
        #         self._replay_buffer = pickle.load(f)
        #         print(f"Loading replay buffer {self._replay_buffer_path} from the last training")
        #         # self.ddpg_agent.change_learning_rate(lr_rate_actor=0.0001, lr_rate_critic=0.0001)
        #         self.ddpg_agent.change_learning_rate(lr_rate_actor=0, lr_rate_critic=0)
        #         for _ in range(1):
        #             minibatch = self._replay_buffer.sample(self._minibatch_size)
        #             _ = self.ddpg_agent.optimize(minibatch)
        # except e:
        #     print(f"exception is: {e}")
        # self._replay_buffer = ReplayMemory(size=2e5)
        # for _ in range(512):
        #     virtual_experience = (np.zeros(12), np.zeros(6), np.zeros(1), np.zeros(12), False)
        #     self._replay_buffer.add(virtual_experience)
        # minibatch = self._replay_buffer.sample(self._minibatch_size)
        # self.ddpg_agent.change_learning_rate(lr_rate_actor=0., lr_rate_critic=0.)
        # _ = self.ddpg_agent.optimize(minibatch)
        # self._replay_buffer = ReplayMemory(size=2e5)

        # # optimization warm up
        self.ddpg_agent.change_learning_rate(lr_rate_actor=self.ddpg_agent.params.learning_rate_actor,
                                             lr_rate_critic=self.ddpg_agent.params.learning_rate_critic)
        self.ddpg_agent.get_learning_rate()
        # print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!r")
        # time.sleep(123)
        # self._ddpg_agent.change_learning_rate(lr_rate_actor=0, lr_rate_critic=0)

        self.ddpg_agent.agent_warmup()
        self.reward = 0
        # time.sleep(15)
        ####################################################################################################

    def _setup_controllers(self):
        print("Setting up the whole body controller...")
        self._clock = lambda: self._robot.time_since_reset

        # Gait Generator
        print("Setting up the gait generator")
        init_gait_phase = np.array(self._gait_config.init_gait_phase)
        gait_param_tuple = tuple(self._gait_config.gait_parameter_tuple)
        self._gait_scheduler = offset_gait_scheduler.OffsetGaitScheduler(
            robot=self._robot,
            init_phase=init_gait_phase,
            gait_parameters=gait_param_tuple
        )

        # State Estimator
        print("Setting up the state estimator")
        window_size = self._vel_estimator_config.window_size
        ground_normal_window_size = self._vel_estimator_config.ground_normal_window_size
        self._velocity_estimator = com_velocity_estimator.COMVelocityEstimator(
            robot=self._robot,
            velocity_window_size=window_size,
            ground_normal_window_size=ground_normal_window_size
        )

        # Swing Leg Controller
        print("Setting up the swing leg controller")
        self._swing_controller = \
            swing_leg_controller.RaibertSwingLegController(
                robot=self._robot,
                gait_scheduler=self._gait_scheduler,
                state_estimator=self._velocity_estimator,
                desired_speed=self._desired_speed,
                desired_twisting_speed=self._desired_twisting_speed,
                desired_com_height=self._desired_com_height,
                swing_params=self._swing_config
            )

        # Stance Leg Controller
        print("Setting up the stance leg controller")
        if self._stance_config.qp_solver == 'quadprog':
            self._stance_controller = \
                stance_leg_controller_quadprog.TorqueStanceLegController(
                    robot=self._robot,
                    gait_scheduler=self._gait_scheduler,
                    state_estimator=self._velocity_estimator,
                    desired_speed=self._desired_speed,
                    desired_twisting_speed=self._desired_twisting_speed,
                    desired_com_height=self._desired_com_height,
                    body_mass=self._mpc_body_mass,
                    body_inertia=self._mpc_body_inertia,
                    stance_params=self._stance_config
                )

        elif self._stance_config.qp_solver == 'qpOASES':
            self._stance_controller = \
                stance_leg_controller_mpc.TorqueStanceLegController(
                    robot=self._robot,
                    gait_scheduler=self._gait_scheduler,
                    state_estimator=self._velocity_estimator,
                    desired_speed=self._desired_speed,
                    desired_twisting_speed=self._desired_twisting_speed,
                    desired_com_height=self._desired_com_height,
                    body_mass=self._mpc_body_mass,
                    body_inertia=self._mpc_body_inertia,
                    stance_params=self._stance_config
                )

        else:
            raise RuntimeError("Unspecified objective function for stance controller")

        print("Whole body controller settle down!")

    @property
    def control_thread(self):
        return self._control_thread

    def stop_thread(self):
        self._is_control = False
        # self._control_thread.join()

    def start_thread(self):
        self._is_control = True
        self._control_thread = threading.Thread(target=self.run)
        self._control_thread.start()

    @property
    def swing_leg_controller(self):
        return self._swing_controller

    @property
    def stance_leg_controller(self):
        return self._stance_controller

    @property
    def gait_scheduler(self):
        return self._gait_scheduler

    @property
    def state_estimator(self):
        return self._velocity_estimator

    @property
    def time_since_reset(self):
        return self._time_since_reset

    # def reset_robot(self):
    #     # self._robot.reset(hard_reset=False)
    #     if self._show_gui and not self._use_real_robot:
    #         self.pybullet_client.configureDebugVisualizer(
    #             self.pybullet_client.COV_ENABLE_RENDERING, 1)

    def reset_controllers(self):
        print("Reset robot whole body controller...")
        self._reset_time = self._clock()
        self._time_since_reset = 0
        self._gait_scheduler.reset()
        self._velocity_estimator.reset(self._time_since_reset)
        self._swing_controller.reset(self._time_since_reset)
        self._stance_controller.reset(self._time_since_reset)
        self._is_control = False

    def update(self):
        self._time_since_reset = self._clock() - self._reset_time
        # print(f"self._time_since_reset: {self._time_since_reset}")

        self._gait_scheduler.update()
        self._velocity_estimator.update(self._gait_scheduler.desired_leg_states)
        self._swing_controller.update(self._time_since_reset)
        # future_contact_estimate = self._gait_scheduler.get_estimated_contact_states(
        #     stance_leg_controller_mpc.PLANNING_HORIZON_STEPS,
        #     stance_leg_controller_mpc.PLANNING_TIMESTEP)
        # self._stance_controller.update(self._time_since_reset, future_contact_estimate)
        self._stance_controller.update(self._time_since_reset)
        self._robot_state = self.state_vector

    def get_action(self, phydrl=False, drl_action=None):
        """Returns the control outputs (e.g. positions/torques) for all motors."""

        # Get PhyDRL action (Inference)
        if phydrl is True and drl_action is None:
            # print(f"Getting action from PhyDRL model {self._ddpg_agent.params.model_path}")

            # Observation
            observation = self.tracking_error

            # s_drl = time.time()
            # drl_action = self._ddpg_agent.get_action(observation, mode='test')
            drl_action = self.get_action_from_lite(observations=observation)
            drl_action *= self._action_magnitude

            # print(f"drl_action magnitude: {self._action_magnitude}")
            # print(f"drl_action: {drl_action}")
            # e_drl = time.time()
            # print(f"get drl action time: {e_drl - s_drl}")

        # s = time.time()
        swing_action = self._swing_controller.get_action()
        # e_swing = time.time()
        phy_ddq = self._stance_controller.get_model_action()
        if drl_action is not None:
            self.hp_action = phy_ddq + drl_action
        else:
            self.hp_action = phy_ddq
        self.ha_action, dwell_flag = self.ha_teacher.get_action()
        # print(f"hp_action: {hp_action}")
        # print(f"ha_action: {ha_action}")
        terminal_stance_ddq, action_mode = self.coordinator.get_terminal_action(hp_action=self.hp_action,
                                                                                ha_action=self.ha_action,
                                                                                plant_state=self.tracking_error,
                                                                                dwell_flag=dwell_flag,
                                                                                epsilon=self.ha_teacher.epsilon)
        stance_action, _ = self.stance_leg_controller.map_ddq_to_action(ddq=terminal_stance_ddq)

        motor_action = self.get_motor_action(swing_action=swing_action, stance_action=stance_action)

        # stance_action, qp_sol = self._stance_controller.get_action(drl_action=drl_action)
        # e_stance = time.time()
        # print(f"swing_action time: {e_swing - s}")
        # print(f"stance_action time: {e_stance - e_swing}")
        # print(f"total get_action time: {e_stance - s}")
        qp_sol = None
        return motor_action, dict(qp_sol=qp_sol)

    def _get_stand_action(self):
        return MotorCommand(
            # desired_position=self._robot.motor_group.init_positions,
            desired_position=Pose.STANDING_POSE,
            kp=self._robot.motor_group.kps,
            desired_velocity=0,
            kd=self._robot.motor_group.kds,
            desired_torque=0)

    def _handle_mode_switch(self):
        print("Entering _handle_mode_switch")
        if self._mode == self._desired_mode:
            return
        self._mode = self._desired_mode
        if self._desired_mode == ControllerMode.DOWN:
            logging.info("Entering joint damping mode.")
            self._flush_logging()

        elif self._desired_mode == ControllerMode.STAND:
            logging.info("Standing up.")
            # self.reset_robot()  # Reset the robot if in standing pose

        else:
            logging.info("Walking.")
            # self.reset_controllers()
            self._clear_logging()

    def _clear_logging(self):
        self._logs = []

    def _update_logging(self):
        frame = dict(
            timestamp=self._time_since_reset,
            tracking_error=self.tracking_error,
            desired_speed=(self._swing_controller.desired_speed,
                           self._swing_controller.desired_twisting_speed),
            desired_com_height=self._desired_com_height,
            # step_counter=self._robot.step_counter,
            # action_counter=self._robot.action_counter,
            base_position=self._robot.base_position,
            base_rpy=self._robot.base_orientation_rpy,
            # base_vel=self._robot.motor_velocities,
            base_linear_vel_in_body_frame=self._velocity_estimator.com_velocity_in_body_frame,
            base_angular_vel_in_body_frame=self._robot.base_angular_velocity_in_body_frame,
            motor_angles=self._robot.motor_angles,
            motor_vels=self._robot.motor_velocities,
            motor_torques=self._robot.motor_torques,
            foot_contacts=self._robot.foot_contacts,
            swing_action=self._swing_controller.swing_action,
            stance_action=self._stance_controller.stance_action,
            stance_ddq=self._stance_controller.stance_ddq,
            stance_ddq_limit=self._stance_controller.stance_ddq_limit,
            desired_ground_reaction_forces=self._stance_controller.ground_reaction_forces,
            gait_scheduler_phase=self._gait_scheduler.current_phase.copy(),
            leg_states=self._gait_scheduler.leg_states,
            ground_orientation=self._velocity_estimator.ground_orientation_in_world_frame,
            student_ddq=self.hp_action,
            teacher_ddq=self.ha_action,
        )
        self._logs.append(frame)

    def _flush_logging(self):
        if not os.path.exists(self._logdir):
            os.makedirs(self._logdir)
        filename = 'log_{}.pkl'.format(datetime.now().strftime('%Y_%m_%d_%H_%M_%S'))
        pickle.dump(self._logs, open(os.path.join(self._logdir, filename), 'wb'))
        logging.info("Data logged to: {}".format(os.path.join(self._logdir, filename)))

    def _handle_gait_switch(self):
        print("Entering _handle_gait_switch")
        if self._gait == self._desired_gait:
            return
        if self._desired_gait == GaitType.CRAWL:
            logging.info("Switched to Crawling gait.")
            self._gait_config = crawl.get_config()

        elif self._desired_gait == GaitType.TROT:
            logging.info("Switched  to Trotting gait.")
            self._gait_config = trot.get_config()

        else:
            logging.info("Switched to Fly-Trotting gait.")
            self._gait_config = flytrot.get_config()

        self._gait = self._desired_gait
        self._gait_scheduler.gait_params = self._gait_config.gait_parameters
        self._swing_controller.foot_lift_height = self._gait_config.foot_clearance_max
        self._swing_controller.foot_landing_clearance = \
            self._gait_config.foot_clearance_land

    def run(self):
        # logging.info("Low level thread started...")
        curr_time = time.time()
        # print(f"control_thread: {self.control_thread}")

        while self._is_control:

            # self._handle_mode_switch()
            # self._handle_gait_switch()
            self.update()

            s = self.tracking_error
            # print(f"self._robot_state: {self._robot_state}")
            self.ha_teacher.update(error_state=s)  # Teacher update

            # logging.debug(f"vx: {self._stance_controller.desired_speed}")
            # logging.debug(f"mode is: {self.mode}")
            # time.sleep(1)

            # if self._mode == ControllerMode.DOWN:
            #     pass
            #     # time.sleep(0.1)
            #
            # elif self._mode == ControllerMode.STAND:
            #     action = self._get_stand_action()
            #     self._robot.step(action)
            #     # time.sleep(0.001)

            if self._mode == ControllerMode.WALK:
                s_action = time.time()
                if self._ddpg_agent is not None:
                    # action, qp_sol = self.get_phydrl_action()
                    action, qp_sol = self.get_action(phydrl=True)
                else:
                    action, qp_sol = self.get_action(phydrl=False)
                e_action = time.time()

                # s3 = time.time()
                # Terminal Action by Coordinator
                # logger.debug(f"ha_action: {ha_action}")
                # logger.debug(f"hp_action: {hp_action}")

                # time.sleep(0.001)
                ss = time.time()
                self._robot.step(action)
                ee = time.time()

                # s_log = time.time()
                self._update_logging()
                # e_log = time.time()
                # print(f"log update time: {e_log - s_log}")
                print(f"get action duration: {e_action - s_action}")
                print(f"step duration: {ee - ss}")

            else:
                logging.info("Running loop terminated, exiting...")
                break

            final_time = time.time()
            duration = final_time - curr_time
            curr_time = final_time
            print(f"running times in this loop:{duration}")
            if duration < self._robot.control_timestep:
                compensate_time = self._robot.control_timestep - duration
                time.sleep(compensate_time)
            # print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

    def set_controller_mode(self, mode):
        self._desired_mode = mode

    def set_gait(self, gait):
        self._desired_gait = gait

    @property
    def is_safe(self):
        if self.mode != ControllerMode.WALK:
            return True
        rot_mat = np.array(
            self._robot.pybullet_client.getMatrixFromQuaternion(
                self._velocity_estimator.com_orientation_quaternion_in_ground_frame)).reshape(
            (3, 3))
        up_vec = rot_mat[2, 2]  # Check the body tilting
        base_height = self._robot.base_position[2]  # Check the body height
        print("============================ checking robot safety =============================")
        print(f"up_vec is: {up_vec}")
        print(f"base_height is: {base_height}")
        print("================================================================================")

        return up_vec > 0.85 and base_height > self._robot.safe_height

    def get_motor_action(self,
                         swing_action: Mapping[int, MotorCommand],
                         stance_action: Mapping[int, MotorCommand]
                         ) -> Mapping[Any, Any]:
        """Convert swing/stance leg action to motor action"""
        actions = []
        for joint_id in range(self._robot.num_motors):
            if joint_id in swing_action:
                actions.append(swing_action[joint_id])
            else:
                assert joint_id in stance_action
                actions.append(stance_action[joint_id])

        motor_action = MotorCommand(
            desired_position=[action.desired_position for action in actions],
            kp=[action.kp for action in actions],
            desired_velocity=[action.desired_velocity for action in actions],
            kd=[action.kd for action in actions],
            desired_torque=[
                action.desired_torque for action in actions
            ])

        return motor_action

    @property
    def mode(self):
        return self._mode

    def set_desired_speed(self, desired_lin_speed_ratio, desired_rot_speed_ratio):
        # desired_lin_speed = (
        #     self._gait_config.max_forward_speed * desired_lin_speed_ratio[0],
        #     self._gait_config.max_side_speed * desired_lin_speed_ratio[1],
        #     0,
        # )
        # desired_rot_speed = \
        #     self._gait_config.max_rot_speed * desired_rot_speed_ratio

        if len(desired_lin_speed_ratio) == 3:
            desired_lin_speed = desired_lin_speed_ratio
            desired_lin_speed[2] = 0
        else:
            desired_lin_speed = (
                desired_lin_speed_ratio[0],
                desired_lin_speed_ratio[1],
                0)

        desired_rot_speed = desired_rot_speed_ratio

        self._ref_point[6] = desired_lin_speed[0]
        self._ref_point[7] = desired_lin_speed[1]
        self._ref_point[11] = desired_rot_speed

        # print(f"set desired_lin_speed: {desired_lin_speed}")
        # print(f"set desired_rot_speed: {desired_rot_speed}")

        self._swing_controller.desired_speed = desired_lin_speed
        self._swing_controller.desired_twisting_speed = desired_rot_speed
        self._stance_controller.desired_speed = desired_lin_speed
        self._stance_controller.desired_twisting_speed = desired_rot_speed

    @property
    def state_vector(self):
        com_position = self.state_estimator.com_position_in_ground_frame
        com_velocity = self.state_estimator.com_velocity_in_body_frame
        com_roll_pitch_yaw = np.array(
            self._robot.pybullet_client.getEulerFromQuaternion(
                self.state_estimator.com_orientation_quaternion_in_ground_frame))
        com_roll_pitch_yaw_rate = self._robot.base_angular_velocity_in_body_frame

        state_vector = np.hstack((com_position, com_roll_pitch_yaw, com_velocity, com_roll_pitch_yaw_rate))
        return state_vector

    @property
    def ddpg_agent(self):
        return self._ddpg_agent

    @property
    def robot_state(self):
        return self._robot_state

    @property
    def ref_point(self):
        return self._ref_point

    @property
    def tracking_error(self):
        return self._robot_state - self._ref_point

    def update_logs(self):
        self._update_logging()

    def clear_logs(self):
        self._clear_logging()

    def dump_logs(self):
        self._flush_logging()

    def set_gait_parameters(self, gait_parameters):
        raise NotImplementedError()

    def set_qp_weight(self, qp_weight):
        raise NotImplementedError()

    def set_mpc_mass(self, mpc_mass):
        raise NotImplementedError()

    def set_mpc_inertia(self, mpc_inertia):
        raise NotImplementedError()

    def set_mpc_foot_friction(self, mpc_foot_friction):
        raise NotImplementedError()

    def set_foot_landing_clearance(self, foot_landing_clearance):
        raise NotImplementedError()

    def get_action_from_lite(self, observations):
        # print("entering get_exploitation_action!")
        import time
        s = time.time()

        observations = tf.cast([observations], tf.float16)
        # observations_tensor = tf.expand_dims(observations, 0)
        self.actor_lite.set_tensor(self.lite_input_details[0]['index'], observations)
        self.actor_lite.invoke()
        action_exploitation = self.actor_lite.get_tensor(self.lite_output_details[0]['index'])
        print(f"action_exploitation: {action_exploitation}")
        e = time.time()
        print(f"get exploitation action time:{e - s}")
        action = np.array(np.squeeze(action_exploitation))

        action += self.beta_distribution_noise
        action = np.clip(action, -1.0, 1.0)
        return action
