import time
import copy
import math
import numpy as np
from tqdm import tqdm
from omegaconf import DictConfig

from src.ha_teacher.ha_teacher import HATeacher
from src.coordinator.coordinator import Coordinator
from src.logger.logger import Logger
from src.hp_student.agents.replay_mem import ReplayMemory
from src.hp_student.agents.ddpg import DDPGAgent
from src.envs.a1_envs import A1Envs
from src.utils.utils import ActionMode


class A1Trainer:
    def __init__(self, config: DictConfig):
        self.params = config

        # HP Student
        self.agent_params = config.hp_student.agents
        self.shape_observations = 12
        self.shape_action = 6
        self.replay_mem = ReplayMemory(config.hp_student.agents.replay_buffer.buffer_size)
        self.agent = DDPGAgent(agent_cfg=config.hp_student.agents,
                               taylor_cfg=config.hp_student.taylor,
                               shape_observations=self.shape_observations,
                               shape_action=self.shape_action,
                               mode=config.logger.mode)
        self.agent.agent_warmup()

        # Environment (Real Plant)
        self.a1_env = A1Envs(a1_envs_cfg=config.envs, agent=self.agent)

        # HA Teacher
        self.ha_teacher = HATeacher(robot=self.a1_env.robot, teacher_cfg=config.ha_teacher)

        # Coordinator
        self.coordinator = Coordinator(config=config.coordinator)

        # Logger
        self.logger = Logger(config.logger)

        self.failed_times = 0

        # Cached variables for real time efficiency when indexing
        self._teacher_learn = self.params.coordinator.teacher_learn
        self._action_magnitude = np.asarray(self.agent_params.action.magnitude)
        self._random_reset_eval = self.agent_params.random_reset.eval
        self._random_reset_train = self.agent_params.random_reset.train
        self._max_steps_per_episode = int(self.agent_params.max_steps_per_episode)
        self._max_training_episodes = int(self.agent_params.max_training_episodes)
        self._evaluation_period = int(self.agent_params.evaluation_period)
        self._buffer_experience_prefill_size = self.agent_params.replay_buffer.experience_prefill_size
        self._buffer_batch_size = self.agent_params.replay_buffer.batch_size


    def interaction_step(self, mode=None):

        s0 = time.time()
        observations = self.a1_env.locomotion_controller.tracking_error
        s = np.asarray(observations)

        s1 = time.time()
        self.ha_teacher.update(error_state=s)  # Teacher update
        self.coordinator.update(state=s)  # Coordinator update
        s2 = time.time()

        # self.a1_env.mpc_controller.set_desired_speed(self.target_lin_speed, self.target_ang_speed)
        # self.a1_env.locomotion_controller.update()  # update the clock

        motor_action, action_mode, nominal_action = self.get_terminal_action(state=s, mode=mode)
        s3 = time.time()

        # Inject Terminal Action
        _, termination, abort = self.a1_env.env_step(motor_action)
        s4 = time.time()

        observations_next = self.a1_env.tracking_error
        s_next = np.asarray(observations_next)

        reward = self.a1_env.get_reward(s=s, s_next=s_next)
        s5 = time.time()

        # print(f"interaction part1 time: {s1 - s0}")
        # print(f"interaction part2 time: {s2 - s1}")
        # print(f"interaction part3 time: {s3 - s2}")
        # print(f"interaction part4 time: {s4 - s3}")
        # print(f"interaction part5 time: {s5 - s4}")

        return observations, nominal_action, observations_next, termination, reward, abort

    def evaluation(self, reset_states=None, mode=None):

        if self._random_reset_eval:
            self.a1_env.random_reset()
        else:
            self.a1_env.reset(reset_states)

        reward_list = []
        distance_score_list = []
        failed = False

        for step in range(self._max_steps_per_episode):
            observations, action, observations_next, failed, r, abort = \
                self.interaction_step(mode='eval')

            reward_list.append(r)

            if failed or abort:
                break

        if len(reward_list) == 0:
            mean_reward = math.nan
            mean_distance_score = math.nan
        else:
            mean_reward = np.mean(reward_list)
            mean_distance_score = np.mean(distance_score_list)

        return mean_reward, mean_distance_score, failed

    def train(self):
        ep = 0
        global_steps = 0
        best_dsas = 0  # Best distance score and survived
        moving_average_dsas = 0.0

        # while global_steps < int(self.agent_params.max_training_episodes):
        for ep_i in range(self._max_training_episodes):
            pbar = tqdm(total=self._max_steps_per_episode, desc="Iteration %d" % ep)

            if self._random_reset_train:
                self.a1_env.random_reset()
            else:
                self.a1_env.reset(step=global_steps)

            ep += 1
            reward_list = []
            critic_loss_list = []

            failed = False
            ep_steps = 0

            for step in range(self._max_steps_per_episode):

                s = time.time()
                observations, action, observations_next, failed, reward, abort = \
                    self.interaction_step(mode='train')
                e = time.time()
                print(f"interaction_step_time: {e - s}")
                # print(f"observations: {observations}")
                # print(f"action is: {action}")
                # print(f"reward is: {reward}")

                # time.sleep(1)

                if abort or failed:
                    print("robot failed, break this loop...")
                    break

                self.replay_mem.add((observations, action, reward, observations_next, failed))

                reward_list.append(reward)

                if self.replay_mem.get_size() > self._buffer_experience_prefill_size:
                    minibatch = self.replay_mem.sample(self._buffer_batch_size)
                    critic_loss = self.agent.optimize(minibatch)
                else:
                    critic_loss = 100

                critic_loss_list.append(critic_loss)
                global_steps += 1
                ep_steps += 1

                pbar.update(1)  # Update the progress bar

            if len(reward_list) == 0:
                continue
            else:
                mean_reward = np.mean(reward_list)
                mean_critic_loss = np.mean(critic_loss_list)

            self.logger.log_training_data(mean_reward, 0, mean_critic_loss, failed, global_steps)
            print(f"Training at {ep} episodes: average_reward: {mean_reward:.6},"
                  f"critic_loss: {mean_critic_loss:.6}, total_steps_ep: {ep_steps} ")

            if ep % self._evaluation_period == 0:
                eval_mean_reward, eval_mean_distance_score, eval_failed = self.evaluation()
                self.logger.log_evaluation_data(eval_mean_reward, eval_mean_distance_score, eval_failed,
                                                global_steps)
                moving_average_dsas = 0.95 * moving_average_dsas + 0.05 * eval_mean_distance_score

                if moving_average_dsas > best_dsas:
                    self.agent.save_weights(self.logger.model_dir + '_best')
                    best_dsas = moving_average_dsas

            self.agent.save_weights(self.logger.model_dir)

    def test(self):
        self.evaluation(mode='test')

    def loop_update(self):
        pass

    def get_terminal_action(self, state: np.ndarray, mode=None):
        observations = state

        s0 = time.time()
        # DRL Action
        drl_raw_action = self.agent.get_action(observations, mode)  # All values from [-1, 1]
        drl_action = drl_raw_action * self._action_magnitude
        # drl_action = np.zeros(6)
        # drl_raw_action = np.zeros(6)
        s1 = time.time()

        # Student Action (Residual form)
        phy_action = self.a1_env.locomotion_controller.stance_leg_controller.get_model_action()
        hp_action = drl_action + phy_action
        # hp_action = phy_action
        # hp_action = drl_action * self.gamma + phy_action * (1 - self.gamma)
        s2 = time.time()

        # Teacher Action
        ha_action = self.ha_teacher.get_action()
        s3 = time.time()
        # Terminal Action by Coordinator
        # logger.debug(f"ha_action: {ha_action}")
        # logger.debug(f"hp_action: {hp_action}")
        terminal_stance_ddq, action_mode = self.coordinator.determine_action(hp_action=hp_action, ha_action=ha_action,
                                                                             epsilon=self.ha_teacher.epsilon)
        s4 = time.time()

        # Decide nominal action to store into replay buffer
        if action_mode == ActionMode.TEACHER:
            if self._teacher_learn:  # Learn from teacher action
                nominal_action = (ha_action - phy_action) / self._action_magnitude
            else:
                nominal_action = drl_raw_action
        elif action_mode == ActionMode.STUDENT:
            nominal_action = drl_raw_action
        else:
            raise NotImplementedError(f"Unknown action mode: {action_mode}")

        s5 = time.time()

        stance_action, _ = self.a1_env.locomotion_controller.stance_leg_controller.map_ddq_to_action(
            ddq=terminal_stance_ddq)
        s51 = time.time()
        swing_action = self.a1_env.locomotion_controller.swing_leg_controller.get_action()
        s52 = time.time()
        motor_action = self.a1_env.locomotion_controller.get_motor_action(swing_action=swing_action,
                                                                          stance_action=stance_action)
        s6 = time.time()
        # print(f"get_terminal_action part1 time: {s1 - s0}")
        # print(f"get_terminal_action part2 time: {s2 - s1}")
        # print(f"get_terminal_action part3 time: {s3 - s2}")
        # print(f"get_terminal_action part4 time: {s4 - s3}")
        # print(f"get_terminal_action part5 time: {s5 - s4}")
        # print(f"get_terminal_action part6 time: {s6 - s5}")
        # print(f"get_terminal_action total time: {s6 - s0}")

        # print(f"terminal_stance_ddq: {terminal_stance_ddq}")
        # print(f"swing_action: {swing_action}")
        return motor_action, action_mode, nominal_action
