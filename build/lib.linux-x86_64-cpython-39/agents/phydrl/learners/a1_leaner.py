import time
import numpy as np
import copy
import math
from tqdm import tqdm
from config.phydrl.env_params import TrainerEnvParams
from config.phydrl.logger_params import LoggerParams
from config.phydrl.ddpg_params import DDPGParams
from config.phydrl.taylor_params import TaylorParams
from config.a1_phydrl_params import A1PhyDRLParams
from agents.phydrl.utils.logger import Logger, plot_trajectory
from agents.phydrl.replay_buffers.replay_memory import ReplayMemory
from agents.phydrl.policies.ddpg import DDPGAgent
from agents.phydrl.envs.a1_trainer_env import A1TrainerEnv


class A1Learner:
    def __init__(self, params: A1PhyDRLParams):
        self.params = params

        self.a1_env = A1TrainerEnv(self.params.trainer_params)
        self.shape_observations = self.a1_env.states_observations_dim
        self.shape_action = self.a1_env.action_dim
        self.replay_mem = ReplayMemory(self.params.agent_params.replay_buffer_size)

        self.agent = DDPGAgent(self.params.agent_params,
                               self.params.taylor_params,
                               shape_observations=self.shape_observations,
                               shape_action=self.shape_action,
                               model_path=self.params.agent_params.model_path,
                               mode=self.params.logger_params.mode)

        self.logger = Logger(self.params.logger_params)

    def interaction_step(self, mode=None):

        # observations = copy.deepcopy(self.a1_env.observation)
        s = time.time()
        observations = copy.deepcopy(self.a1_env.get_tracking_error())
        e1 = time.time()

        action = self.agent.get_action(observations, mode)
        e2 = time.time()

        _, terminal, abort = self.a1_env.step(action, action_mode=self.params.agent_params.action_mode)
        # import time
        # time.sleep(1)
        e3 = time.time()

        observations_next = self.a1_env.get_tracking_error()
        e4 = time.time()

        r = self.a1_env.get_reward()
        e5 = time.time()

        return observations, action, observations_next, terminal, r, abort

    def evaluation(self, reset_states=None, mode=None):

        if self.params.trainer_params.random_reset_eval:
            self.a1_env.random_reset()
        else:
            self.a1_env.reset(reset_states)

        reward_list = []
        distance_score_list = []
        failed = False

        for step in range(self.params.agent_params.max_episode_steps):
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

        # with tqdm(total=int(self.params.agent_params.training_episode),
        #           desc='Iteration %d' % progress) as pbar:

        while global_steps < self.params.agent_params.training_episode:

            pbar = tqdm(total=self.params.agent_params.max_episode_steps, desc="Iteration %d" % ep)

            if self.params.trainer_params.random_reset_train:
                self.a1_env.random_reset()
            else:
                # print("a1 resetting!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                # print(global_steps)
                # import time
                # time.sleep(12)
                self.a1_env.reset(step=global_steps)

            ep += 1
            reward_list = []
            critic_loss_list = []

            failed = False
            ep_steps = 0

            for step in range(self.params.agent_params.max_episode_steps):

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

                if self.replay_mem.get_size() > self.params.agent_params.experience_prefill_size:
                    minibatch = self.replay_mem.sample(self.params.agent_params.batch_size)
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

            if ep % self.params.logger_params.evaluation_period == 0:
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
