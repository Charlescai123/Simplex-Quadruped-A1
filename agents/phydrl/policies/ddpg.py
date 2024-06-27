import math
import copy
import os
import numpy as np
import tensorflow as tf
from collections import deque
from config.phydrl.ddpg_params import DDPGParams
from config.phydrl.taylor_params import TaylorParams
from agents.phydrl.utils.utils import OrnsteinUhlenbeckActionNoise
from agents.phydrl.networks.network import build_mlp_model, TaylorModel


class DDPGAgent:
    def __init__(self,
                 params: DDPGParams,
                 taylor_params: TaylorParams,
                 shape_observations,
                 shape_action,
                 model_path=None,
                 mode='train'):

        self.params = params
        self.actor = None
        self.actor_target = None
        self.critic = None
        self.critic_target = None
        self.optimizer_actor = None
        self.optimizer_critic = None
        self.taylor_params = taylor_params

        self.exploration_steps = 0

        # Action Buffer
        self.action_buffer = deque([np.zeros(6), np.zeros(6), np.zeros(6)])

        if model_path is not None:
            if self.actor is None:
                self.create_model(shape_observations, shape_action)
            self.load_weights(model_path, mode=mode)
        else:
            self.create_model(shape_observations, shape_action)
            self.hard_update()

        self.add_action_noise = True
        if self.params.action_noise == 'no':
            self.add_action_noise = False
        elif self.params.action_noise == 'OU':
            self.action_noise = OrnsteinUhlenbeckActionNoise(shape_action)
        else:
            raise NotImplementedError(f"{self.params.action_noise} noise is not implemented")

    def save_weights(self, model_save_path):
        print("entering save_weights!")
        self.actor.save_weights(os.path.join(model_save_path, "actor"))
        self.actor_target.save_weights(os.path.join(model_save_path, "actor_target"))
        self.critic.save_weights(os.path.join(model_save_path, "critic"))
        self.critic_target.save_weights(os.path.join(model_save_path, "critic_target"))

    def load_weights(self, model_path, mode='train'):

        self.actor.load_weights(os.path.join(model_path, "actor"))

        if mode == "train":
            self.actor_target.load_weights(os.path.join(model_path, "actor_target"))
            self.critic.load_weights(os.path.join(model_path, "critic"))
            self.critic_target.load_weights(os.path.join(model_path, "critic_target"))

        print("Pretrained weights are loaded")

    def create_model(self, shape_observations, shape_action):
        print("Entering create_model...")
        if self.params.use_taylor_nn:
            self.actor = TaylorModel(self.taylor_params, shape_observations, shape_action, output_activation='tanh')
            self.actor_target = TaylorModel(self.taylor_params, shape_observations, shape_action,
                                            output_activation='tanh')
            self.critic = TaylorModel(self.taylor_params, shape_observations + shape_action, 1, output_activation=None,
                                      taylor_editing=self.params.taylor_editing)
            self.critic_target = TaylorModel(self.taylor_params, shape_observations + shape_action, 1,
                                             output_activation=None, taylor_editing=self.params.taylor_editing)
        else:
            self.actor = build_mlp_model(shape_observations, shape_action, name="actor", output_activation='tanh')
            self.actor_target = \
                build_mlp_model(shape_observations, shape_action, name="actor_target", output_activation='tanh')
            #
            # self.actor = build_mlp_model(shape_observations, shape_action, name="actor")
            # self.actor_target = \
            #     build_mlp_model(shape_observations, shape_action, name="actor_target")

            self.critic = build_mlp_model(shape_observations + shape_action, 1, name="critic")
            self.critic_target = build_mlp_model(shape_observations + shape_action, 1, name="critic_target")
            self.actor.summary()
            self.critic.summary()

    def hard_update(self):
        self.actor_target.set_weights(self.actor.get_weights())
        self.critic_target.set_weights(self.critic.get_weights())

    def soft_update(self):
        soft_alpha = tf.convert_to_tensor(self.params.soft_alpha, dtype=tf.float32)
        self._soft_update(soft_alpha)

    @tf.function
    def _soft_update(self, soft_alpha):
        # Obtain weights directly as tf.Variables
        actor_weights = self.actor.weights
        actor_target_weights = self.actor_target.weights
        critic_weights = self.critic.weights
        critic_target_weights = self.critic_target.weights

        for w_new, w_old in zip(actor_weights, actor_target_weights):
            w_old.assign(w_new * soft_alpha + w_old * (1. - soft_alpha))

        for w_new, w_old in zip(critic_weights, critic_target_weights):
            w_old.assign(w_new * soft_alpha + w_old * (1. - soft_alpha))

    def get_action(self, observations, mode='train'):
        if mode == 'train':
            return self.get_exploration_action(observations)
        else:
            return self.get_exploitation_action(observations)

    def get_exploration_action(self, observations):
        # print("entering get_exploration_action!")
        if self.add_action_noise is False:
            action_noise = 0
        else:
            action_noise = self.action_noise.sample() * self.params.action_noise_factor

        observations_tensor = tf.expand_dims(observations, 0)
        action = tf.squeeze(self.actor(observations_tensor)).numpy()  # squeeze to kill batch_size

        action_saturated = np.clip((action + action_noise), a_min=-1, a_max=1, dtype=float)

        self.exploration_steps += 1
        self.noise_factor_decay()

        return action_saturated

    def get_exploitation_action(self, observations):
        # print("entering get_exploitation_action!")
        import time
        s = time.time()

        observations_tensor = tf.expand_dims(observations, 0)
        action_exploitation = self.actor(observations_tensor)
        print(f"action_exploitation: {action_exploitation}")

        e = time.time()
        print(f"get exploitation action time:{e - s}")
        return tf.squeeze(action_exploitation).numpy()

    def noise_factor_decay(self):
        decay_rate = 0.693 / self.params.action_noise_half_decay_time
        self.params.action_noise_factor *= math.exp(-decay_rate * self.exploration_steps)

    def optimize(self, mini_batch):
        # print("Entering optimize...")
        if self.optimizer_critic is None:
            self.optimizer_critic = tf.keras.optimizers.Adam(learning_rate=self.params.learning_rate_critic)
        if self.optimizer_actor is None:
            self.optimizer_actor = tf.keras.optimizers.Adam(learning_rate=self.params.learning_rate_actor)

        ob1_tf = tf.convert_to_tensor(mini_batch[0], dtype=tf.float32)
        a1_tf = tf.convert_to_tensor(mini_batch[1], dtype=tf.float32)
        r1_tf = tf.convert_to_tensor(mini_batch[2], dtype=tf.float32)
        ob2_tf = tf.convert_to_tensor(mini_batch[3], dtype=tf.float32)
        cra_tf = tf.convert_to_tensor(mini_batch[4], dtype=tf.float32)

        loss_critic = self._optimize_critic(ob1_tf, a1_tf, r1_tf, ob2_tf, cra_tf)
        self._optimize_actor(ob1_tf)
        self.soft_update()
        return loss_critic.numpy().mean()

    @tf.function
    def _optimize_critic(self, ob1, a1, r1, ob2, cra):
        # print("Entering optimize_critic")
        # ---------------------- optimize critic ----------------------
        with tf.GradientTape() as tape:
            a2 = self.actor_target(ob2)

            if self.params.add_target_action_noise:
                action_noise = tf.clip_by_value(
                    tf.random.normal(shape=(self.params.batch_size, 1), mean=0, stddev=0.3),
                    clip_value_min=-0.5, clip_value_max=0.5)
                a2 = tf.clip_by_value((a2 + action_noise), clip_value_min=-1, clip_value_max=1)

            critic_target_input = tf.concat([ob2, a2], axis=-1)
            q_e = self.critic_target(critic_target_input)

            y_exp = r1 + self.params.gamma_discount * q_e * (1 - cra)
            critic_input = tf.concat([ob1, a1], axis=-1)
            y_pre = self.critic(critic_input)

            loss_critic = tf.keras.losses.mean_squared_error(y_exp, y_pre)

        q_grads = tape.gradient(loss_critic, self.critic.trainable_variables)
        self.optimizer_critic.apply_gradients(zip(q_grads, self.critic.trainable_variables))
        return loss_critic

    @tf.function
    def _optimize_actor(self, ob1):
        # print("Entering optimize_actor!")
        with tf.GradientTape() as tape:
            a1_predict = self.actor(ob1)
            critic_input = tf.concat([ob1, a1_predict], axis=-1)
            actor_value = -1 * tf.math.reduce_mean(self.critic(critic_input))
        actor_gradients = tape.gradient(actor_value, self.actor.trainable_variables)
        self.optimizer_actor.apply_gradients(zip(actor_gradients, self.actor.trainable_variables))

    # Firstly get an action from the saved policy and abandon it. The Aim is to warm up the
    # memory cache to speed up following get_action function on devices with RTOS
    def agent_warmup(self):
        observations = np.array([0, 0, 0, 0., 0., 0., 0, 0., 0., 0., 0., 0.])
        self.get_action(observations, mode='test')

    def get_delayed_action(self, drl_action):
        # print(f"add action delay...")
        self.action_buffer.append(copy.deepcopy(drl_action))
        # self.action_buffer = copy.deepcopy(self.action_buffer[1:])
        self.action_buffer.popleft()
        idx = np.random.choice(len(self.action_buffer))
        delayed_drl_action = self.action_buffer[idx]
        return delayed_drl_action
        # print(self.action_buffer)
        # print(drl_action)
        # print("..............")