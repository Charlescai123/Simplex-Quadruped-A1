import os
import tensorflow as tf
import numpy as np
import shutil
import distutils.util
import matplotlib.pyplot as plt
from config.phydrl.logger_params import LoggerParams


class Logger:
    def __init__(self, params: LoggerParams):

        self.params = params
        self.log_dir = 'logs/' + self.params.model_name
        self.model_dir = 'models/' + self.params.model_name
        if self.params.mode == 'train':
            self.clear_cache()
        self.training_log_writer = tf.summary.create_file_writer(self.log_dir + '/training')
        self.evaluation_log_writer = tf.summary.create_file_writer(self.log_dir + '/eval')

    def log_training_data(self, average_reward, average_distance_score, critic_loss, failed, global_steps):
        with self.training_log_writer.as_default():
            tf.summary.scalar('train_eval/Average_Reward', average_reward, global_steps)
            tf.summary.scalar('train_eval/distance_score', average_distance_score, global_steps)
            tf.summary.scalar('train_eval/critic_loss', critic_loss, global_steps)
            tf.summary.scalar('train_eval/distance_score_and_survived', average_distance_score * (1 - failed),
                              global_steps)

    def log_evaluation_data(self, average_reward, average_distance_score, failed, global_steps):
        with self.evaluation_log_writer.as_default():
            tf.summary.scalar('train_eval/Average_Reward', average_reward, global_steps)
            tf.summary.scalar('train_eval/distance_score', average_distance_score, global_steps)
            tf.summary.scalar('train_eval/distance_score_and_survived', average_distance_score * (1 - failed),
                              global_steps)

    def clear_cache(self):
        if os.path.isdir(self.log_dir):
            if self.params.force_override:
                shutil.rmtree(self.log_dir)
            else:
                print(self.log_dir, 'already exists.')
                resp = input('Override log file? [Y/n]\n')
                # resp = ''
                if resp == '' or distutils.util.strtobool(resp):
                    print('Deleting old log dir')
                    shutil.rmtree(self.log_dir)
                else:
                    print('Okay bye')
                    exit(1)


def plot_trajectory(trajectory_tensor, reference_trajectory_tensor=None):
    """
   trajectory_tensor: a numpy array [n, 4], where n is the length of the trajectory,
                       5 is the dimension of each point on the trajectory, containing [x, x_dot, theta, theta_dot]
   """
    trajectory_tensor = np.array(trajectory_tensor)
    reference_trajectory_tensor = np.array(
        reference_trajectory_tensor) if reference_trajectory_tensor is not None else None
    n, c = trajectory_tensor.shape

    y_label_list = ["x", "x_dot", "theta", "theta_dot"]

    plt.figure(figsize=(9, 6))

    for i in range(c):

        plt.subplot(c, 1, i + 1)
        plt.plot(np.arange(n), trajectory_tensor[:, i], label=y_label_list[i])

        if reference_trajectory_tensor is not None:
            plt.plot(np.arange(n), reference_trajectory_tensor[:, i], label=y_label_list[i])

        plt.legend(loc='best')
        plt.grid(True)

    plt.tight_layout()
    plt.savefig("trajectory.png", dpi=300)
    # plt.show()
