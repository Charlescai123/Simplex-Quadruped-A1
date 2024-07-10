import os
import tensorflow as tf
import numpy as np
import shutil
import distutils.util
import matplotlib.pyplot as plt
from omegaconf import OmegaConf, DictConfig


class Logger:
    def __init__(self, params: DictConfig):

        self.params = params
        self.log_dir = self.params.log_dir
        self.model_dir = self.params.model_save_dir
        self.plot_dir = self.params.plot_dir

        if self.params.mode == 'train':
            self.clear_cache()
        self.training_log_writer = tf.summary.create_file_writer(self.log_dir + '/training')
        self.evaluation_log_writer = tf.summary.create_file_writer(self.log_dir + '/eval')

    def log_training_data(self, average_reward, average_distance_score, critic_loss, failed, global_steps):
        with self.training_log_writer.as_default():
            tf.summary.scalar('train_eval/Average_Reward', average_reward, global_steps)
            tf.summary.scalar('train_eval/Distance_score', average_distance_score, global_steps)
            tf.summary.scalar('train_eval/Critic_loss', critic_loss, global_steps)
            tf.summary.scalar('train_eval/Distance_score_and_survived', average_distance_score * (1 - failed),
                              global_steps)

    def log_evaluation_data(self, average_reward, average_distance_score, failed, global_steps):
        with self.evaluation_log_writer.as_default():
            tf.summary.scalar('train_eval/Average_Reward', average_reward, global_steps)
            tf.summary.scalar('train_eval/Distance_score', average_distance_score, global_steps)
            tf.summary.scalar('train_eval/Distance_score_and_survived', average_distance_score * (1 - failed),
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

    def plot_reward(self, rewards, alpha=0.99, fig_name="demo.png"):

        steps = np.arange(len(rewards))
        rewards = np.asarray(rewards)
        rewards = np.squeeze(rewards)
        rewards = rewards / (alpha - 1)

        fig, axes = plt.subplots()

        axes.plot(steps, rewards, label='reward')
        axes.set_xlabel('Steps')
        axes.set_ylabel('r / (alpha - 1)')
        axes.set_title('Reward Plot')
        axes.legend()

        # Save figure
        if not os.path.exists(self.plot_dir):
            print(f"plot_dir {self.plot_dir} does not exist, creating...")
            os.makedirs(self.plot_dir)
        plt.savefig(f"{self.plot_dir}/{fig_name}")


    def plot_acceleration(self, rewards, alpha=0.99, fig_name="demo.png"):

        steps = np.arange(len(rewards))
        rewards = np.asarray(rewards)
        rewards = np.squeeze(rewards)
        rewards = rewards / (alpha - 1)

        fig, axes = plt.subplots()

        axes.plot(steps, rewards, label='reward')
        axes.set_xlabel('Steps')
        axes.set_ylabel('r/(alpha - 1)')
        axes.set_title('Reward Plot')
        axes.legend()

        # Save figure
        if not os.path.exists(self.plot_dir):
            print(f"plot_dir {self.plot_dir} does not exist, creating...")
            os.makedirs(self.plot_dir)
        plt.savefig(f"{self.plot_dir}/{fig_name}")


def test(rewards, alpha=0.99):
    steps = np.arange(len(rewards))
    rewards = np.asarray(rewards)
    rewards = np.squeeze(np.asarray(rewards))
    rewards = rewards / (alpha - 1)

    fig, axes = plt.subplots()

    axes.plot(steps, rewards, label='reward')
    axes.set_xlabel('Steps')
    axes.set_ylabel('r/(alpha - 1)')
    axes.set_title('Reward Plot')
    axes.legend()

    # Save figure
    path = "saved/plots/Demo"
    if not os.path.exists(path):
        print(f"plot_dir {path} not exist, creating...")
        os.makedirs(path)
    plt.savefig(f"{path}/demo.png")
    plt.plot()


if __name__ == '__main__':
    rewards = [np.array([[-1]]), np.array([[-2]]), np.array([[-3]])]
    print(rewards)
    test(rewards)