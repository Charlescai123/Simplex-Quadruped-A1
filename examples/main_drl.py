"""Running the PhyDRL example to train A1 in simulation

To run the training process:
python -m examples.main_drl  --gpu  --mode=train  --id={your phydrl name}

To run the testing process:
python -m examples.main_drl  --gpu  --mode=test  --id={your phydrl name}  --show_gui

The basic config parameters are loaded from the file: config->a1_phydrl_params.py

"""
import os
import argparse
import tensorflow as tf
from locomotion import wbc_controller
from agents.phydrl.learners.a1_leaner import A1Learner
from config.a1_phydrl_params import A1PhyDRLParams
from utils import *


def main_train(p):
    learner = A1Learner(p)
    learner.train()


def main_test(p):
    learner = A1Learner(p)
    learner.test()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', action='store_true', help='Activate usage of GPU')
    parser.add_argument('--generate_config', action='store_true', help='Enable to write default config only')
    # parser.add_argument('--config', default='./config/local_ddpgips.json', help='Path to config file')
    parser.add_argument('--id', default=None, help='If set overrides the logfile name and the save name')
    parser.add_argument('--force', action='store_true', help='Override log file without asking')
    parser.add_argument('--weights', default=None, help='Path to pretrained weights')
    # parser.add_argument('--params', nargs='*', default=None)
    parser.add_argument('--mode', default='train', help='Choose the mode, train or test')
    parser.add_argument('--show_gui', action='store_true', help='Whether to show GUI during training/testing')

    args = parser.parse_args()

    if args.generate_config:
        generate_config(Params(), "config/train.json")
        exit("config file generated")

    if not args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    else:
        physical_devices = tf.config.list_physical_devices('GPU')
        try:
            tf.config.experimental.set_memory_growth(physical_devices[0], True)
        except:
            exit("GPU allocated failed")

    # params = read_config(args.config)
    params = A1PhyDRLParams()

    print(params.logger_params.mode)

    # if args.params is not None:
    #     params = override_params(params, args.params)
    if args.show_gui:
        params.trainer_params.show_gui = True

    if args.id is not None:
        params.logger_params.model_name = args.id
        if args.mode == 'test':
            params.agent_params.model_path = f'models/{args.id}'

    if args.force:
        params.logger_params.force_override = True

    if args.weights is not None:
        params.agent_params.model_path = args.weights

    params.logger_params.mode = args.mode

    if args.mode == 'train':
        main_train(params)

    elif args.mode == 'test':

        if params.agent_params.model_path is None:
            exit("Please load the pretrained weights")
        else:
            main_test(params)
    else:
        assert NameError('No such mode. train or test?')
