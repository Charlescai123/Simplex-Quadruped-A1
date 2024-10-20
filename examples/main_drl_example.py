"""Running the PhyDRL example to train A1 in simulation

To run the training process:
python -m examples.main_drl  --gpu  --mode=train  --id={your phydrl name}

To run the testing process:
python -m examples.main_drl  --gpu  --mode=test  --id={your phydrl name}  --show_gui

The basic config parameters are loaded from the file: config->a1_phydrl_params.py

"""
import os
import time
import hydra
import tensorflow as tf
from omegaconf import DictConfig

from src.trainer.a1_trainer import A1Trainer
from src.utils.utils import *


def main_train(cfg):
    runner = A1Trainer(cfg)
    runner.train()


def main_test(cfg):
    learner = A1Trainer(cfg)
    learner.test()


@hydra.main(version_base=None, config_path="../config", config_name="base_config.yaml")
def main(cfg: DictConfig):
    # Do not use GPU for training
    if not cfg.general.use_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    # Use GPU for training
    # else:
    #     physical_devices = tf.config.list_physical_devices('GPU')
    #     try:
    #         tf.config.experimental.set_memory_growth(physical_devices[0], True)
    #     except:
    #         exit("GPU allocated failed")

    if cfg.general.mode == 'train':
        main_train(cfg)
    elif cfg.general.mode == 'test':
        main_test(cfg)
    else:
        assert NameError(f'No such mode: {cfg.general.mode}. train or test?')


if __name__ == '__main__':
    start = time.time()
    main()
    end = time.time()
    print(f"total training time: {end - start}")