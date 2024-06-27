"""Predefined poses for A1 Robot"""
from dataclasses import dataclass
import numpy as np

@dataclass
class Pose:
    """Hardcoded Position for 12 Motors on A1 Robot ."""

    # LAY_DOWN_POSE = {'FR': [-0.275, 1.091, -2.7], 'FL': [0.31914, 1.081, -2.72],
    #                  'RR': [-0.299, 1.0584, -2.675], 'RL': [0.28307, 1.083, -2.685]}
    #
    # STANDING_POSE = {'FR': [0, 0.9, -1.8], 'FL': [0, 0.9, -1.8],
    #                  'RR': [0, 0.9, -1.8], 'RL': [0, 0.9, -1.8]}

    LAY_DOWN_POSE = np.array(
        [-0.275, 1.091, -2.7, 0.31914, 1.081, -2.72, -0.299, 1.0584, -2.675, 0.28307, 1.083, -2.685])

    STANDING_POSE = np.array([0, 0.9, -1.8] * 4)