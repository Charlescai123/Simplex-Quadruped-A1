"""A1 Parameters for Swing Controller"""
import numpy as np


# Swing Controller Parameters
class SwingControllerParams:

    def __init__(self):
        # Robot Foot Swing settings
        self.foot_lift_height = 0.1
        self.foot_landing_clearance = 0.01

        # The position correction coefficients in Raibert's formula.
        self.raibert_kp = np.array([0.01, 0.01, 0.01]) * 3

        # At the end of swing, leave a small clearance to prevent unexpected foot collision.
        self.foot_placement_interval = np.array([0.15, 0.1, 0.05])  # in x, y, z

        # Use Raibert's formula to calc the foot placement
        self.use_raibert_heuristic = True
