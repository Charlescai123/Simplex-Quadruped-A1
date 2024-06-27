"""A1 Parameters for Stance Controller"""
import numpy as np


class StanceControllerParams:

    def __init__(self):
        self.objective_function = 'acceleration'  # 'state' or 'acceleration'

        self.force_dimensions = 3

        self.qp_kp = np.array((0., 0., 100., 100., 100., 0.))
        self.qp_kd = np.array((40., 30., 10., 10., 10., 30.))

        self.max_ddq = np.array((10., 10., 10., 20., 20., 20.))
        self.min_ddq = -1 * self.max_ddq

        # self.friction_coeff = 0.6
        self.friction_coeff = 0.45
        self.reg_weight = 1e-4
        self.mpc_weights = (1., 1., 0, 0, 0, 10, 0., 0., .1, .1, .1, .0, 0)
        self.acc_weights = np.array([1., 1., 1., 10., 10, 1.])

        # These weights also give good results.
        # _MPC_WEIGHTS = (1., 1., 0, 0, 0, 20, 0., 0., .1, 1., 1., .0, 0.)

        self.planning_horizon_steps = 10
        self.planning_timestep = 0.025
