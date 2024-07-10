"""Set up the zeroth-order QP problem for stance leg control.

For details, please refer to this paper:
https://arxiv.org/abs/2009.10019
"""

import time
import numpy as np
import quadprog  # pytype:disable=import-error

np.set_printoptions(precision=3, suppress=True)


# ACC_WEIGHT = np.array([1., 1., 1., 10., 10, 1.])


class QPTorqueOptimizer:
    """QP Torque Optimizer Class."""

    def __init__(self,
                 robot_mass,
                 robot_inertia,
                 friction_coef=0.45,
                 f_min_ratio=0.1,
                 f_max_ratio=10.):
        self.mpc_body_mass = robot_mass
        self.inv_mass = np.eye(3) / robot_mass

        # robot_inertia = np.array([0.068, 0., 0., 0., 0.228, 0., 0., 0., 0.256])  # from robot-simulation
        #   robot_inertia = np.array(
        # (0.027, 0, 0, 0, 0.057, 0, 0, 0, 0.064)) * 5.

        self.inv_inertia = np.linalg.inv(robot_inertia.reshape((3, 3)))
        self.friction_coef = friction_coef
        self.f_min_ratio = f_min_ratio
        self.f_max_ratio = f_max_ratio

        # print(f"QPTorqueOptimizer robot:")
        # print(self.mpc_body_mass)
        # print(robot_inertia)
        # print(self.inv_mass)
        # print(self.inv_inertia)
        # print(self.friction_coef)
        # print(self.f_min_ratio)
        # print(self.f_max_ratio)
        # import time
        # time.sleep(12)

        # Precompute constraint matrix A
        self.A = np.zeros((24, 12))
        for leg_id in range(4):
            self.A[leg_id * 2, leg_id * 3 + 2] = 1
            self.A[leg_id * 2 + 1, leg_id * 3 + 2] = -1

        # Friction constraints
        for leg_id in range(4):
            row_id = 8 + leg_id * 4
            col_id = leg_id * 3
            self.A[row_id, col_id:col_id + 3] = np.array([1, 0, self.friction_coef])
            self.A[row_id + 1,
            col_id:col_id + 3] = np.array([-1, 0, self.friction_coef])
            self.A[row_id + 2,
            col_id:col_id + 3] = np.array([0, 1, self.friction_coef])
            self.A[row_id + 3,
            col_id:col_id + 3] = np.array([0, -1, self.friction_coef])

    def compute_mass_matrix(self, foot_positions):
        mass_mat = np.zeros((6, 12))
        mass_mat[:3] = np.concatenate([self.inv_mass] * 4, axis=1)

        for leg_id in range(4):
            x = foot_positions[leg_id]
            foot_position_skew = np.array([[0, -x[2], x[1]], [x[2], 0, -x[0]],
                                           [-x[1], x[0], 0]])
            mass_mat[3:6, leg_id * 3:leg_id * 3 + 3] = \
                self.inv_inertia.dot(foot_position_skew)
        return mass_mat

    def compute_constraint_matrix(self, contacts):
        f_min = self.f_min_ratio * self.mpc_body_mass * 9.8
        f_max = self.f_max_ratio * self.mpc_body_mass * 9.8
        lb = np.ones(24) * (-1e-7)
        contact_ids = np.nonzero(contacts)[0]
        lb[contact_ids * 2] = f_min
        lb[contact_ids * 2 + 1] = -f_max
        return self.A.T, lb

    def compute_objective_matrix(self, mass_matrix, desired_acc, acc_weights,
                                 reg_weight):
        s = time.time()

        g = np.array([0., 0., 9.8, 0., 0., 0.])
        e11 = time.time()
        Q = np.diag(acc_weights)
        e12 = time.time()

        R = np.ones(12) * reg_weight
        e13 = time.time()
        # R = np.ones(12) * reg_weight * 0.001

        e1 = time.time()

        quad_term = mass_matrix.T.dot(Q).dot(mass_matrix) + R

        e2 = time.time()

        linear_term = 1 * (g + desired_acc).T.dot(Q).dot(mass_matrix)

        e3 = time.time()
        # print(f"compute_objective time 1: {e1 - s}")
        # print(f"compute_objective time 11: {e11 - s}")
        # print(f"compute_objective time 12: {e12 - e11}")
        # print(f"compute_objective time 13: {e13 - e12}")
        # print(f"compute_objective time 2: {e2 - e1}")
        # print(f"compute_objective time 3: {e3 - e2}")

        # g = np.array([0., 0., 9.8, 0., 0., 0.])
        # Q = np.diag(acc_weights)
        # R = np.ones(12) * reg_weight
        # # R = np.ones(12) * reg_weight * 0.001
        #
        # quad_term = mass_matrix.T.dot(Q).dot(mass_matrix) + R
        # linear_term = 1 * (g + desired_acc).T.dot(Q).dot(mass_matrix)
        # print(f"g: {g}")
        # print(f"Q: {Q}")
        # print(f"R: {R}")
        # print(f"quad_term: {quad_term}")
        # print(f"linear_term: {linear_term}")
        return quad_term, linear_term

    def compute_contact_force(self,
                              foot_positions,
                              desired_acc,
                              contacts,
                              acc_weights,
                              reg_weight=1e-4):

        mass_matrix = self.compute_mass_matrix(foot_positions)
        G, a = self.compute_objective_matrix(mass_matrix, desired_acc, acc_weights,
                                             reg_weight)
        C, b = self.compute_constraint_matrix(contacts)
        G += 1e-4 * np.eye(12)

        result = quadprog.solve_qp(G, a, C, b)

        # print(f"contacts: {contacts}")
        # print(f"mpc_body_mass: {self.mpc_body_mass}")
        # print(f"G: {G}")
        # print(f"a: {a}")
        # print(f"C: {C}")
        # print(f"b: {b}")
        # print(f"result: {result}")

        return -result[0].reshape((4, 3))
