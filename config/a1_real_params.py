"""Robot Parameters for A1 in Real Plant"""
import numpy as np
from locomotion.robots.motors import MotorControlMode
from config.locomotion.robots.pose import Pose
from config.locomotion.robots.a1_params import A1Params
from config.locomotion.robots.a1_robot_params import A1RobotParams
from config.locomotion.robots.motor_params import MotorGroupParams
from config.locomotion.controllers.swing_params import SwingControllerParams
from config.locomotion.controllers.stance_params import StanceControllerParams


# A1 Params for Real
class A1RealParams:

    def __init__(self):
        self.a1_params = A1RobotParams()
        self.motor_params = MotorGroupParams()
        self.swing_params = SwingControllerParams()
        self.stance_params = StanceControllerParams()
        # A1Params.__init__(self)
        # MotorGroupParams.__init__(self)
        # SwingControllerParams.__init__(self)
        # StanceControllerParams.__init__(self)

        # Simulator settings
        # self.time_step = 0.002
        # self.action_repeat = 1
        # self.reset_time = 3
        # self.num_solver_iterations = 30
        # self.enable_cone_friction = 0
        # self.on_rack = False
        # self.init_rack_position = [0, 0, 1]
        # self.init_position = [0, 0, 0.32]
        self.a1_params.urdf_path = "urdf/a1.urdf"
        self.a1_params.reset_time = 3
        self.a1_params.sync_gui = False  # Whether to sync simulator and real-world action in GUI

        # Motor settings
        self.a1_params.motor_control_mode = MotorControlMode.HYBRID
        self.a1_params.motor_init_position = Pose.LAY_DOWN_POSE
        self.a1_params.motor_init_target_position = Pose.STANDING_POSE

        # Gait settings
        self.a1_params.init_gait_phase = np.array([-0.2 * np.pi, 0, 0, -0.2 * np.pi])  # Stance for a while after start
        self.a1_params.gait_params = [2., np.pi, np.pi, 0, 0.4]  # [freq, theta1, theta2, theta3, theta_swing_cutoff]

        # Stance Leg settings
        self.a1_params.mpc_body_height = 0.24
        self.a1_params.mpc_body_mass = 108 / 9.8
        # self.a1_params.mpc_body_mass = 108 / 9.8
        # self.mpc_body_inertia = np.array((0.017, 0, 0, 0, 0.057, 0, 0, 0, 0.064)) * 10.

        # self.a1_params.mpc_body_inertia = np.array(
        #     (0.027, 0, 0, 0, 0.057, 0, 0, 0, 0.064)) * 5.  # from fast_and_efficient
        # self.a1_params.mpc_body_inertia = np.array([0.068, 0., 0., 0., 0.228, 0., 0., 0., 0.256])  # from locomotion-simulation

        self.a1_params.window_size = 60  # should be [40 - 70]

        self.a1_params.safe_height = 0.12
        self.a1_params.desired_vx = 0.3
        self.a1_params.desired_vy = 0.
        self.a1_params.desired_wz = 0.

        # Stance Controller
        # self.stance_params.objective_function = 'state'   # use state
        self.stance_params.objective_function = 'acceleration'  # use acceleration

        # Original
        # self.stance_params.qp_kp = np.diag((0., 0., 85, 76, 76, 100))
        # self.stance_params.qp_kd = np.diag((24., 20., 20., 24., 24., 27.))
        self.stance_params.qp_kp = np.diag((0., 0., 100., 100., 100., 0.))
        self.stance_params.qp_kd = np.diag((40., 30., 10., 10., 10., 30.))
        # self.stance_params.qp_kp = np.array([[0, 0, 0, 0, 0, 0],
        #                                      [0, 0, 0, 0, 0, 0],
        #                                      [0, 0, 128, 0, 0, 0],
        #                                      [0, 0, 0, 83, -25, -2],
        #                                      [0, 0, 0, -33, 80, 2],
        #                                      [0, 0, 0, 1, 0, 80]])
        # self.stance_params.qp_kd = np.array([[39, 0, 0, 0, 0, 0],
        #                                      [0, 35, 0, 0, 0, 0],
        #                                      [0, 0, 35, 0, 0, 0],
        #                                      [0, 0, 0, 37, -1, -9],
        #                                      [0, 0, 0, -1, 37, 9],
        #                                      [0, 0, 0, 0, 0, 40]])

        # Phydrl
        # self.stance_params.qp_kp = np.array((0.1, 0.1, 100., 100., 100., 100))
        # self.stance_params.qp_kd = np.array((40., 30., 10., 10., 10., 30.))
        # self.stance_params.reg_weight = 1e-4 * 0.001

        self.stance_params.max_ddq = np.array((10., 10., 10., 20., 20., 20.))  # Original
        # self.stance_params.max_ddq = 1.8 * np.array((10., 10., 10., 20., 20., 20.))  # Phydrl

        self.stance_params.friction_coeff = 0.45
        self.stance_params.mpc_weights = (1., 1., 0, 0, 0, 10, 0., 0., .1, .1, .1, .0, 0)
        self.stance_params.acc_weights = np.array([1., 1., 1., 10., 10, 1.])
        # self.stance_params.acc_weights = np.array([1.81666331, 1.81892955, 1.87235756,
        #                                            1.89121405, 1.88585412, 1.88390268])


if __name__ == '__main__':
    a1_sim = A1SimParams()
    print(a1_sim.a1_params.time_step)
