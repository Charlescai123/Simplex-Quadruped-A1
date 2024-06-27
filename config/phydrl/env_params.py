"""Robot Parameters for Trainer"""
import numpy as np
from locomotion.robots.motors import MotorControlMode
from config.locomotion.robots.pose import Pose
from config.a1_sim_params import A1SimParams
from config.a1_real_params import A1RealParams


# A1 Params for PhyDRL Training
class TrainerEnvParams(A1RealParams):

    def __init__(self):
        super().__init__()

        self.ref_vx = 0.3
        self.ref_px = 0
        self.ref_pz = 0.24

        self.show_gui = False
        self.use_real_urdf = True

        if self.use_real_urdf:           # Whether to use a more realistic urdf file for training
            self.a1_params.urdf_path = "a1.urdf"
        else:
            self.a1_params.urdf_path = "a1/a1.urdf"

        # control frequency = 500Hz
        self.a1_params.time_step = 0.001
        self.a1_params.action_repeat = 2
        # self.a1_params.time_step = 0.002
        # self.a1_params.action_repeat = 1

        self.a1_params.sync_gui = False  # Whether to sync simulator and real-world action in GUI
        self.a1_params.reset_time = 0

        # Motor settings
        self.a1_params.motor_control_mode = MotorControlMode.HYBRID
        self.a1_params.motor_init_position = Pose.STANDING_POSE
        self.a1_params.motor_init_target_position = Pose.STANDING_POSE

        # Gait settings
        self.a1_params.init_gait_phase = np.array([-0.2 * np.pi, 0, 0, -0.2 * np.pi])  # Stance for a while after start
        self.a1_params.gait_params = [2., np.pi, np.pi, 0, 0.4]  # [freq, theta1, theta2, theta3, theta_swing_cutoff]

        # Stance Leg settings
        self.a1_params.mpc_body_height = self.ref_pz
        # self.mpc_body_mass = 110 / 9.8
        self.a1_params.mpc_body_mass = 108 / 9.8
        # # self.mpc_body_inertia = np.array((0.017, 0, 0, 0, 0.057, 0, 0, 0, 0.064)) * 10.
        #
        # self.a1_params.mpc_body_inertia = np.array(
        #     (0.027, 0, 0, 0, 0.057, 0, 0, 0, 0.064)) * 5.  # from fast_and_efficient (more stable in simulation)

        self.a1_params.mpc_body_inertia = np.array(
            (0.017, 0, 0, 0, 0.057, 0, 0, 0, 0.064)) * 4.  # from locomotion-simulation

        self.a1_params.window_size = 20
        self.a1_params.safe_height = 0.12

        self.a1_params.desired_vx = self.ref_vx
        self.a1_params.desired_vy = 0
        self.a1_params.desired_wz = 0

        # Phydrl for Stance Control
        self.stance_params.qp_kp = np.array((0.1, 0.1, 100., 100., 100., 100))
        self.stance_params.qp_kd = np.array((40., 30., 10., 10., 10., 30.))
        self.stance_params.max_ddq = 1.8 * np.array((10., 10., 10., 20., 20., 20.))
        self.stance_params.min_ddq = -1 * self.stance_params.max_ddq

        # self.friction_coeff = 0.6
        self.stance_params.objective_function = 'acceleration'  # 'state' or 'acceleration'
        self.stance_params.friction_coeff = 0.45
        # self.stance_params.reg_weight = 1e-4 * 0.001
        self.stance_params.reg_weight = 1e-4
        self.stance_params.mpc_weights = (1., 1., 0, 0, 0, 10, 0., 0., .1, .1, .1, .0, 0)
        self.stance_params.acc_weights = np.array(
            [1.81666331, 1.81892955, 1.87235756, 1.89121405, 1.88585412, 1.88390268])

        # Parameters for learners itself
        self.if_add_terrain = False
        self.random_reset_eval = False
        self.random_reset_train = False
        self.if_record_video = False
        self.action_magnitude = 1

        self.fall_threshold = 0.12
        # self.friction = 0.44          # Original icy road
        self.friction = 0.7


if __name__ == '__main__':
    a1_train = TrainParams()
    print(a1_train.a1_params.time_step)
