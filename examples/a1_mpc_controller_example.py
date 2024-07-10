"""Example of MPC controller on A1 robot.

To run:
python -m examples.a1_mpc_controller_example   --show_gui=[True|False]   --max_time_secs=[xxx] \
        --use_real_robot=[True|False]   --world=[plane|slope|stair|uneven]   --camera_fixed=[True|False]

For example, to run in simulation:
python -m examples.a1_mpc_controller_example --show_gui=True --max_time_secs=20 --use_real_robot=False --world=plane

For example, to run in real A1:
python -m examples.a1_mpc_controller_example --show_gui=False --max_time_secs=15 --use_real_robot=True
"""

import time
import numpy as np
from absl import app
from absl import flags
from absl import logging

import pybullet
from pybullet_utils import bullet_client
from src.envs.locomotion import locomotion_controller
from src.envs.locomotion.locomotion_controller import ControllerMode
from src.envs.locomotion.locomotion_controller import GaitType
from src.envs.simulator.worlds import uneven_world
from src.envs.simulator.worlds import slope_world, plane_world, stair_world
from src.envs.locomotion.robots import a1, a1_robot
from src.envs.locomotion import MatEngine
from config_json.locomotion.controllers.swing_params import SwingControllerParams
from config_json.locomotion.controllers.stance_params import StanceControllerParams
from config_json.locomotion.robots.motor_params import MotorGroupParams
from config_json.locomotion.robots.a1_params import A1Params
from config_json.a1_real_params import A1RealParams
from config_json.locomotion.robots.a1_robot_params import A1RobotParams
from config_json.phydrl.ddpg_params import DDPGParams
from config_json.phydrl.taylor_params import TaylorParams
from config_json.a1_sim_params import A1SimParams
from src.ha_teacher import ha_teacher
from src.hp_student.agents import ddpg

flags.DEFINE_string("logdir", "logs", "where to log trajectories.")
flags.DEFINE_bool("use_real_robot", False,
                  "whether to use real robot or simulation")
flags.DEFINE_bool("show_gui", True, "whether to show GUI.")
flags.DEFINE_float("max_time_secs", 1., "maximum time to run the robot.")
flags.DEFINE_enum("world", "plane",
                  ["plane", "slope", "stair", "uneven"],
                  "world type to choose from.")
flags.DEFINE_bool("camera_fixed", False, "whether to fix camera in simulation")
FLAGS = flags.FLAGS

WORLD_NAME_TO_CLASS_MAP = dict(plane=plane_world.PlaneWorld,
                               slope=slope_world.SlopeWorld,
                               stair=stair_world.StairWorld,
                               uneven=uneven_world.UnevenWorld)


def main(argv):
    del argv  # unused

    # Matlab Engine
    # mat_engine = MatEngine().mat_engine
    mat_engine = None

    # Construct robot
    if FLAGS.show_gui and not FLAGS.use_real_robot:
        p = bullet_client.BulletClient(connection_mode=pybullet.GUI)
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
    else:
        p = bullet_client.BulletClient(connection_mode=pybullet.DIRECT)

    p.resetSimulation()
    # pp = p.getPhysicsEngineParameters()
    # print(f"pp is: {pp}")
    # import time
    # time.sleep(123)
    p.setAdditionalSearchPath('simulator/sim_envs_v2')
    # p.setAdditionalSearchPath('simulator/sim_envs_unitree_ros')
    # quadruped = p.loadURDF('simulator/sim_envs_unitree_ros/urdf/a1.urdf')
    # quadruped = p.loadURDF('simulator/sim_envs_unitree_bullet/laikago/urdf/laikago.urdf')
    # quadruped = p.loadURDF('urdf/plane.urdf')
    # import time
    # time.sleep(123)

    p.setPhysicsEngineParameter(numSolverIterations=30)
    p.setTimeStep(0.001)
    p.setGravity(0, 0, -9.8)
    p.setPhysicsEngineParameter(enableConeFriction=0)

    # Camera setup:
    # p.resetDebugVisualizerCamera(
    #     cameraDistance=4.,
    #     cameraYaw=-105,
    #     cameraPitch=-15,
    #     cameraTargetPosition=[3, 0, 0]
    # )
    p.addUserDebugLine([0, 0, 0], [1, 0, 0], lineColorRGB=[255, 0, 0])
    p.addUserDebugLine([0, 0, 0], [0, 1, 0], lineColorRGB=[0, 255, 0])
    p.addUserDebugLine([0, 0, 0], [0, 0, 1], lineColorRGB=[0, 0, 255])

    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    # p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, f"/home/mao/Desktop/record.mp4")

    world = WORLD_NAME_TO_CLASS_MAP['plane'](p)
    world.build_world()

    # pp = p.getPhysicsEngineParameters()
    # print(f"pp is: {pp}")
    # import time
    # time.sleep(123)

    ddpg_agent = ddpg.DDPGAgent(
        params=DDPGParams(),
        taylor_params=TaylorParams(),
        shape_action=6,
        shape_observations=12,
        # model_path=f"models/phydrl-new",
        # model_path=f"models/phydrl-a1-modified-yaw_zero",
        # model_path=f"models/phydrl-a1",
        model_path=f"models/friction_0.7_delay_vel_0.6_sub_optimal1_best",
        # model_path=f"models/phydrl-a1-origin_model-yaw_zero",
        mode='test'
    )
    ddpg_agent.agent_warmup()
    ddpg_agent = None

    # Construct robot class:
    if FLAGS.use_real_robot:

        a1_real_params = A1RealParams()
        members = vars(a1_real_params)
        for k, v in vars(a1_real_params.a1_params).items():
            print(f"{k}: {v}")

        for k, v in vars(a1_real_params.stance_params).items():
            print(f"{k}: {v}")

        for k, v in vars(a1_real_params.motor_params).items():
            print(f"{k}: {v}")

        for k, v in vars(a1_real_params.swing_params).items():
            print(f"{k}: {v}")
        # members = dir(a1_real_params)
        # filtered_members = [m for m in members if not m.startswith('__')]
        # for member in filtered_members:
        #
        #     mms = dir(member)
        #     filtered_mms = [m for m in mms if not m.startswith('__')]
        #     for mm in mms:
        #         value = getattr(member, mm)
        #         print(f"{mm}: {value}")

        #a1_real_params.stance_params.qp_kp = np.diag((0., 0., 63., 33., 33., 31.))
        #a1_real_params.stance_params.qp_kd = np.diag((24, 20., 20., 22., 22., 22.))
        a1_real_params.stance_params.friction_coeff = 0.7

        a1_real_params.a1_params.desired_vx = 0.2

        # a1_real_params.stance_params.objective_function = 'state'
        a1_real_params.stance_params.objective_function = 'acceleration'

        robot = a1_robot.A1Robot(
            pybullet_client=p,
            ddpg_agent=ddpg_agent,
            mat_engine=mat_engine,
            a1_robot_params=a1_real_params.a1_params,
            motor_params=a1_real_params.motor_params,
            swing_params=a1_real_params.swing_params,
            stance_params=a1_real_params.stance_params,
            logdir='saved/logs/real_plant'
        )
    else:
        a1_sim_params = A1SimParams()
        # a1_sim_params.stance_params.objective_function = 'state'
        a1_sim_params.stance_params.objective_function = 'acceleration'
        a1_sim_params.a1_params.desired_vx = 0.35

        robot = a1.A1(
            pybullet_client=p,
            mat_engine=mat_engine,
            ddpg_agent=ddpg_agent,
            a1_params=a1_sim_params.a1_params,
            motor_params=a1_sim_params.motor_params,
            swing_params=a1_sim_params.swing_params,
            stance_params=a1_sim_params.stance_params,
            logdir='saved/logs/simulation'
        )

    # Set moving speed
    # robot.controller.set_desired_speed([0.2, 0], 0)

    try:
        # start_time = controller.time_since_reset
        start_time = time.time()
        current_time = start_time
        last_time = current_time

        robot.controller.start_thread()
        # ha_teacher.mp_start()

        while current_time - start_time < FLAGS.max_time_secs:
            current_time = time.time()
            print(f"FLAGS.max_time_secs: {FLAGS.max_time_secs}")
            print(f"time elapsed: {current_time - start_time}")
            # print(f"execution time: {current_time - last_time}")
            last_time = current_time
            # time.sleep(0.005)
            time.sleep(5)
            print("main thread updating...")

            if not robot.controller.is_safe:
                logging.info("the robot is likely to fail, cut off for safety concern!")
                break

    finally:
        robot.controller.set_controller_mode(
            wbc_controller.ControllerMode.TERMINATE)

        robot.controller.dump_logs()  # Record and dump logs
        # time.sleep(5)
        robot.controller.stop_thread()


if __name__ == "__main__":
    app.run(main)
