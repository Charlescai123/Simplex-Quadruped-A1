"""Example of locomotion on A1 robot.

To run:
python -m examples.a1_locomotion_example   --show_gui=[True|False]   --max_time_secs=[xxx] \
        --use_real_robot=[True|False]   --world=[plane|slope|stair|uneven]   --camera_fixed=[True|False]

For example, to run in simulation:
python -m examples.a1_locomotion_example --show_gui=True --max_time_secs=20 --use_real_robot=False --world=plane

For example, to run in real A1:
python -m examples.a1_locomotion_example --show_gui=False --max_time_secs=15 --use_real_robot=True
"""

import time
import hydra
import pybullet
import numpy as np
from absl import app
from absl import flags
from absl import logging
from pybullet_utils import bullet_client
from omegaconf import DictConfig, OmegaConf

from src.envs.robot import locomotion_controller
from src.envs.robot.unitree_a1 import a1
from src.envs.robot.unitree_a1 import a1_robot
from src.hp_student.agents import ddpg
# from locomotion.wbc_controller import ControllerMode
# from locomotion.wbc_controller import GaitType
from src.envs.simulator.worlds import plane_world, slope_world, stair_world, uneven_world

#
# flags.DEFINE_string("logdir", "logs", "where to log trajectories.")
# flags.DEFINE_bool("use_real_robot", False,
#                   "whether to use real robot or simulation")
# flags.DEFINE_bool("show_gui", True, "whether to show GUI.")
# flags.DEFINE_float("max_time_secs", 1., "maximum time to run the robot.")
# flags.DEFINE_enum("world", "plane",
#                   ["plane", "slope", "stair", "uneven"],
#                   "world type to choose from.")
# flags.DEFINE_bool("camera_fixed", False, "whether to fix camera in simulation")
# FLAGS = flags.FLAGS

WORLD_NAME_TO_CLASS_MAP = dict(plane=plane_world.PlaneWorld,
                               slope=slope_world.SlopeWorld,
                               stair=stair_world.StairWorld,
                               uneven=uneven_world.UnevenWorld)
import tensorflow as tf
import os
# use_gpu = True


# if not use_gpu:
#     os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# else:
#     physical_devices = tf.config.list_physical_devices('GPU')
#     try:
#         tf.config.experimental.set_memory_growth(physical_devices[0], True)
#     except:
#         exit("GPU allocated failed")


def mp_patch_start():
    # data = {
    #     'roll': self.triggered_roll,
    #     'pitch': self.triggered_pitch,
    #     'yaw': self.triggered_yaw,
    #     'kp': self._patch_kp,
    #     'kd': self._patch_kd
    # }
    import copy
    import multiprocessing as mp
    default_kp = np.diag((0., 0., 100., 100., 100., 0.))
    default_kd = np.diag((40., 30., 10., 10., 10., 30.))


    lock = mp.Lock()
    triggered_roll = mp.Value('d', 0)
    triggered_pitch = mp.Value('d', 0)
    triggered_yaw = mp.Value('d', 0)
    # state_update_flag = manager.Value('b', 0)
    # _f_kp = manager.list(copy.deepcopy(default_kp.reshape(36)))
    # _f_kd = manager.list(copy.deepcopy(default_kd.reshape(36)))
    _f_kp = None
    _f_kd = None
    from src.ha_teacher.ha_teacher import HATeacher
    print("creating process for patch computing")
    patch_process = mp.Process(
        # target=HATeacher.patch_compute, args=(triggered_roll, triggered_pitch, triggered_yaw, _f_kp, _f_kd, lock)
        target=HATeacher.patch_compute, args=()
    )
    patch_process.daemon = True
    print("starting patch process")
    patch_process.start()
    print(f"Pid of patch process: {patch_process.pid}")


@hydra.main(version_base=None, config_path="../config", config_name="base_config.yaml")
def main(cfg: DictConfig):
    envs_cfg = cfg.envs.simulator
    robot_cfg = cfg.envs.robot

    # Multiprocessing
    # mp_patch_start()

    # Show GUI or not
    # if envs_cfg.show_gui:
    if robot_cfg.interface.model == 'a1':
        # p = bullet_client.BulletClient(connection_mode=pybullet.DIRECT)
        p = bullet_client.BulletClient(connection_mode=pybullet.GUI)
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
    else:
        p = bullet_client.BulletClient(connection_mode=pybullet.DIRECT)

    # Pybullet settings
    p.resetSimulation()
    p.setGravity(0, 0, -9.8)
    p.setTimeStep(envs_cfg.fixed_time_step)
    p.setAdditionalSearchPath(envs_cfg.envs_path)
    p.setPhysicsEngineParameter(numSolverIterations=envs_cfg.num_solver_iterations)
    p.setPhysicsEngineParameter(enableConeFriction=envs_cfg.enable_cone_friction)

    # pp = p.getPhysicsEngineParameters()
    # print(f"pp is: {pp}")
    # import time
    # time.sleep(123)

    # Camera setup:
    if envs_cfg.camera.is_on:
        p.resetDebugVisualizerCamera(
            cameraDistance=envs_cfg.camera.distance,
            cameraYaw=envs_cfg.camera.yaw,
            cameraPitch=envs_cfg.camera.pitch,
            cameraTargetPosition=envs_cfg.camera.target_position
        )

    # Add origin frame
    p.addUserDebugLine([0, 0, 0], [1, 0, 0], lineColorRGB=[255, 0, 0])
    p.addUserDebugLine([0, 0, 0], [0, 1, 0], lineColorRGB=[0, 255, 0])
    p.addUserDebugLine([0, 0, 0], [0, 0, 1], lineColorRGB=[0, 0, 255])

    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)

    world = WORLD_NAME_TO_CLASS_MAP['plane'](p)
    world.build_world()

    # Record a video or not
    if envs_cfg.video.record:
        p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, cfg.envs.video.save_path)

    # Whether to deploy PhyDRL for inference
    if cfg.general.deploy_phydrl:
        ddpg_agent = ddpg.DDPGAgent(
            agent_cfg=cfg.hp_student.agents,
            taylor_cfg=cfg.hp_student.taylor,
            shape_action=6,
            shape_observations=12,
            mode='test'
        )
        ddpg_agent.actor.save("tf_models/actor")
        # print(f"Testing the trained model: {cfg.phydrl.id}")
        ddpg_agent.agent_warmup()
    elif cfg.general.train_with_phydrl:
        ddpg_agent = ddpg.DDPGAgent(
            agent_cfg=cfg.hp_student.agents,
            taylor_cfg=cfg.hp_student.taylor,
            shape_action=6,
            shape_observations=12,
            mode='train'
        )
        ddpg_agent.actor.save("tf_models/actor")
        # print(f"Testing the trained model: {cfg.phydrl.id}")
        ddpg_agent.agent_warmup()
    else:
        ddpg_agent = None

    # Construct robot class:
    if robot_cfg.interface.model == 'a1':
        robot = a1.A1(
            pybullet_client=p,
            ddpg_agent=ddpg_agent,
            cmd_params=robot_cfg.command,
            a1_params=robot_cfg.interface,
            gait_params=robot_cfg.gait_scheduler,
            swing_params=robot_cfg.swing_controller,
            stance_params=robot_cfg.stance_controller,
            motor_params=robot_cfg.motor_group,
            vel_estimator_params=robot_cfg.com_velocity_estimator,
            logdir=f"{cfg.general.log_dir}/{cfg.general.name}"
        )
    elif robot_cfg.interface.model == 'a1_robot':
        robot = a1_robot.A1Robot(
            pybullet_client=p,
            ddpg_agent=ddpg_agent,
            cmd_params=robot_cfg.command,
            a1_robot_params=robot_cfg.interface,
            gait_params=robot_cfg.gait_scheduler,
            swing_params=robot_cfg.swing_controller,
            stance_params=robot_cfg.stance_controller,
            motor_params=robot_cfg.motor_group,
            vel_estimator_params=robot_cfg.com_velocity_estimator,
            logdir=f"{cfg.general.log_dir}r/{cfg.general.name}"
        )
    else:
        raise RuntimeError("Cannot find predefined robot model")
    # Set moving speed
    # robot.controller.set_desired_speed([0.5, 0], 0)

    try:
        # start_time = controller.time_since_reset
        start_time = time.time()
        current_time = start_time
        last_time = current_time

        robot.controller.start_thread()

        while current_time - start_time < cfg.general.runtime:
            current_time = time.time()
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
            locomotion_controller.ControllerMode.TERMINATE)
        robot.controller.dump_logs()  # Record and dump logs

        # robot.controller.stop_thread()


if __name__ == "__main__":
    test_a = tf.constant([1.0, 2.0, 3.0])
    main()
    # app.run(main)
