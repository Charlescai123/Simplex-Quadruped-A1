from locomotion import wbc_controller
from config.locomotion.robots.a1_robot_params import A1RobotParams
from config.phydrl.env_params import TrainerEnvParams
from agents.phydrl.envs.a1_trainer_env import A1TrainerEnv
import time

trainer_params = TrainerEnvParams()
trainer_params.show_gui = True
trainer_params.use_real_urdf = True
trainer_params.a1_params.urdf_path = "a1.urdf"
# trainer_params.use_real_urdf = False
# trainer_params.a1_params.urdf_path = "a1/a1.urdf"

a1_env = A1TrainerEnv(trainer_params)


actions = [0.] * 60

for _ in range(100000):

    a1_env.step(actions, action_mode="mpc")
    # time.sleep(1)

    print(a1_env.get_states_vector())
