from src.envs.locomotion import locomotion_controller
from config_json.locomotion.robots.a1_robot_params import A1RobotParams
from config_json.phydrl.env_params import TrainerEnvParams
from src.envs.a1_envs import A1TrainerEnv
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
