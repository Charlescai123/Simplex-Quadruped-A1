"""Robot Parameters for A1 Training"""

from config.phydrl.ddpg_params import DDPGParams
from config.phydrl.logger_params import LoggerParams
from config.phydrl.taylor_params import TaylorParams
from config.phydrl.env_params import TrainerEnvParams


# A1 Parameters for PhyDRL Training
class A1PhyDRLParams():

    def __init__(self):
        self.trainer_params = TrainerEnvParams()
        self.agent_params = DDPGParams()
        self.logger_params = LoggerParams()
        self.taylor_params = TaylorParams()

        self.trainer_params.show_gui = False

        self.trainer_params.use_real_urdf = True
        self.trainer_params.a1_params.urdf_path = "urdf/a1.urdf"
        # self.trainer_params.use_real_urdf = False
        # self.trainer_params.a1_params.urdf_path = "a1/a1.urdf"

        self.logger_params.mode = 'train'
        self.logger_params.model_name = 'phydrl-a1'
        # self.agent_params.model_path = f"models/{self.logger_params.model_name}"


if __name__ == '__main__':
    a1_train = A1PhyDRLParams()
    print(a1_train.trainer_params.a1_params.time_step)
