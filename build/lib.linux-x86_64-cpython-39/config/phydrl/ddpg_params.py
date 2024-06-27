"""DDPG Parameters"""

class DDPGParams:
    def __init__(self):
        self.action_noise = "no"  # exploration noise for action
        self.action_noise_factor = 1
        self.action_noise_half_decay_time = 1e6
        self.soft_alpha = 0.005
        self.learning_rate_actor = 0.0003
        self.learning_rate_critic = 0.0003
        self.batch_size = 300
        self.add_target_action_noise = True
        self.gamma_discount = 0.1
        self.model_path = None
        self.training_episode = 1e6
        self.max_episode_steps = 10200
        self.experience_prefill_size = 300  # no less than the batch_size
        self.mode = 'train'
        self.action_mode = 'residual'
        self.use_taylor_nn = False
        self.taylor_editing = False
        self.replay_buffer_size = 51000
        self.add_action_delay = False
