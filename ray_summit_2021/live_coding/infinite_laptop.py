import anyscale

#anyscale.project_dir("/Users/sven/.anyscale/scratch_sven/")
anyscale.connect()

import ray
from ray.rllib.agents.ppo import PPOTrainer

assert ray.is_initialized(), "ERROR!"

from gym.envs.classic_control.cartpole import CartPoleEnv
from ray.rllib.examples.env.random_env import RandomEnv
class MyEnv(CartPoleEnv):
    def __init__(self, config=None):
        super().__init__()

trainer = PPOTrainer(env=RandomEnv)
trainer.train()
