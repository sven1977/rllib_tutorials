import anyscale

#anyscale.project_dir("/Users/sven/.anyscale/scratch_sven/")
anyscale.connect()

import ray
from ray.rllib.agents.ppo import PPOTrainer

assert ray.is_initialized(), "ERROR!"

trainer = PPOTrainer(env="CartPole-v0")

trainer.train()
