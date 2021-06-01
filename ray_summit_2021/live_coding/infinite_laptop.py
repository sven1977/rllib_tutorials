import ray
#import anyscale

ray.client("anyscale://tutorial_project").connect()

from ray.rllib.examples.env.gpu_requiring_env import GPURequiringEnv #random_env import RandomEnv
from ray import tune

config={
    "env": GPURequiringEnv,
    "framework": "torch",
}
#ray.init()
tune.run("PPO", config=config, stop={"training_iteration": 1})
