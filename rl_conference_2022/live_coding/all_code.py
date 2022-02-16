# Let's get started with some basic imports.

import ray  # .. of course

import gym
import numpy as np
import os
import pprint
import re
import torch


from typing import List, Optional
from ray.rllib.utils.numpy import softmax


class RecommSys001(gym.Env):
    def __init__(self,
                 config=None,
                 ):
        """Initializes a RecommSys001 instance.

        Args:
            num_categories: Number of topics a user could be interested in and a
                document may be classified with. This is the embedding size for
                both users and docs. Each category in the user/doc embeddings
                can have values between 0.0 and 1.0.
            num_docs_to_select_from: The number of documents to present to the agent
                each timestep. The agent will then have to pick a slate out of these.
            slate_size: The size of the slate to recommend to the user at each
                timestep.
            num_docs_in_db: The total number of documents in the DB. Set this to None,
                in case you would like to resample docs from an infinite pool.
            num_users_in_db: The total number of users in the DB. Set this to None,
                in case you would like to resample users from an infinite pool.
            user_time_budget: The total time budget a user has throughout an episode.
                Once this time budget is used up (through engagements with
                clicked/selected documents), the episode ends.
        """
        self.num_categories = config["num_categories"]
        self.num_docs_to_select_from = config["num_docs_to_select_from"]
        self.slate_size = config["slate_size"]

        self.num_docs_in_db = config.get("num_docs_in_db")
        self.docs_db = None
        self.num_users_in_db = config.get("num_users_in_db")
        self.users_db = None

        self.user_time_budget = config.get("user_time_budget", 60.0)

        self.observation_space = gym.spaces.Dict({
            # The D docs our agent sees at each timestep. It has to select a k-slate
            # out of these.
            "doc": gym.spaces.Dict({
                str(idx): gym.spaces.Box(0.0, 1.0, shape=(self.num_categories,), dtype=np.float32) for idx in range(self.num_docs_to_select_from)
            }),
            # The user engaging in this timestep/episode.
            "user": gym.spaces.Box(0.0, 1.0, shape=(self.num_categories,), dtype=np.float32),
            # For each item in the previous slate, was it clicked? If yes, how
            # long was it being engaged with (e.g. watched)?
            "response": gym.spaces.Tuple([
                gym.spaces.Dict({
                    # Clicked or not?
                    "click": gym.spaces.Discrete(2),
                    # Engagement time (how many minutes watched?).
                    "engagement": gym.spaces.Box(0.0, 100.0, shape=(), dtype=np.float32),
                }) for _ in range(self.slate_size)
            ]),
        })
        # Our action space is
        self.action_space = gym.spaces.MultiDiscrete([
            self.num_docs_to_select_from for _ in range(self.slate_size)
        ])

    def reset(self):
        # Reset the current user's time budget.
        self.current_user_budget = self.user_time_budget

        # Sample a user for the next episode/session.
        # Pick from a only-once-sampled user DB.
        if self.num_users_in_db is not None:
            if self.users_db is None:
                self.users_db = [softmax(np.random.normal(scale=2.0, size=(self.num_categories,))) for _ in range(self.num_users_in_db)]
            self.current_user = self.users_db[np.random.choice(self.num_users_in_db)]
        # Pick from an infinite pool of users.
        else:
            self.current_user = softmax(np.random.normal(scale=2.0, size=(self.num_categories,)))

        return self._get_obs()

    def step(self, action: List[int]):
        # Action is the suggested slate (indices of the docs in the suggested ones).

        # User choice model: User picks a doc stochastically,
        # where probs are dot products between user- and doc feature
        # (categories) vectors. There is also a no-click "doc" for which all features
        # are 0.5.
        user_doc_overlaps = [
            np.dot(self.current_user, self.currently_suggested_docs[str(doc_idx)])
            for doc_idx in action
        ] + [np.dot(self.current_user, np.array([1.0 / self.num_categories for _ in range(self.num_categories)]))]

        which_clicked = np.random.choice(
            np.arange(self.slate_size + 1),
            p=softmax(np.array(user_doc_overlaps) * 5.0),  # TODO explain why *5.0 -> lower temperature of distribution
        )

        # Reward is the overlap, if clicked. 0.0 if nothing clicked.
        reward = 0.0

        # If anything clicked, deduct from the current user's time budget and compute
        # reward.
        if which_clicked < self.slate_size:
            reward = user_doc_overlaps[which_clicked]
            self.current_user_budget -= 1.0
        done = self.current_user_budget <= 0.0

        # Compile response.
        response = tuple({
            "click": int(idx == which_clicked),
            "engagement": reward if idx == which_clicked else 0.0,
        } for idx in range(len(user_doc_overlaps) - 1))

        return self._get_obs(response=response), reward, done, {}

    def _get_obs(self, response=None):
        # Sample D docs from infinity or our pre-existing docs.
        # Pick from a only-once-sampled docs DB.
        if self.num_docs_in_db is not None:
            if self.docs_db is None:
                self.docs_db = [softmax(np.random.normal(scale=2.0, size=(self.num_categories,))) for _ in range(self.num_docs_in_db)]
            self.currently_suggested_docs = {
                str(i): self.docs_db[doc_idx].astype(np.float32) for i, doc_idx in enumerate(np.random.choice(self.num_docs_in_db, size=(self.num_docs_to_select_from,), replace=False))
            }
        # Pick from an infinite pool of docs.
        else:
            self.currently_suggested_docs = {
                str(idx): softmax(np.random.normal(scale=2.0, size=(self.num_categories,))).astype(np.float32) for idx in range(self.num_docs_to_select_from)
            }

        return {
            "user": self.current_user.astype(np.float32),
            "doc": self.currently_suggested_docs,
            "response": response if response else self.observation_space["response"].sample()
        }

env = RecommSys001(config=dict(
    num_categories=10,
    num_docs_to_select_from=30,
    slate_size=2,
    num_docs_in_db=100,
    num_users_in_db=100,
))
env

# !LIVE CODING!

# 1) Reset the env.
obs = env.reset()

# Number of episodes already done.
num_episodes = 0
# Current episode's accumulated reward.
episode_reward = 0.0
# Collect all episode rewards here to be able to calculate a random baseline reward.
episode_rewards = []

# 2) Enter an infinite while loop (to step through the episode).
while num_episodes < 200:
    # 3) Calculate agent's action, using random sampling via the environment's action space.
    action = env.action_space.sample()
    # action = trainer.compute_single_action([obs])

    # 4) Send the action to the env's `step()` method to receive: obs, reward, done, and info.
    obs, reward, done, info = env.step(action)
    episode_reward += reward

    # 5) Check, whether the episde is done, if yes, break out of the while loop.
    if done:
        # print(f"Episode done - accumulated reward={episode_reward}")
        num_episodes += 1
        env.reset()
        episode_rewards.append(episode_reward)
        episode_reward = 0.0

# 6) Run it and print out mean episode reward! :)
print(f"Avg. episode reward={np.mean(episode_rewards)}")


# Start a new instance of Ray (when running this tutorial locally) or
# connect to an already running one (when running this tutorial through Anyscale).

ray.init(local_mode=True)  # Hear the engine humming? ;)

# In case you encounter the following error during our tutorial: `RuntimeError: Maybe you called ray.init twice by accident?`
# Try: `ray.shutdown() + ray.init()` or `ray.init(ignore_reinit_error=True)`


# Import a Trainable (one of RLlib's built-in algorithms):
# We use the SlateQ algorithm here b/c it is specialized in solving slate recommendation problems
# and works well with RLlib's RecSim environment adapter.

from ray.rllib.agents.slateq import SlateQTrainer
from ray import tune

tune.register_env("my_env", lambda config: RecommSys001(**config))


# Specify a very simple config, defining our environment and some environment
# options (see environment.py).
config = {
    "env": RecommSys001,#"my_env",  # "my_env" <- if we previously have registered the env with `tune.register_env("[name]", lambda config: [returns env object])`.
    "env_config": {
        "num_categories": 10,
        "num_docs_to_select_from": 30,
        "slate_size": 2,
        "num_docs_in_db": 100,
        "num_users_in_db": 100,
    },
    # Most of RLlib's algos work with both framework=torch|tf, however,
    # SlateQ so far is torch-only.
    # "framework": "torch",
}
# Instantiate the Trainer object using above config.
rllib_trainer = SlateQTrainer(config=config)
rllib_trainer


results = rllib_trainer.train()

# Delete the config from the results for clarity.
# Only the stats will remain, then.
del results["config"]
# Pretty print the stats.
pprint.pprint(results)


# Run `train()` n times. Repeatedly call `train()` now to see rewards increase.
# Move on once you see episode rewards of 1050.0 or more.
for _ in range(10):
    results = rllib_trainer.train()
    print(f"Iteration={rllib_trainer.iteration}: R(\"return\")={results['episode_reward_mean']}")


# Let's actually "look inside" our Trainer to see what's in there.
#from ray.rllib.utils.numpy import softmax

# To get the policy inside the Trainer, use `Trainer.get_policy([policy ID]="default_policy")`:
policy = rllib_trainer.get_policy()
print(f"Our Policy right now is: {policy}")

# To get to the model inside any policy, do:
model = policy.model
# print(f"Our Policy's model is: {model}")

# Print out the policy's action and observation spaces.
print(f"Our Policy's observation space is: {policy.observation_space}")
print(f"Our Policy's action space is: {policy.action_space}")

# Produce a random obervation (B=1; batch of size 1).
obs = env.observation_space.sample()
print(torch.stack([torch.from_numpy(v) for k, v in obs["doc"].items()]))

# Get the action logits (as torch tensor).
per_slate_q_values = model.get_per_slate_q_values(
    user=torch.from_numpy(obs["user"]).unsqueeze(0),
    doc=torch.stack([torch.from_numpy(v) for k, v in obs["doc"].items()]).unsqueeze(
        0))
per_slate_q_values = per_slate_q_values.detach().cpu().numpy()
print(f"per_slate_q_values={per_slate_q_values}")

rllib_trainer.stop()


# Configuration dicts and Ray Tune.
# Where are the default configuration dicts stored?

# SlateQ algorithm:
from ray.rllib.agents.slateq import DEFAULT_CONFIG as SLATEQ_DEFAULT_CONFIG
print(f"SlateQ's default config is:")
pprint.pprint(SLATEQ_DEFAULT_CONFIG)

# DQN algorithm:
#from ray.rllib.agents.dqn import DEFAULT_CONFIG as DQN_DEFAULT_CONFIG
#print(f"DQN's default config is:")
#pprint.pprint(DQN_DEFAULT_CONFIG)

# Common (all algorithms).
#from ray.rllib.agents.trainer import COMMON_CONFIG
#print(f"RLlib Trainer's default config is:")
#pprint.pprint(COMMON_CONFIG)


# Plugging in Ray Tune.
# Note that this is the recommended way to run any experiments with RLlib.
# Reasons:
# - Tune allows you to do hyperparameter tuning in a user-friendly way
#   and at large scale!
# - Tune automatically allocates needed resources for the different
#   hyperparam trials and experiment runs on a cluster.

from ray import tune

# Running stuff with tune, we can re-use the exact
# same config that we used when working with RLlib directly!
tune_config = config.copy()

# Let's add our first hyperparameter search via our config.
tune_config["train_batch_size"] = tune.grid_search([32, 64])

# We will configure an "output" location here to make sure we record all environment interactions.
# This for the second part of this tutorial, in which we will explore offline RL.
tune_config["output"] = "logdir"

# Now that we will run things "automatically" through tune, we have to
# define one or more stopping criteria.
# Tune will stop the run, once any single one of the criteria is matched (not all of them!).
stop = {
    # Note that the keys used here can be anything present in the above `rllib_trainer.train()` output dict.
    "training_iteration": 1,#TODO: 10
    "episode_reward_mean": 9.0,
}

# "SlateQ" is a registered name that points to RLlib's SlateQTrainer.
# See `ray/rllib/agents/registry.py`

# Run a simple experiment until one of the stopping criteria is met.
results = tune.run(
    "SlateQ",
    config=tune_config,
    stop=stop,

    # Note that no trainers will be returned from this call here.
    # Tune will create n Trainers internally, run them in parallel and destroy them at the end.
    # However, you can ...
    checkpoint_at_end=True,  # ... create a checkpoint when done.
    checkpoint_freq=10,  # ... create a checkpoint every 10 training iterations.
)

# The previous tune.run (the one we did before the break) produced "historic data" output.
# We will use this output in the following as input to a newly initialized, untrained offline RL algorithm.

# Let's take a look at the generated file(s) first:
output_dir = results.get_best_logdir(metric="episode_reward_mean", mode="max")
print(output_dir)

# Here is what the best log directory contains:
print("The logdir contains the following files:")
all_files = os.listdir(os.path.dirname(output_dir + "/"))
output_file = all_files[[re.match("^output.+\.json$", f) is not None for f in all_files].index(True)]
print(output_file)

# Let's configure a new RLlib Trainer, one that's capable of reading the JSON input described
# above and able to learn from this input.

# For simplicity, we'll start with a behavioral cloning (BC) trainer:
from ray.rllib.agents.marwil import BCTrainer

offline_rl_config = {
    # Specify your offline RL algo's historic (JSON) input:
    "input": output_dir + "/" + output_file,
    # Note: For non-offline RL algos, this is set to "sampler" by default.
    #"input": "sampler",
    "observation_space": env.observation_space,
    "action_space": env.action_space,
}

bc_trainer = BCTrainer(config=offline_rl_config)
bc_trainer


# Let's train our new behavioral cloning Trainer for some iterations:
for _ in range(3):
    results = bc_trainer.train()
    print(results["episode_reward_mean"])


# Oh no! What happened?
# We don't have an environment! No way to measure rewards per episode!

# A quick fix would be:
# a) We cheat! Let's use our environment from above to run some separate evaluation workers on while we train:

"""offline_rl_config.update({
    # Setup an "evaluation track": A separate set of remote workers
    # that have their own config overrides.
    # ----------

    # Evaluate every training iteration.
    "evaluation_interval": 1,
    # Run evaluation in parallel to training. This saves time, but has the
    # effect that evaluation results are always 1 iteration behind reported
    # training results.
    "evaluation_parallel_to_training": True,
    # 1 remote worker.
    "evaluation_num_workers": 1,
    # Each time trainer.evaluate() is called, run n episodes.
    "evaluation_duration": 100,
    "evaluation_duration_unit": "episodes",
    # Config overrides: Use Trainer's main config + these changes below.
    "evaluation_config": {
        "env": RecommSys001,
        "env_config": {
            "num_categories": 10,
            "num_docs_to_select_from": 30,
            "slate_size": 2,
            "num_docs_in_db": 100,
            "num_users_in_db": 100,
        },
        "input": "sampler",
    },
})"""

# Ok, let's try again and see, whether we are getting evaluation results:
#bc_trainer.stop()
#bc_trainer = BCTrainer(config=offline_rl_config)
#for _ in range(5):
#    results = bc_trainer.train()
#    print(results["episode_reward_mean"])
# This looks much better.

# But it's not a solution! Remember, we really, truly don't have an env :(

# b) We use OPE ("off policy estimation"). RLlib comes with two built-in estimators:
# IS (importance sampling) and WIS (weightd importance sampling).
# Both are already pre-configured by default and used whenever you have an input file
# specified (as in our case). The results of these estimations can be found in our results
# dict. Let's take a look at the "off_policy_estimator" key:

from ray.rllib.agents.marwil import MARWILTrainer

marwil_trainer = MARWILTrainer(config=offline_rl_config)
marwil_trainer

for _ in range(3):
    results = marwil_trainer.train()
    print(results["off_policy_estimator"])


# We use the `Trainer.save()` method to create a checkpoint.
checkpoint_file = marwil_trainer.save()
print(f"Trainer (at iteration {marwil_trainer.iteration} was saved in '{checkpoint_file}'!")

# Here is what a checkpoint directory contains:
print("The checkpoint directory contains the following files:")
print(os.listdir(os.path.dirname(checkpoint_file)))


# Serving RLlib Models in Production.
# Let's assume we have trained a couple of differently configured BC- or MARWIL
# Trainers using RLlib and Tune and - according to their OPE scores - would like to
# deploy one of these into our production system for live action serving.



# You create deployments with Ray Serve by using the `@serve.deployment` on a class that implements two methods:

# - The `__init__` call creates the deployment instance and loads your data once.
#   In the below example we restore our `PPOTrainer` from the checkpoint we just created.
# - The `__call__` method will be invoked every request.
#   For each incoming request, this method has access to a `request` object,
#   which is a [Starlette Request](https://www.starlette.io/requests/).

# We can load the request body as a JSON object and, assuming there is a key called `observation`,
# in your deployment you can use `request.json()["observation"]` to retrieve observations (`obs`) and
# pass them into the restored `trainer` using the `compute_single_action` method.

from starlette.requests import Request
from ray import serve


@serve.deployment(route_prefix="/ray-serve-endpoint")
class ServeTrainedModel:
    def __init__(self, checkpoint_file) -> None:
        self.trainer = MARWILTrainer(config=offline_rl_config)
        self.trainer.restore(checkpoint_file)

    async def __call__(self, request: Request):
        json_input = await request.json()
        obs = json_input["observation"]

        action = self.trainer.compute_single_action(obs)
        return {"action": np.array(action)}

# Now that we've defined our `ServePPOModel` service, let's deploy it to Ray Serve.
# The deployment will be exposed through the `/ray-serve-endpoint` route.

serve.start()
ServeTrainedModel.deploy(checkpoint_file)


# Note that the `checkpoint_file` that we passed to the `deploy()` method will be passed to
# the `__init__` method of the `ServePPOModel` class that we defined above.

# Now that the model is deployed, let's query it!

import requests

for _ in range(5):
    obs = env.reset()

    print(f"-> Sending observation {obs}")
    resp = requests.get(
        "http://localhost:8000/ray-serve-endpoint", json={"observation": obs}
    )
    print(f"<- Received response {resp.json()}")
