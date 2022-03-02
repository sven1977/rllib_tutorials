# Let's get started with some basic imports.

import ray  # .. of course

import gym
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pprint
from scipy.stats import sem  # standard error of the mean
import torch


from typing import List, Optional

from ray.rllib.utils.numpy import softmax


class RecommSys001(gym.Env):

    def __init__(self, config=None):

        config = config or {}

        # E (embedding size)
        self.num_features = config["num_features"]
        # D
        self.num_items_to_select_from = config["num_items_to_select_from"]
        # k
        self.slate_size = config["slate_size"]

        self.num_items_in_db = config.get("num_items_in_db")
        self.items_db = None
        # Generate an items-DB containing n items, once.
        if self.num_items_in_db is not None:
            self.items_db = [np.random.uniform(0.0, 1.0, size=(self.num_features,))
                            for _ in range(self.num_items_in_db)]

        self.num_users_in_db = config.get("num_users_in_db")
        self.users_db = None
        # Store the user that's currently undergoing the episode/session.
        self.current_user = None

        # How much time does the user have to consume
        self.user_time_budget = config.get("user_time_budget", 1.0)
        self.current_user_budget = self.user_time_budget

        self.observation_space = gym.spaces.Dict({
            # The D items our agent sees at each timestep. It has to select a k-slate
            # out of these.
            "doc": gym.spaces.Dict({
                str(idx):
                    gym.spaces.Box(0.0, 1.0, shape=(self.num_features,), dtype=np.float32)
                    for idx in range(self.num_items_to_select_from)
            }),
            # The user engaging in this timestep/episode.
            "user": gym.spaces.Box(0.0, 1.0, shape=(self.num_features,), dtype=np.float32),
            # For each item in the previous slate, was it clicked? If yes, how
            # long was it being engaged with (e.g. watched)?
            "response": gym.spaces.Tuple([
                gym.spaces.Dict({
                    # Clicked or not?
                    "click": gym.spaces.Discrete(2),
                    # Engagement time (how many minutes watched?).
                    "watch_time": gym.spaces.Box(-np.inf, np.inf, shape=(), dtype=np.float32),
                }) for _ in range(self.slate_size)
            ]),
        })
        # Our action space is
        self.action_space = gym.spaces.MultiDiscrete([
            self.num_items_to_select_from for _ in range(self.slate_size)
        ])

    def reset(self):
        # Reset the current user's time budget.
        self.current_user_budget = self.user_time_budget

        # Sample a user for the next episode/session.
        # Pick from a only-once-sampled user DB.
        if self.num_users_in_db is not None:
            if self.users_db is None:
                self.users_db = [np.random.uniform(0.0, 1.0, size=(self.num_features,))
                                 for _ in range(self.num_users_in_db)]
            self.current_user = self.users_db[np.random.choice(self.num_users_in_db)]
        # Pick from an infinite pool of users.
        else:
            self.current_user = np.random.uniform(0.0, 1, size=(self.num_features,))

        return self._get_obs()

    def step(self, action):
        # Action is the suggested slate (indices of the items in the suggested ones).

        scores = [np.dot(self.current_user, item)
                  for item in self.currently_suggested_items]
        best_reward = np.max(scores)

        # User choice model: User picks an item stochastically,
        # where probs are dot products between user- and item feature
        # vectors.
        # There is also a no-click item whose weight is 1.0.
        user_item_overlaps = np.array([scores[a] for a in action] + [1.0])
        which_clicked = np.random.choice(
            np.arange(self.slate_size + 1), p=softmax(user_item_overlaps))

        # Reward is the overlap, if clicked. 0.0 if nothing clicked.
        reward = 0.0
        # If anything clicked, deduct from the current user's time budget and compute
        # reward.
        if which_clicked < self.slate_size:
            regret = best_reward - user_item_overlaps[which_clicked]
            reward = 1.0 - regret
            self.current_user_budget -= 1.0
        done = self.current_user_budget <= 0.0

        # Compile response.
        response = tuple({
            "click": int(idx == which_clicked),
            "watch_time": reward if idx == which_clicked else 0.0,
        } for idx in range(len(user_item_overlaps) - 1))

        # Return 4-tuple: Next-observation, reward, done (True if episode has terminated), info dict (empty; not used here).
        return self._get_obs(response=response), reward, done, {}

    def _get_obs(self, response=None):
        # Sample D items from infinity or our pre-existing items.
        # Pick from a only-once-sampled items DB.
        if self.num_items_in_db is not None:
            self.currently_suggested_items = [
                self.items_db[item_idx].astype(np.float32)
                for item_idx in np.random.choice(self.num_items_in_db,
                                                size=(self.num_items_to_select_from,),
                                                replace=False)
            ]
        # Pick from an infinite pool of itemsdocs.
        else:
            self.currently_suggested_items = [
                np.random.uniform(0.0, 1, size=(self.num_features,)).astype(np.float32)
                for _ in range(self.num_items_to_select_from)
            ]

        return {
            "user": self.current_user.astype(np.float32),
            "doc": {
                str(idx): item for idx, item in enumerate(self.currently_suggested_items)
            },
            "response": response if response else self.observation_space["response"].sample()
        }

env = RecommSys001(config={
    "num_features": 20,  # E (embedding size)

    "num_items_in_db": 100,  # total number of items in our database
    "num_items_to_select_from": 10,  # number of items to present to the agent to pick a k-slate from
    "slate_size": 1,  # k
    "num_users_in_db": 1,  # total number  of users in our database
})
env


# !LIVE CODING!

def test_env(env):

    # 1) Reset the env.
    obs = env.reset()

    # Number of episodes already done.
    num_episodes = 0
    # Current episode's accumulated reward.
    episode_reward = 0.0
    # Collect all episode rewards here to be able to calculate a random baseline reward.
    episode_rewards = []

    # 2) Enter an infinite while loop (to step through the episode).
    while num_episodes < 1000:
        # 3) Calculate agent's action, using random sampling via the environment's action space.
        action = env.action_space.sample()
        # action = trainer.compute_single_action([obs])

        # 4) Send the action to the env's `step()` method to receive: obs, reward, done, and info.
        obs, reward, done, info = env.step(action)
        episode_reward += reward

        # 5) Check, whether the episde is done, if yes, break out of the while loop.
        if done:
            #print(f"Episode done - accumulated reward={episode_reward}")
            num_episodes += 1
            env.reset()
            episode_rewards.append(episode_reward)
            episode_reward = 0.0

    # 6) Print out mean episode reward!
    env_mean_random_reward = np.mean(episode_rewards)
    print(f"Mean episode reward when acting randomly: {env_mean_random_reward:.2f}+/-{sem(episode_rewards):.2f}")

    return env_mean_random_reward, sem(episode_rewards)

env_mean_random_reward, env_sem_random_reward = test_env(env)


# Start a new instance of Ray (when running this tutorial locally) or
# connect to an already running one (when running this tutorial through Anyscale).

ray.init()  # Hear the engine humming? ;)

# In case you encounter the following error during our tutorial: `RuntimeError: Maybe you called ray.init twice by accident?`
# Try: `ray.shutdown() + ray.init()` or `ray.init(ignore_reinit_error=True)`


# Import a Trainable (one of RLlib's built-in algorithms):
# We start our endeavor with the Bandit algorithms here b/c they are specialized in solving
# n-arm/recommendation problems.
from ray.rllib.agents.bandit import BanditLinUCBTrainer

# Environment wrapping tools for:
# a) Converting MultiDiscrete action space (k-slate recommendations) down to Discrete action space (we only have k=1 for now anyways).
# b) Making sure our google RecSim-style environment is understood by RLlib's Bandit Trainers.
from ray.rllib.env.wrappers.recsim import MultiDiscreteToDiscreteActionWrapper, \
    RecSimObservationBanditWrapper

from ray import tune

tune.register_env(
    "recomm-sys-001-for-bandits",
    lambda config: RecSimObservationBanditWrapper(MultiDiscreteToDiscreteActionWrapper(RecommSys001(config))))

bandit_config = {
    # Use our tune-registered "RecommSys001" class.
    "env": "recomm-sys-001-for-bandits",
    "env_config": {
        "num_features": 20,  # E

        "num_items_in_db": 100,
        "num_items_to_select_from": 10,  # D
        "slate_size": 1,  # k=1

        "num_users_in_db": 1,
    },
    #"evaluation_duration_unit": "episodes",
    "timesteps_per_iteration": 1,
}

# Create the RLlib Trainer using above config.
bandit_trainer = BanditLinUCBTrainer(config=bandit_config)

# Train for n iterations (timesteps) and collect n-arm rewards.
rewards = []
for _ in range(300):
    result = bandit_trainer.train()
    rewards.append(result["episode_reward_mean"])
    print(".", end="")

# Plot per-timestep (episode) rewards.
plt.figure(figsize=(10,7))
plt.plot(rewards)#x=[i for i in range(len(rewards))], y=rewards, xerr=None, yerr=[sem(rewards) for i in range(len(rewards))])
plt.title("Mean reward")
plt.xlabel("Time/Training steps")

# Add mean random baseline reward (red line).
plt.axhline(y=env_mean_random_reward, color="r", linestyle="-")

plt.show()



# Update our env_config: Making things harder.
bandit_config.update({
    "env_config": {
        "num_features": 20,  # E (no change)

        "num_items_in_db": 100,  # (no change)
        "num_items_to_select_from": 10,  # D (no change)
        "slate_size": 2,  # k=2

        "num_users_in_db": None,  # More users!
        "user_time_budget": 10.0,  # Longer episodes.
    },
})

# Re-computing our random baseline.
harder_env = RecommSys001(config=bandit_config["env_config"])
harder_env_mean_random_reward, _ = test_env(harder_env)


# Create the RLlib Trainer using above config.
bandit_trainer = BanditLinUCBTrainer(config=bandit_config)

# Train for n iterations (timesteps) and collect n-arm rewards.
rewards = []
for _ in range(1200):
    result = bandit_trainer.train()
    rewards.append(result["episode_reward_mean"])
    print(".", end="")

# Plot per-timestep (episode) rewards.
plt.figure(figsize=(10,7))
plt.plot(rewards)#x=[i for i in range(len(rewards))], y=rewards, xerr=None, yerr=[sem(rewards) for i in range(len(rewards))])
plt.title("Mean reward")
plt.xlabel("Time/Training steps")

# Add mean random baseline reward (red line).
plt.axhline(y=harder_env_mean_random_reward, color="r", linestyle="-")

plt.show()


