# Solution Exercise #3

import ray
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray import tune


class MyCallback(DefaultCallbacks):
    def on_episode_start(self, *, worker, base_env,
                         policies, episode,
                         env_index, **kwargs):
        # Set per-episode object to capture, which states (observations)
        # have been visited by agent1.
        episode.user_data["ground_covered"] = set()
        # Set per-episode agent2-blocks counter (how many times has agent2 blocked agent1?).
        episode.user_data["num_blocks"] = 0

    def on_episode_step(self, *, worker, base_env,
                        episode, env_index, **kwargs):
        # Add agent1's observation to our set of unique observations.
        ag1_obs = episode.last_raw_obs_for("agent1")
        episode.user_data["ground_covered"].add(ag1_obs)
        # If agent2's reward > 0.0, it means she has blocked agent1.
        ag2_r = episode.prev_reward_for("agent2")
        if ag2_r > 0.0:
            episode.user_data["num_blocks"] += 1

    def on_episode_end(self, *, worker, base_env,
                       policies, episode,
                       env_index, **kwargs):
        # Reset everything.
        episode.user_data["ground_covered"] = set()
        episode.user_data["num_blocks"] = 0



ray.init()

stop = {"training_iteration": 10}
# Specify env and custom callbacks in our config (leave everything else
# as-is (defaults)).
config = {
    "env": MultiAgentArena,
    "callbacks": MyCallback,
}

# Run for a few iterations.
tune.run("PPO", stop=stop, config=config)

# Check tensorboard.
