# Exercise #3:
# ============
# The episode mean rewards reported to us thus far were always the sum
# of both agents, which doesn't seem to make too much sense given that
# the agents are adversarial.
# Instead, we would like to know, what the individual agents' rewards are in
# our environment.
# Write your own custom callback class (sub-class
# ray.rllib.agents.callback::DefaultCallbacks) and override one or more methods
# therein to manipulate and collect the following data:

#TODO

# a) Extract each agent's individual rewards from ...
# b) Store each agents reward under the new "reward_agent1" and
#    "reward_agent2" keys in the custom metrics.
# c) Run a simple experiment and confirm that you are seeing these two new stats
#    in the tensorboard output.
# Good luck! :)


if __name__ == "__main__":

    # Solution:
    import ray
    from ray.rllib.agents.callbacks import DefaultCallbacks
    from ray import tune

    from environment import MultiAgentArena

    class MyCallback(DefaultCallbacks):
        def on_episode_step(self,
                        *,
                        worker,
                        base_env,
                        episode,
                        env_index=None,
                        **kwargs):
            print()

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
