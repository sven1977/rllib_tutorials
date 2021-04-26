# Exercise #2:
# ============
# Try learning our environment using Ray tune.run and a simple
# hyperparameter grid_search over 2 different learning rates
# (pick your own values) and 2 different `train_batch_size` settings
# (use 2000 and 4000). Also make RLlib use a [64, 64] dense layer
# stack as the NN model.
# Good luck! :)


if __name__ == "__main__":

    # Solution:
    from ray import tune
    from environment import MultiAgentArena

    stop = {
        "episode_reward_mean": 100.0,
        "training_iteration": 50,
    }

    tune.run("PPO", config={
        "env": MultiAgentArena,
        # Test 2 reasonable learning rates.
        "lr": tune.grid_search([0.001, 0.002]),
        # # Test 2 reasonable batch sizes.
        "train_batch_size": tune.grid_search([2000, 4000]),
        # Change RLlib's default model's fully connected layer stack
        # to [64, 64] (from the default, which is [256, 256]).
        "model": {
            "fcnet_hiddens": [64, 64],
        },
    }, stop=stop, verbose=2, checkpoint_at_end=True)
