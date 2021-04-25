# Prerequisites for running this tutorial.

# pip install ray[rllib]
# pip install [tensorflow|torch]  # <- ether one works!

import ray
from environment import MultiAgentArena


if __name__ == "__main__":
    # 4.
    # Plugging in RLlib.
    from ray.rllib.agents.ppo import PPOTrainer

    ray.init()

    config = {
        "lr": 0.0001,
        "env": MultiAgentArena,
        "env_config": {
            "config": {
                "width": 10,
                "height": 10,
            },
        },
    }
    rllib_trainer = PPOTrainer(config=config)
    print(rllib_trainer.train())

    # 4.a)
    # Using Ray tune to run things instead of rllib directly.
    # This will help you in the future to hyperparam-tune your
    # algos and experiments.
    from ray import tune
    # Now that we will run things "automatically" through tune, we have to
    # define one or more stopping criteria.
    stop = {
        # explain that keys here can be anything present in the above print(trainer.train())
        "training_iteration": 5,
    }
    tune.run("PPO", config=config, verbose=2, stop=stop)

    # 5. Talk about config dict.
    # Where do we find the defaults for each agent?
    # e.g.
    from ray.rllib.agents.ppo import DEFAULT_CONFIG
    print(DEFAULT_CONFIG)
    config.update(
        {
            # Try different learning rates.
            "lr": tune.grid_search([0.0001, 0.5]),
            # NN model config to tweak the default model
            # that'll be created by RLlib for the policy.
            "model": {
                # e.g. change the dense layer stack.
                "fcnet_hiddens": [256, 256, 256],
                # Alternatively, you can specify a custom model here
                # (we'll cover that later).
                # "custom_model": ...
                # Pass kwargs to your custom model.
                # "custom_model_config": {}
            },
        }
    )
    # Repeat our experiment using tune's grid-search feature.
    results = tune.run(
        "PPO",
        config=config,
        verbose=2,
        stop=stop,
        checkpoint_at_end=True,  # create a checkpoint when done.
        checkpoint_freq=1,  # create a checkpoint on every iteration.
    )

    # 5.a
    # Picking up an experiment from a saved checkpoint.
    print(results)

    # 7.
    # Exercise No2:
    # Try learning our environment using Ray tune.run and a simple
    # hyperparameter grid_search over 2 different learning rates
    # (pick your own values) and 2 different `train_batch_size` settings
    # (use 2000 and 4000). Also make RLlib use a [64, 64] dense layer
    # stack as the NN model.

    # 7.a
    # Solution:
    tune.run("PPO", config={
        "lr": tune.grid_search([0.001, 0.002]),
        "train_batch_size": tune.grid_search([2000, 4000]),
        "model": {
            "fcnet_hiddens": [64, 64],
        },
    }, stop=stop, verbose=2, checkpoint_at_end=True)

