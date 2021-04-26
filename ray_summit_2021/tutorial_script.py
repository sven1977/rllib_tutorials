# 0) Prerequisites for running this tutorial.

# pip install ray[rllib]
# pip install [tensorflow|torch]  # <- ether one works!


import ray
from environment import MultiAgentArena


# 1) Showing slides on RL cycle + RLlib.
#    - Why is RLlib so popular among industry users?
# 2) -> environment.py: Coding/defining our "problem" via an RL environment.
# 3) -> exercise_1.py: Exercise #1.
# 4) .. continue below ..


if __name__ == "__main__":
    # 4) Plugging in RLlib.

    # Import a Trainable (one of RLlib's built-in algorithms):
    # We use the PPO algorithm here b/c its very flexible wrt its supported
    # action spaces and model types and b/c it learns well almost any problem.
    from ray.rllib.agents.ppo import PPOTrainer

    # Start a new instance of Ray or connect to an already running one.
    ray.init()

    # Specify a very simple config, defining our environment and some environment
    # options (see environment.py).
    config = {
        "env": MultiAgentArena,
        "env_config": {
            "config": {
                "width": 10,
                "height": 10,
            },
        },
    }
    # Instantiate the Trainer object using above config.
    rllib_trainer = PPOTrainer(config=config)
    # That's it, we are ready to train.
    # Calling `train` once runs a single "training iteration". One iteration
    # for most algos contains a) sampling from the environment(s) + b) using the
    # sampled data (observations, actions taken, rewards) to update the policy
    # model (neural network), such that it would pick better actions in the future,
    # leading to higher rewards.
    print(rllib_trainer.train())

    # Run `train()` n times.
    for _ in range(4):
        print(rllib_trainer.train())
    # Save our trainer.
    checkpoint_path = rllib_trainer.save()
    print(f"Trainer was saved in '{checkpoint_path}'!")

    # Pretend, we wanted to pick up training from a previous run:
    new_trainer = PPOTrainer(config=config)
    # Restoring the trained state into the `new_trainer` object.
    new_trainer.restore(checkpoint_path)
    new_trainer.train()

    # 5) Configuration dicts and Ray Tune.
    # Where are the default configuration dicts stored?
    from ray.rllib.agents.ppo import DEFAULT_CONFIG as PPO_DEFAULT_CONFIG
    print(f"PPO's default config is: {PPO_DEFAULT_CONFIG}")
    from ray.rllib.agents.dqn import DEFAULT_CONFIG as DQN_DEFAULT_CONFIG
    print(f"DQN's default config is: {DQN_DEFAULT_CONFIG}")
    # Plugging in Ray Tune.
    # Note that this is the recommended way to run any experiments with RLlib.
    # Reasons:
    # - Tune allows you to do hyperparameter tuning in a user-friendly way
    #   and at large scale!
    # - Tune automatically allocates needed resources for the different
    #   hyperparam trials and experiment runs.
    from ray import tune
    # Now that we will run things "automatically" through tune, we have to
    # define one or more stopping criteria.
    stop = {
        # explain that keys here can be anything present in the above print(trainer.train())
        "training_iteration": 5,
        "episode_reward_mean": 9999.9,
    }
    # "PPO" is a registered name that points to RLlib's PPOTrainer.
    # See `ray/rllib/agents/registry.py`
    # Run our simple experiment until one of the stop criteria is met.
    tune.run("PPO", config=config, stop=stop)

    # Updating an algo's default config dict and adding hyperparameter tuning
    # options to it.
    # Note: Hyperparameter tuning options (e.g. grid_search) will only work,
    # if we run these configs via `tune.run`.
    config.update(
        {
            # Try 2 different learning rates.
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
        stop=stop,
        checkpoint_at_end=True,  # create a checkpoint when done.
        checkpoint_freq=1,  # create a checkpoint on every iteration.
    )
    print(results)

    # 6) Using Anyscale's infinite laptop to start an experiment.
    # We will try to learn a more complex multi-agent environment
    # using a Griddly multi-agent environment.
    # We will check on the results later in this tutorial to see,
    # whether RLlib was able to learn it.
    # ...

    # 7) -> exercise_2.py: Exercise #2.
