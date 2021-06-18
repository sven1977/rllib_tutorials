# Solution to Exercise #2:

# Update our config and set it up for 2x tune grid-searches (leading to 4 parallel trials in total).
config.update({
    "lr": tune.grid_search([0.0001, 0.0005]),
    "train_batch_size": tune.grid_search([2000, 3000]),
    "num_envs_per_worker": 5,
    # Change our model to be simpler.
    "model": {
        "fcnet_hiddens": [128, 128],
    },
})

# Run the experiment.
analysis = tune.run(
    "PPO",
    config=config,
    stop={"episode_reward_mean": 15.0, "training_iteration": 100},
    checkpoint_freq=10,
    checkpoint_at_end=True,
)
