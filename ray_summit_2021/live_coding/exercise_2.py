# Solution to Exercise #2:

# Run for longer this time (not just 2 iterations) and try to reach 40.0 reward (sum of both agents).
stop = {
    "training_iteration": 200,
    "episode_reward_mean": 50.0,
}

# tune_config.update({
# ???
# })

# analysis = tune.run(...)

tune_config["lr"] = 0.00005
tune_config["train_batch_size"] = 4000
tune_config["num_envs_per_worker"] = 10
tune_config["num_workers"] = 5
tune_config["model"] = {"fcnet_hiddens": [512, 512]}

analysis = tune.run("PPO", config=tune_config, stop=stop, checkpoint_at_end=True, checkpoint_freq=10)
