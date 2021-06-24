# Solution to Exercise #2:

tune_config["lr"] = 0.0001
tune_config["train_batch_size"] = 4000
tune_config["num_envs_per_worker"] = 5
tune_config["num_workers"] = 5

analysis = tune.run("PPO", config=tune_config, stop=stop, checkpoint_at_end=True, checkpoint_freq=5)


