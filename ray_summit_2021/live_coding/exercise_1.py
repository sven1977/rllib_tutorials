# Solution to Exercise #1
# ...

env = MultiAgentArena()
obs = env.reset()

while True:
    # Compute actions separately for each agent.
    a1 = dummy_trainer.compute_action(obs["agent1"])
    a2 = dummy_trainer.compute_action(obs["agent2"])

    # Send the action-dict to the env.
    obs, rewards, dones, _ = env.step({"agent1": a1, "agent2": a2})

    # Get a rendered image from the env.
    out.clear_output(wait=True)
    env.render()
    time.sleep(0.1)

    if dones["agent1"]:
        break
