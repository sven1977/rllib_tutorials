# Solution to Exercise #1
# !LIVE CODING!
# Solution:
env = MultiAgentArena(config={"width": 10, "height": 10})
obs = env.reset()
# Play through a single episode.
done = {"__all__": False}
return_ag1 = return_ag2 = 0.0
num_episodes = 0
while num_episodes < 5:
    # Compute actions separately for each agent.
    action1 = dummy_trainer.compute_action(obs["agent1"])
    action2 = dummy_trainer.compute_action(obs["agent2"])

    # Send the action-dict to the env.
    obs, rewards, done, _ = env.step({"agent1": action1, "agent2": action2})
    return_ag1 += rewards["agent1"]
    return_ag2 += rewards["agent2"]    
    if done["__all__"]:
        print(f"Episode done. R1={return_ag1} R2={return_ag2}")
        num_episodes += 1
        return_ag1 = return_ag2 = 0.0
        obs = env.reset()

    # Optional: render.
    env.render()
    time.sleep(0.15)

# Shutdown the viewer (becomes unstable otherwise).
env.viewer.close()
