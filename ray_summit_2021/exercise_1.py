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
    action = dummy_trainer.compute_action(obs)
    obs, rewards, done, _ = env.step(action)
    return_ag1 += rewards["agent1"]
    return_ag2 += rewards["agent2"]    
    if done["__all__"]:
        print(f"Episode done. R1={return_ag1} R2={return_ag2}")
        num_episodes += 1
        return_ag1 = return_ag2 = 0.0
        obs = env.reset()
    # Optional:
    env.render()

    import time
    time.sleep(0.15)

