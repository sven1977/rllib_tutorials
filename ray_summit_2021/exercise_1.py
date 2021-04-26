import numpy as np

from environment import MultiAgentArena


# Exercise #1:
# ============
# Write an "environment loop" using our `MultiAgentArena` class.
# 1) Create an env object.
# 2) `reset` it to get the first observation.
# 3) `step` through the environment using a provided
#    "Trainer.compute_action([obs])" method to compute action dicts
#    (see below).
# 4) When an episode is done, remember to `reset()` the env before the
#    next call to `step()`.
# Good luck! :)


class Trainer(object):
    """Dummy Trainer class used in this exercise.

    Use its compute_action method to get a new action, given some environment
    observation.
    """

    def compute_action(self, obs):
        # Returns a random action.
        return {
            "agent1": np.random.randint(4),
            "agent2": np.random.randint(4)
        }


if __name__ == "__main__":

    trainer = Trainer()
    # Check, whether it's working.
    print(trainer.compute_action({"agent1": 0, "agent2": 1}))

    # Solution:
    env = MultiAgentArena(config={"width": 10, "height": 10})
    obs = env.reset()
    # Play through a single episode.
    done = False
    while not done:
        action = trainer.compute_action(obs)
        obs, reward, done, _ = env.step(action)
        if done["__all__"]:
            obs = env.reset()
        # Optional:
        # env.render()
