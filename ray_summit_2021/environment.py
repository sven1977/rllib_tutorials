import gym
from gym.spaces import Discrete
import numpy as np
import random

# NOTE: Do this only once we are reaching point 4.
from ray.rllib.env.multi_agent_env import MultiAgentEnv

# 2. Env
# NOTE: First make this a gym.Env, then - once we need to plug in RLlib -
# we have to simply change it to MultiAgentEnv.
class MultiAgentArena(MultiAgentEnv):  # gym.Env
    def __init__(self, config=None):
        config = config or {}
        self.width = config.get("width", 10)
        self.height = config.get("height", 10)

        # 0=up, 1=right, 2=down, 3=left.
        self.action_space = Discrete(4)
        self.observation_space = Discrete(self.width * self.height)
        # End an episode after this many timesteps.
        self.timestep_limit = config.get("ts", 100)
        # Reset env.
        self.reset()

    def reset(self):
        # Row-major coords.
        self.agent1_pos = [0, 0]
        self.agent2_pos = [self.height - 1, self.width - 1]
        # Reset agent1's visited states.
        self.agent1_visited_states = set()
        # How many timesteps have we done in this episode.
        self.timesteps = 0

        return self.get_obs()

    def step(self, action: dict):
        self.timesteps += 1
        # Determine, who is allowed to move first.
        agent1_first = random.random() > 0.5
        # Move first agent (could be agent 1 or 2).
        if agent1_first:
            r1, r2 = self.move(self.agent1_pos, action["agent1"], is_agent1=True)
            add = self.move(self.agent2_pos, action["agent2"], is_agent1=False)
        else:
            r1, r2 = self.move(self.agent2_pos, action["agent2"], is_agent1=False)
            add = self.move(self.agent1_pos, action["agent1"], is_agent1=True)
        r1 += add[0]
        r2 += add[1]

        obs = self.get_obs()

        reward = {"agent1": r1, "agent2": r2}

        done = self.timesteps >= self.timestep_limit
        done = {"agent1": done, "agent2": done, "__all__": done}

        return obs, reward, done, {}

    def get_obs(self):
        obs1 = self.agent1_pos[0] * self.width + \
               (self.agent1_pos[1] % self.width)
        obs2 = self.agent2_pos[0] * self.width + \
               (self.agent2_pos[1] % self.width)
        return {"agent1": obs1, "agent2": obs2}

    def move(self, coords, action, is_agent1):
        orig_coords = coords[:]
        # Change the row: 0=up (-1), 2=down (+1)
        coords[0] += -1 if action == 0 else 1 if action == 2 else 0
        # Change the column: 1=right (+1), 3=left (-1)
        coords[1] += 1 if action == 1 else -1 if action == 3 else 0

        # Solve collisions.
        # Make sure, we don't end up on the other agent's position.
        # If yes, don't move (we are blocked).
        if is_agent1 and coords == self.agent2_pos:
            coords[0], coords[1] = orig_coords
            # Agent2 blocked agent1 -> +1 for agent2.
            return 0.0, 1.0
        elif not is_agent1 and coords == self.agent1_pos:
            coords[0], coords[1] = orig_coords
            # Agent1 blocked agent2 -> No reward for either agent.
            return 0.0, 0.0

        # No agent blocking -> check walls.
        if coords[0] < 0:
            coords[0] = 0
        elif coords[0] >= self.height:
            coords[0] = self.height - 1
        if coords[1] < 0:
            coords[1] = 0
        elif coords[1] >= self.width:
            coords[1] = self.width - 1

        # If agent1 -> +1.0 if new tile covered.
        if is_agent1 and not tuple(coords) in self.agent1_visited_states:
            self.agent1_visited_states.add(tuple(coords))
            return 1.0, -0.1
        # No new tile for agent1 -> No reward for either agent.
        return 0.0, 0.0

    # Optionally: Add `render` method returning some img.
    def render(self, mode=None):
        return np.random.randint(0, 256, (20, 20, 3), dtype=np.uint8)


if __name__ == "__main__":
    # 3.
    # Exercise No1:
    # =============
    # Write an "environment loop" using our new Env class above.
    # 1) Create an env object.
    # 2) `reset` it to get the first obs.
    # 3) `step` through the env using a provided "Trainer.compute_action([obs])" method
    #    to compute action dicts.
    # 4) When an episode is done, remember to `reset()` the env before the
    #    next call to `step()`.
    # Good luck! :)

    class Trainer(object):
        def compute_action(self, obs):
            return {
                "agent1": np.random.randint(4),
                "agent2": np.random.randint(4)
            }
    trainer = Trainer()
    # Debug this.
    #print(trainer.compute_action({"agent1": 0, "agent2": 1}))

    # 3.a)
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
