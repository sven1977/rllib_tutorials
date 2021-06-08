def _reset(self):
    # Row-major coords!
    self.agent1_pos = [0, 0]
    self.agent2_pos = [self.height - 1, self.width - 1]
    # Reset agent1's visited states.
    self.agent1_visited_states = set()
    # How many timesteps have we done in this episode.
    self.timesteps = 0

    return self._get_obs()

def _step(self, action: dict):
    # increase our time steps counter by 1.
    self.timesteps += 1

    # Determine, who is allowed to move first (50:50).
    if random.random() > 0.5:
        # events = [collision|new_field]
        events = self._move(self.agent1_pos, action["agent1"], is_agent1=True)
        events |= self._move(self.agent2_pos, action["agent2"], is_agent1=False)
    else:
        events = self._move(self.agent2_pos, action["agent2"], is_agent1=False)
        events |= self._move(self.agent1_pos, action["agent1"], is_agent1=True)

    # Determine rewards based on the collected events:
    rewards = {
        "agent1": -1.0 if "collision" in events else 1.0 if "new_field" in events else -0.5,
        "agent2": 1.0 if "collision" in events else -0.1,
    }
    # Get observations (based on new agent positions).
    obs = self._get_obs()

    # Generate a `done` dict (per-agent and total).
    # We are done only when we reach the time step limit.
    is_done = self.timesteps >= self.timestep_limit
    dones = {
        "agent1": is_done,
        "agent2": is_done,
        # special `__all__` key indicates that the episode is done for all agents.
        "__all__": is_done,
    }

    return obs, rewards, dones, {}
