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

# Optionally: Add `render` method returning some img.
def render(self, mode=None):
    #return np.random.randint(0, 256, size=(300, 400, 3), dtype=np.uint8)
    field_size = 40

    if not hasattr(self, "viewer"):
        from gym.envs.classic_control import rendering
        self.viewer = Viewer(400, 400)
        self.fields = {}
        # Add our grid, and the two agents to the viewer.
        for i in range(self.width):
            l = i * field_size
            r = l + field_size
            for j in range(self.height):
                b = 400 - j * field_size - field_size
                t = b + field_size
                field = rendering.PolyLine([(l, b), (l, t), (r, t), (r, b)], close=True)
                field.set_color(.0, .0, .0)
                field.set_linewidth(1.0)
                self.fields[(j, i)] = field
                self.viewer.add_geom(field)

        agent1 = rendering.make_circle(radius=field_size // 2 - 4)
        agent1.set_color(.0, 0.8, 0.1)
        self.agent1_trans = rendering.Transform()
        agent1.add_attr(self.agent1_trans)
        agent2 = rendering.make_circle(radius=field_size // 2 - 4)
        agent2.set_color(.5, 0.1, 0.1)
        self.agent2_trans = rendering.Transform()
        agent2.add_attr(self.agent2_trans)
        self.viewer.add_geom(agent1)
        self.viewer.add_geom(agent2)

    # Mark those fields green that have been covered by agent1,
    # all others black.
    for i in range(self.width):
        for j in range(self.height):
            self.fields[(j, i)].set_color(.0, .0, .0)
            self.fields[(j, i)].set_linewidth(1.0)
    for (j, i) in self.agent1_visited_fields:
        self.fields[(j, i)].set_color(.1, .5, .1)
        self.fields[(j, i)].set_linewidth(5.0)

    # Edit the pole polygon vertex
    self.agent1_trans.set_translation(self.agent1_pos[1] * field_size + field_size / 2, 400 - (self.agent1_pos[0] * field_size + field_size / 2))
    self.agent2_trans.set_translation(self.agent2_pos[1] * field_size + field_size / 2, 400 - (self.agent2_pos[0] * field_size + field_size / 2))

    return self.viewer.render(return_rgb_array=True)#TODO mode == 'rgb_array')

