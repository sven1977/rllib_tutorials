pol = rllib_trainer.get_policy()

print(f"Policy: {pol}")
print(f"Observation-space: {pol.observation_space}")
print(f"Action-space: {pol.action_space}")
print(f"Action distribution class: {pol.dist_class}")
print(f"Model: {pol.model}")

# Create a fake numpy B=1 (single) observation consisting of both agents positions ("one-hot'd" and "concat'd").
from ray.rllib.utils.numpy import one_hot
single_obs = np.concatenate([one_hot(0, depth=100), one_hot(99, depth=100)])
single_obs = np.array([single_obs])
#single_obs.shape

# Generate the Model's output.
out, state_out = pol.model({"obs": single_obs})

# tf1.x (static graph) -> Need to run this through a tf session.
numpy_out = pol.get_session().run(out)

# RLlib then passes the model's output to the policy's "action distribution" to sample an action.
action_dist = pol.dist_class(out)
action = action_dist.sample()

# Show us the actual action.
pol.get_session().run(action)
