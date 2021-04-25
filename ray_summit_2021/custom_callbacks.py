import numpy as np

from ray.rllib.agents.callbacks import DefaultCallbacks


class MyCallbacks(DefaultCallbacks):
    def on_episode_start(self, *, worker, base_env,
                         policies,
                         episode, env_index, **kwargs):
        pass

    def on_episode_step(self, *, worker, base_env,
                        episode, env_index, **kwargs):
        # Make sure this episode is ongoing.
        assert episode.length > 0, \
            "ERROR: `on_episode_step()` callback should not be called right " \
            "after env reset!"
        pole_angle = abs(episode.last_observation_for()[2])
        raw_angle = abs(episode.last_raw_obs_for()[2])
        assert pole_angle == raw_angle
        episode.user_data["pole_angles"].append(pole_angle)

    def on_episode_end(self, *, worker, base_env,
                       policies, episode,
                       env_index, **kwargs):
        # Make sure this episode is really done.
        assert episode.batch_builder.policy_collectors[
            "default_policy"].buffers["dones"][-1], \
            "ERROR: `on_episode_end()` should only be called " \
            "after episode is done!"
        pole_angle = np.mean(episode.user_data["pole_angles"])
        print("episode {} (env-idx={}) ended with length {} and pole "
              "angles {}".format(episode.episode_id, env_index, episode.length,
                                 pole_angle))
        episode.custom_metrics["pole_angle"] = pole_angle
        episode.hist_data["pole_angles"] = episode.user_data["pole_angles"]

    def on_sample_end(self, *, worker, samples, **kwargs):
        pass

    def on_train_result(self, *, trainer, result: dict, **kwargs):
        pass

    def on_learn_on_batch(self, *, policy, train_batch, result: dict, **kwargs):
        pass

    def on_postprocess_trajectory(
            self, *, worker, episode,
            agent_id, policy_id, policies,
            postprocessed_batch,
            original_batches, **kwargs):
        pass
