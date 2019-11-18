from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import gym
from ray.rllib.models import FullyConnectedNetwork, Model, ModelCatalog
from gym.spaces import Discrete, Box
from gym.utils import seeding

import tensorflow as tf
import tensorflow.contrib.slim as slim
from ray.rllib.models.misc import normc_initializer

import ray
from ray import tune
from ray.tune.registry import register_env
from ray.tune import grid_search


class SimpleCorridor(gym.Env):
    """Example of a custom env in which you have to walk down a corridor.
    You can configure the length of the corridor via the env config."""
    actions = []
    action_obj = []
    action_list = []
    action_space = None
    observation_space = None
    obs = []
    reward_range = [float(0), float(1)]

    def __init__(self, config):
        self.end_pos = 9 #config["corridor_length"]
        self.cur_pos = 0
        self.action_space = Discrete(2)
        self.observation_space = Box(
            0.0, self.end_pos, shape=(1, ), dtype=np.float32)

    def reset(self):
        self.cur_pos = 0
        return [self.cur_pos]

    def step(self, action):
        assert action in [0, 1], action
        if action == 0 and self.cur_pos > 0:
            self.cur_pos -= 1
        elif action == 1:
            self.cur_pos += 1
        done = self.cur_pos >= self.end_pos
        return [self.cur_pos], 1 if done else 0, done, {}

    def render(self, mode='human', close=False):
        return self.cur_pos

    def close(self):
        return

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

class CustomModel(Model):
    """Example of a custom model.
    This model just delegates to the built-in fcnet.
    """

    def _build_layers_v2(self, input_dict, num_outputs, options):
        self.obs_in = input_dict["obs"]
        print("aiaiaiaiaiaiaiai")
        print(input_dict)
        print(num_outputs)
        print(options)
        self.fcnet = FullyConnectedNetwork(input_dict, self.obs_space,
                                           self.action_space, num_outputs,
                                           options)
        '''
        print(self.fcnet)
        print("QQQQQQQQQQQQQQQQQQQQQQQQQ")
        flatten = tf.layers.flatten(
            self.fcnet.last_layer,
            name=None,
            data_format='channels_last'
        )
        label = "fc_out2"
        output = slim.fully_connected(
            flatten,
            num_outputs,
            weights_initializer=normc_initializer(0.01),
            activation_fn=None,
            scope=label)
        return output, flatten #self.fcnet.outputs, self.fcnet.last_layer
        '''
        return self.fcnet.outputs, self.fcnet.last_layer


if __name__ == "__main__":
    # Can also register the env creator function explicitly with:
    #register_env("corridor", lambda config: SimpleCorridor(config))
    ray.init()
    ModelCatalog.register_custom_model("my_model", CustomModel)
    tune.run(
        "DQN",
        stop={
            "timesteps_total": 1 #10 #10000,
        },
        config={
            "env": "simple-corridor-ray-v0", #"CM1-postgres-card-ray-v0", #SimpleCorridor,  # or "corridor" if registered above
            "model": {
                "custom_model": "my_model",
                #"conv_filters": [3,9],
            },
            "lr": 1e-2, #grid_search([1e-2, 1e-4, 1e-6]),  # try different lrs
            "num_workers": 1,  # parallelism
        },
)
'''
import ray
import ray.rllib.agents.dqn as dqn
from ray.tune.logger import pretty_print

ray.init()
config = dqn.DEFAULT_CONFIG.copy()
config["num_gpus"] = 0
config["num_workers"] = 1
config["conv_filter"] = [3,9]
trainer = dqn.DQNAgent(config=config, env="CM1-postgres-card-ray-v0")


# Can optionally call trainer.restore(path) to load a checkpoint.

for i in range(1000):
   # Perform one iteration of training the policy with DQN
   result = trainer.train()
   print(pretty_print(result))

   if i % 100 == 0:
       checkpoint = trainer.save()
       print("checkpoint saved at", checkpoint)
'''