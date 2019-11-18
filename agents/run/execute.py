"""
    Executes series of experiments

̣__author__ = "Jonas Heitz"
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

import numpy as np
import gym
from ray.rllib.models import FullyConnectedNetwork, Model, ModelCatalog
from gym.spaces import Discrete, Box, Dict

import tensorflow as tf
import tensorflow.contrib.slim as slim
from ray.rllib.models.misc import normc_initializer
from ray.tune.registry import register_env

import ray
from ray import tune

from agents.run.models import  CustomModel,CustomModel2
from agents.run.masking_envs_cross import CrossVal0, CrossVal1, CrossVal2, CrossVal3
from agents.run.configs import SIMPLE_CONFIG, DOUBLE_PRIO, PPO_CONFIG




if __name__ == "__main__":
    # Can also register the env creator function explicitly with:
    # register_env("corridor", lambda config: SimpleCorridor(config))
    ray.init()
    ModelCatalog.register_custom_model("my_model", CustomModel)
    ModelCatalog.register_custom_model("my_model_deep", CustomModel2)
    register_env("CM1-postgres-card-job-masking-v0", lambda _: CrossVal0())
    register_env("CM1-postgres-card-job-masking-v1", lambda _: CrossVal1())
    register_env("CM1-postgres-card-job-masking-v2", lambda _: CrossVal2())
    register_env("CM1-postgres-card-job-masking-v3", lambda _: CrossVal3())



    iteration = range(0,6)
    cross = range(0,4)

    model = "my_model"


    CONFIGS = [SIMPLE_CONFIG,DOUBLE_PRIO]

    for cfg in CONFIGS:
        for j in cross: #crossval
            for i in iteration: #range(0, 5):
                print(i)
                tune.run(
                    "DQN",
                    checkpoint_at_end=True,
                    resources_per_trial={"cpu": 4, "gpu": 0},
                    stop={
                        "timesteps_total": 20000*2,
                    },
                    config=dict({
                        "env": "CM1-postgres-card-job-masking-v"+str(j),
                        "model": {
                            "custom_model": model,
                        },
                    }, **cfg),

                )
        model = "my_model_deep"

    cfg = PPO_CONFIG
    for j in cross: #crossval
       for i in iteration: #range(0, 5):
            print(i)
            tune.run(
                "PPO",
                checkpoint_at_end=True,
                resources_per_trial={"cpu": 4, "gpu": 0},
                stop={
                    "timesteps_total": 20000*2*5,
                },
                config=dict({
                    "env": "CM1-postgres-card-job-masking-v"+str(j),
                    "model": {
                        "custom_model": "my_model",
                    },
                }, **cfg),)
