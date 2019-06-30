"""
        CrossVal0-3:
            Environments extensions for action masking and cross validation experiments

"""
̣__author__ = "Jonas Heitz"
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
from ray.tune import grid_search
import json

class CrossVal0(gym.Env):

    def __init__(self):
        self.wrapped = gym.make("CM1-postgres-card-job-cross-v0")
        self.action_space = self.wrapped.action_space
        self.reward_range = self.wrapped.reward_range
        self.observation_space = Dict({
            "action_mask": Box(0, 1, shape=(self.wrapped.action_space.n, )),
            "db": self.wrapped.observation_space,
        })
        self.counter=0
        self.scounter=0

    def update_avail_actions(self):
        self.validActions = self.wrapped.getValidActions()
        self.action_mask = np.array([0.] * self.action_space.n)
        for i in self.validActions:
            self.action_mask[i] = 1


    def reset(self):
        self.counter+=1
        self.scounter=0
        obs =self.wrapped.reset()
        self.update_avail_actions()
        return {
            "action_mask": self.action_mask,
            "db": obs,
        }

    def step(self, action):
        self.scounter+=1
        if action not in self.validActions:
            #self.failurecounter+=1
            print("failure steps:")

        orig_obs, rew, done, info = self.wrapped.step(action)
        self.update_avail_actions()
        obs = {
            "action_mask": self.action_mask,
            "db": orig_obs,
        }
        return obs, rew, done, info

class CrossVal1(gym.Env):

    def __init__(self):
        self.wrapped = gym.make("CM1-postgres-card-job-cross-v1")
        self.action_space = self.wrapped.action_space
        self.reward_range = self.wrapped.reward_range
        self.observation_space = Dict({
            "action_mask": Box(0, 1, shape=(self.wrapped.action_space.n, )),
            "db": self.wrapped.observation_space,
        })
        self.counter=0
        self.scounter=0

    def update_avail_actions(self):
        self.validActions = self.wrapped.getValidActions()
        self.action_mask = np.array([0.] * self.action_space.n)
        for i in self.validActions:
            self.action_mask[i] = 1


    def reset(self):
        self.counter+=1
        self.scounter=0
        obs =self.wrapped.reset()
        self.update_avail_actions()
        return {
            "action_mask": self.action_mask,
            "db": obs,
        }

    def step(self, action):
        self.scounter+=1
        if action not in self.validActions:
            #self.failurecounter+=1
            print("failure steps:")

        orig_obs, rew, done, info = self.wrapped.step(action)
        self.update_avail_actions()
        obs = {
            "action_mask": self.action_mask,
            "db": orig_obs,
        }
        return obs, rew, done, info

class CrossVal2(gym.Env):

    def __init__(self):
        self.wrapped = gym.make("CM1-postgres-card-job-cross-v2")
        self.action_space = self.wrapped.action_space
        self.reward_range = self.wrapped.reward_range
        self.observation_space = Dict({
            "action_mask": Box(0, 1, shape=(self.wrapped.action_space.n, )),
            "db": self.wrapped.observation_space,
        })
        self.counter=0
        self.scounter=0

    def update_avail_actions(self):
        self.validActions = self.wrapped.getValidActions()
        self.action_mask = np.array([0.] * self.action_space.n)
        for i in self.validActions:
            self.action_mask[i] = 1


    def reset(self):
        self.counter+=1
        self.scounter=0
        obs =self.wrapped.reset()
        self.update_avail_actions()
        return {
            "action_mask": self.action_mask,
            "db": obs,
        }

    def step(self, action):
        self.scounter+=1
        if action not in self.validActions:
            #self.failurecounter+=1
            print("failure steps:")

        orig_obs, rew, done, info = self.wrapped.step(action)
        self.update_avail_actions()
        obs = {
            "action_mask": self.action_mask,
            "db": orig_obs,
        }
        return obs, rew, done, info

class CrossVal3(gym.Env):

    def __init__(self):
        self.wrapped = gym.make("CM1-postgres-card-job-cross-v3")
        self.action_space = self.wrapped.action_space
        self.reward_range = self.wrapped.reward_range
        self.observation_space = Dict({
            "action_mask": Box(0, 1, shape=(self.wrapped.action_space.n, )),
            "db": self.wrapped.observation_space,
        })
        self.counter=0
        self.scounter=0

    def update_avail_actions(self):
        self.validActions = self.wrapped.getValidActions()
        self.action_mask = np.array([0.] * self.action_space.n)
        for i in self.validActions:
            self.action_mask[i] = 1


    def reset(self):
        self.counter+=1
        self.scounter=0
        obs =self.wrapped.reset()
        self.update_avail_actions()
        return {
            "action_mask": self.action_mask,
            "db": obs,
        }

    def step(self, action):
        self.scounter+=1
        if action not in self.validActions:
            #self.failurecounter+=1
            print("failure steps:")

        orig_obs, rew, done, info = self.wrapped.step(action)
        self.update_avail_actions()
        obs = {
            "action_mask": self.action_mask,
            "db": orig_obs,
        }
        return obs, rew, done, info

