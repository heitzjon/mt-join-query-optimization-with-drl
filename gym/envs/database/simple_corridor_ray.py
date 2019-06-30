import gym
from gym import error, spaces, utils
from gym.utils import seeding
import os
import numpy as np
import random
from itertools import permutations
from gym.spaces import Discrete, Box

"""
    Simple example for environment tests
"""

class SimpleCorridor(gym.Env):
  actions = []
  action_obj = []
  action_list = []
  action_space = None
  observation_space = None
  obs = []
  reward_range = [float(0), float(1)]

  def __init__(self):#, config):
    self.end_pos = 9 #config["corridor_length"]
    self.cur_pos = 0
    self.action_space = Discrete(2)
    self.observation_space = Box(0.0, self.end_pos, shape=(1,), dtype=np.float32)

  def reset(self):
    self.cur_pos = 0
    return [self.cur_pos]#, 0, False

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





