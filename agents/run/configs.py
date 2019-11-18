"""
    Config dict for vanilla DQN(SIMPLE_CONFIG), DDQN with Priority Replay(DOUBLE_PRIO) and PPO(PPO_CONFIG)

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
from ray.tune import grid_search
import json


OPTIMIZER_SHARED_CONFIGS = [
    "buffer_size", "prioritized_replay", "prioritized_replay_alpha",
    "prioritized_replay_beta", "schedule_max_timesteps",
    "beta_annealing_fraction", "final_prioritized_replay_beta",
    "prioritized_replay_eps", "sample_batch_size", "train_batch_size",
    "learning_starts"
]

SIMPLE_CONFIG = {
    # === Model ===
    # Number of atoms for representing the distribution of return. When
    # this is greater than 1, distributional Q-learning is used.
    # the discrete supports are bounded by v_min and v_max
    "num_atoms": 0,
    "v_min": -10.0,
    "v_max": 10.0,
    # Whether to use noisy network
    "noisy": False,
    # control the initial value of noisy nets
    "sigma0": 0.5,
    # Whether to use dueling dqn
    "dueling": False,
    # Whether to use double dqn
    "double_q": False,
    # Postprocess model outputs with these hidden layers to compute the
    # state and action values. See also the model config in catalog.py.
    "hiddens": [],
    # N-step Q learning
    "n_step": 2,

    # === Evaluation ===
    # Evaluate with epsilon=0 every `evaluation_interval` training iterations.
    # The evaluation stats will be reported under the "evaluation" metric key.
    # Note that evaluation is currently not parallelized, and that for Ape-X
    # metrics are already only reported for the lowest epsilon workers.
    "evaluation_interval": None,
    # Number of episodes to run per evaluation period.
    "evaluation_num_episodes": 10,

    # === Exploration ===
    # Max num timesteps for annealing schedules. Exploration is annealed from
    # 1.0 to exploration_fraction over this number of timesteps scaled by
    # exploration_fraction
    "schedule_max_timesteps": 100000,
    # Number of env steps to optimize for before returning
    "timesteps_per_iteration": 1000,
    # Fraction of entire training period over which the exploration rate is
    # annealed
    "exploration_fraction": 0.1,
    # Final value of random action probability
    "exploration_final_eps": 0.02,
    # Update the target network every `target_network_update_freq` steps.
    "target_network_update_freq": 500,
    # Use softmax for sampling actions. Required for off policy estimation.
    "soft_q": False,
    # Softmax temperature. Q values are divided by this value prior to softmax.
    # Softmax approaches argmax as the temperature drops to zero.
    "softmax_temp": 1.0,
    # If True parameter space noise will be used for exploration
    # See https://blog.openai.com/better-exploration-with-parameter-noise/
    "parameter_noise": False,

    # === Replay buffer ===
    # Size of the replay buffer. Note that if async_updates is set, then
    # each worker will have a replay buffer of this size.
    "buffer_size": 50000*100,
    # If True prioritized replay buffer will be used.
    "prioritized_replay": False,#True,
    # Alpha parameter for prioritized replay buffer.
    "prioritized_replay_alpha": 0.6,
    # Beta parameter for sampling from prioritized replay buffer.
    "prioritized_replay_beta": 0.4,
    # Fraction of entire training period over which the beta parameter is
    # annealed
    "beta_annealing_fraction": 0.2,
    # Final value of beta
    "final_prioritized_replay_beta": 0.4,
    # Epsilon to add to the TD errors when updating priorities.
    "prioritized_replay_eps": 1e-6,
    # Whether to LZ4 compress observations
    "compress_observations": False,

    # === Optimization ===
    # Learning rate for adam optimizer
    "lr": 5e-4,
    # Adam epsilon hyper parameter
    "adam_epsilon": 1e-8,
    # If not None, clip gradients during optimization at this value
    "grad_norm_clipping": None,#40,
    # How many steps of the model to sample before learning starts.
    "learning_starts": 1000,
    # Update the replay buffer with this many samples at once. Note that
    # this setting applies per-worker if num_workers > 1.
    "sample_batch_size": 1,
    # Size of a batched sampled from replay buffer for training. Note that
    # if async_updates is set, then each worker returns gradients for a
    # batch of this size.
    "train_batch_size": 32,

    # === Parallelism ===
    # Number of workers for collecting samples with. This only makes sense
    # to increase if your environment is particularly slow to sample, or if
    # you"re using the Async or Ape-X optimizers.
    "num_workers": 0,
    # Optimizer class to use.
    "optimizer_class": "SyncReplayOptimizer",
    # Whether to use a distribution of epsilons across workers for exploration.
    "per_worker_exploration": False,
    # Whether to compute priorities on workers.
    "worker_side_prioritization": False,
    # Prevent iterations from going lower than this time span
    "min_iter_time_s": 1,

}


DOUBLE_PRIO = SIMPLE_CONFIG.copy()
DOUBLE_PRIO["schedule_max_timesteps"] = 100000*2
DOUBLE_PRIO["timesteps_per_iteration"] = 1000
DOUBLE_PRIO["exploration_fraction"] = 0.1
DOUBLE_PRIO["exploration_final_eps"] = 0.00
DOUBLE_PRIO["target_network_update_freq"] = 32000
DOUBLE_PRIO["buffer_size"] = 100000*10*5
DOUBLE_PRIO["lr"] = 5e-4
DOUBLE_PRIO["adam_epsilon"] = 1.5*10e-4
DOUBLE_PRIO["grad_norm_clipping"] = 40
DOUBLE_PRIO["learning_starts"] = 80000*2
DOUBLE_PRIO['double_q']=True
DOUBLE_PRIO['prioritized_replay']=True




PPO_CONFIG = {
    # If true, use the Generalized Advantage Estimator (GAE)
    # with a value function, see https://arxiv.org/pdf/1506.02438.pdf.
    "use_gae": True,
    # GAE(lambda) parameter
    "lambda": 1.0,
    # Initial coefficient for KL divergence
    "kl_coeff": 0.2,
    # Size of batches collected from each worker
    "sample_batch_size": 200,
    # Number of timesteps collected for each SGD round
    "train_batch_size": 4000,
    # Total SGD batch size across all devices for SGD
    "sgd_minibatch_size": 128,
    # Number of SGD iterations in each outer loop
    "num_sgd_iter": 30,
    # Stepsize of SGD
    "lr": 0.03, #0.01, #5e-5,
    # Learning rate schedule
    "lr_schedule": None,
    # Share layers for value function
    #"vf_share_layers": False,
    # Coefficient of the value function loss
    "vf_loss_coeff": 1.0,
    # Coefficient of the entropy regularizer
    "entropy_coeff": 0.0,
    # PPO clip parameter
    "clip_param": 0.3,
    # Clip param for the value function. Note that this is sensitive to the
    # scale of the rewards. If your expected V is large, increase this.
    "vf_clip_param": 10.0,
    # If specified, clip the global norm of gradients by this amount
    "grad_clip": None,
    # Target value for KL divergence
    "kl_target": 0.01,
    # Whether to rollout "complete_episodes" or "truncate_episodes"
    "batch_mode": "complete_episodes", #"truncate_episodes",
    # Which observation filter to apply to the observation
    #"observation_filter": "NoFilter",
    # Uses the sync samples optimizer instead of the multi-gpu one. This does
    # not support minibatches.
    "simple_optimizer": False,
    # (Deprecated) Use the sampling behavior as of 0.6, which launches extra
    # sampling tasks for performance but can waste a large portion of samples.
    "straggler_mitigation": False,

    #because of action masking
    "observation_filter": "NoFilter",  # don't filter the action list
    "vf_share_layers": True,  # don't create duplicate value model
}