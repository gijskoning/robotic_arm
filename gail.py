"""Code imported from https://github.com/HumanCompatibleAI/imitation/blob/master/examples/quickstart.py""" 

"""Trains BC, GAIL and AIRL models on saved CartPole-v1 demonstrations."""

from os import environ
import pathlib
import pickle
import tempfile

import stable_baselines3 as sb3
import gym

from imitation.algorithms import bc
from imitation.algorithms.adversarial import airl, GAIL
from imitation.data import rollout
from imitation.util import logger, util

# Load pickled test demonstrations.
with open("expert_data", "rb") as f:
    # This is a list of `imitation.data.types.Trajectory`, where
    # every instance contains observations and actions for a single expert
    # demonstration.
    trajectories = pickle.load(f)

# Convert List[types.Trajectory] to an instance of `imitation.data.types.Transitions`.
# This is a more general dataclass containing unordered
# (observation, actions, next_observation) transitions.
transitions = rollout.flatten_trajectories(trajectories)

environment_name = ""
env = gym.make(environment_name)

tempdir = tempfile.TemporaryDirectory(prefix="quickstart")
tempdir_path = pathlib.Path(tempdir.name)
print(f"All Tensorboards and logging are being written inside {tempdir_path}/.")

# Train GAIL on expert data.
# GAIL, and AIRL also accept as `demonstrations` any Pytorch-style DataLoader that
# iterates over dictionaries containing observations, actions, and next_observations.
gail_logger = logger.configure(tempdir_path / "GAIL/")
gail_trainer = GAIL.GAIL(
    venv=env,
    demonstrations=transitions,
    demo_batch_size=32,
    gen_algo=sb3.PPO("CnnPolicy", env, verbose=1, n_steps=1024),
    custom_logger=gail_logger,
)
gail_trainer.train(total_timesteps=2048)

CNN_model = gail_trainer.gen_algo
