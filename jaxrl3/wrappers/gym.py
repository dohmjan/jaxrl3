import numpy as np
import gymnasium as gym
from gymnasium.wrappers import ClipAction, FlattenObservation, RescaleAction

from jaxrl3.wrappers.single_precision import SinglePrecision


def wrap_gym(env: gym.Env, rescale_actions: bool = True) -> gym.Env:
    env = SinglePrecision(env)
    if rescale_actions:
        env = RescaleAction(env, -1, 1)

    if isinstance(env.observation_space, gym.spaces.Dict):
        env = FlattenObservation(env)

    env = ClipAction(env)

    if "dm_control" in env.spec.id:
        env.spec.max_episode_steps = 1000

    return env


def set_universal_seed(env: gym.Env, seed: int):
    _, _ = env.reset(seed=seed)
    env.observation_space.seed(seed)
    env.action_space.seed(seed)
    np.random.seed(seed)
