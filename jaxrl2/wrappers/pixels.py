from typing import Optional

import gymnasium as gym
from gymnasium.wrappers import ClipAction, FlattenObservation, PixelObservationWrapper, RescaleAction

from jaxrl2.wrappers.frame_stack import FrameStack
from jaxrl2.wrappers.repeat_action import RepeatAction


def wrap_pixels(
    env: gym.Env,
    action_repeat: int,
    image_size: int = 84,
    num_stack: Optional[int] = 3,
    camera_id: int = 0,
) -> gym.Env:
    if "dm_control" in env.spec.id:
        env.spec.max_episode_steps = 1000
        # dm_control envs have dict obs space
        env = FlattenObservation(env)

    if action_repeat > 1:
        env = RepeatAction(env, action_repeat)

    env = RescaleAction(env, -1, 1)

    env = PixelObservationWrapper(env, pixels_only=True)

    env = FrameStack(env, num_stack=num_stack)

    env = ClipAction(env)

    return env
