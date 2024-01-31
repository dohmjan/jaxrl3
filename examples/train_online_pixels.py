#! /usr/bin/env python
import os
import pickle

import gymnasium as gym
from gymnasium.envs.registration import registry
import tqdm
import wandb
from absl import app, flags
from ml_collections import config_flags

from jaxrl3.agents import DrQLearner
from jaxrl3.data import MemoryEfficientReplayBuffer
from jaxrl3.evaluation import evaluate
from jaxrl3.wrappers import wrap_pixels, set_universal_seed

FLAGS = flags.FLAGS

flags.DEFINE_string("env_name", "cheetah-run-v0", "Environment name.")
flags.DEFINE_string("save_dir", "./tmp/", "Tensorboard logging dir.")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_integer("eval_episodes", 10, "Number of episodes used for evaluation.")
flags.DEFINE_integer("log_interval", 1000, "Logging interval.")
flags.DEFINE_integer("eval_interval", 5000, "Eval interval.")
flags.DEFINE_integer("batch_size", 256, "Mini batch size.")
flags.DEFINE_integer("max_steps", int(5e5), "Number of training steps.")
flags.DEFINE_integer(
    "start_training", int(1e3), "Number of training steps to start training."
)
flags.DEFINE_integer("image_size", 64, "Image size.")
flags.DEFINE_integer("num_stack", 3, "Stack frames.")
flags.DEFINE_integer(
    "replay_buffer_size", None, "Number of training steps to start training."
)
flags.DEFINE_integer(
    "action_repeat", None, "Action repeat, if None, uses 2 or PlaNet default values."
)
flags.DEFINE_boolean("tqdm", True, "Use tqdm progress bar.")
flags.DEFINE_boolean("save_video", False, "Save videos during evaluation.")
flags.DEFINE_boolean("save_buffer", False, "Save the replay buffer.")
config_flags.DEFINE_config_file(
    "config",
    "configs/drq_default.py",
    "File path to the training hyperparameter configuration.",
    lock_config=False,
)

PLANET_ACTION_REPEAT = {
    "cartpole-swingup-v0": 8,
    "reacher-easy-v0": 4,
    "cheetah-run-v0": 4,
    "finger-spi-n-0": 2,
    "ball_in_cup-catch-v0": 4,
    "walker-walk-v0": 2,
}


def main(_):
    wandb.init(project="jaxrl3_online_pixels")
    wandb.config.update(FLAGS)

    if FLAGS.action_repeat is not None:
        action_repeat = FLAGS.action_repeat
    else:
        action_repeat = PLANET_ACTION_REPEAT.get(FLAGS.env_name, 2)

    def check_env_id(env_id):
        dm_control_env_ids = [
            id
            for id in registry
            if id.startswith("dm_control/") and id != "dm_control/compatibility-env-v0"
        ]
        if not env_id.startswith("dm_control/"):
            for id in dm_control_env_ids:
                if env_id in id:
                    env_id = "dm_control/" + env_id
        if env_id not in registry:
            raise ValueError("Provide valid env id.")
        return env_id

    def make_and_wrap_env(env_id):
        env_id = check_env_id(env_id)

        if "quadruped" in env_id:
            camera_id = 2
        else:
            camera_id = 0

        render_kwargs = dict(camera_id=camera_id, height=FLAGS.image_size, width=FLAGS.image_size)
        if env_id.startswith("dm_control"):
            env = gym.make(env_id, render_mode="rgb_array", render_kwargs=render_kwargs)
        else:
            render_kwargs.pop("camera_id")
            env = gym.make(env_id, render_mode="rgb_array", **render_kwargs)

        return wrap_pixels(
            env,
            action_repeat=action_repeat,
            num_stack=FLAGS.num_stack,
        )

    env = make_and_wrap_env(FLAGS.env_name)
    env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=1)
    set_universal_seed(env, FLAGS.seed)

    eval_env = make_and_wrap_env(FLAGS.env_name)
    set_universal_seed(eval_env, FLAGS.seed + 2)

    kwargs = dict(FLAGS.config)
    agent = DrQLearner(
        FLAGS.seed, env.observation_space.sample(), env.action_space.sample(), **kwargs
    )
    replay_buffer_size = FLAGS.replay_buffer_size or FLAGS.max_steps // action_repeat
    replay_buffer = MemoryEfficientReplayBuffer(
        env.observation_space, env.action_space, replay_buffer_size
    )
    replay_buffer.seed(FLAGS.seed)
    replay_buffer_iterator = replay_buffer.get_iterator(
        sample_args={"batch_size": FLAGS.batch_size, "include_pixels": False}
    )

    observation, _ = env.reset()
    done = False
    for i in tqdm.tqdm(
        range(1, FLAGS.max_steps // action_repeat + 1),
        smoothing=0.1,
        disable=not FLAGS.tqdm,
    ):
        if i < FLAGS.start_training:
            action = env.action_space.sample()
        else:
            action = agent.sample_actions(observation)
        next_observation, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        if not terminated:
            mask = 1.0
        else:
            mask = 0.0

        replay_buffer.insert(
            dict(
                observations=observation,
                actions=action,
                rewards=reward,
                masks=mask,
                dones=done,
                next_observations=next_observation,
            )
        )
        observation = next_observation

        if done:
            observation, _ = env.reset()
            done = False
            for k, v in info["episode"].items():
                decode = {"r": "return", "l": "length", "t": "time"}
                wandb.log({f"training/{decode[k]}": v}, step=i * action_repeat)

        if i >= FLAGS.start_training:
            batch = next(replay_buffer_iterator)
            update_info = agent.update(batch)

            if i % FLAGS.log_interval == 0:
                for k, v in update_info.items():
                    wandb.log({f"training/{k}": v}, step=i * action_repeat)

        if i % FLAGS.eval_interval == 0:
            if FLAGS.save_buffer:
                dataset_folder = os.path.join("datasets")
                os.makedirs("datasets", exist_ok=True)
                dataset_file = os.path.join(dataset_folder, f"{FLAGS.env_name}")
                with open(dataset_file, "wb") as f:
                    pickle.dump(replay_buffer, f)

            eval_info = evaluate(agent, eval_env, num_episodes=FLAGS.eval_episodes)
            for k, v in eval_info.items():
                wandb.log({f"evaluation/{k}": v}, step=i * action_repeat)


if __name__ == "__main__":
    app.run(main)
