#! /usr/bin/env python
import gymnasium as gym
from gymnasium.envs.registration import registry

import tqdm
import wandb
from absl import app, flags
from ml_collections import config_flags

from jaxrl2.agents import SACLearner
from jaxrl2.data import ReplayBuffer
from jaxrl2.evaluation import evaluate
from jaxrl2.wrappers import wrap_gym, set_universal_seed

FLAGS = flags.FLAGS

flags.DEFINE_string("env_name", "HalfCheetah-v4", "Environment name.")
flags.DEFINE_string("save_dir", "./tmp/", "Tensorboard logging dir.")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_integer("eval_episodes", 10, "Number of episodes used for evaluation.")
flags.DEFINE_integer("log_interval", 1000, "Logging interval.")
flags.DEFINE_integer("eval_interval", 5000, "Eval interval.")
flags.DEFINE_integer("batch_size", 256, "Mini batch size.")
flags.DEFINE_integer("max_steps", int(1e6), "Number of training steps.")
flags.DEFINE_integer(
    "start_training", int(1e4), "Number of training steps to start training."
)
flags.DEFINE_boolean("tqdm", True, "Use tqdm progress bar.")
flags.DEFINE_boolean("wandb", True, "Log wandb.")
flags.DEFINE_boolean("save_video", False, "Save videos during evaluation.")
config_flags.DEFINE_config_file(
    "config",
    "configs/sac_default.py",
    "File path to the training hyperparameter configuration.",
    lock_config=False,
)


def main(_):
    wandb.init(project="jaxrl2_online")
    wandb.config.update(FLAGS)

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
        env = gym.make(check_env_id(env_id))
        return wrap_gym(env, rescale_actions=True)

    env = make_and_wrap_env(FLAGS.env_name)
    env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=1)
    set_universal_seed(env, FLAGS.seed)

    eval_env = make_and_wrap_env(FLAGS.env_name)
    set_universal_seed(eval_env, FLAGS.seed + 2)

    kwargs = dict(FLAGS.config)
    agent = SACLearner(FLAGS.seed, env.observation_space, env.action_space, **kwargs)

    replay_buffer = ReplayBuffer(
        env.observation_space, env.action_space, FLAGS.max_steps
    )
    replay_buffer.seed(FLAGS.seed)

    observation, _ = env.reset()
    done = False
    for i in tqdm.tqdm(
        range(1, FLAGS.max_steps + 1), smoothing=0.1, disable=not FLAGS.tqdm
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
                wandb.log({f"training/{decode[k]}": v}, step=i)

        if i >= FLAGS.start_training:
            batch = replay_buffer.sample(FLAGS.batch_size)
            update_info = agent.update(batch)

            if i % FLAGS.log_interval == 0:
                for k, v in update_info.items():
                    wandb.log({f"training/{k}": v}, step=i)

        if i % FLAGS.eval_interval == 0:
            eval_info = evaluate(agent, eval_env, num_episodes=FLAGS.eval_episodes)
            for k, v in eval_info.items():
                wandb.log({f"evaluation/{k}": v}, step=i)


if __name__ == "__main__":
    app.run(main)
