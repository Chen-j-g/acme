# -*- coding: utf-8 -*-
# @Time : 2023/4/9 11:02
# @Author : Jiageng Chen
import collections
from typing import Optional
from acme.agents.jax.dqn import losses
from absl import app
from acme.jax import experiments
from acme.utils import lp_utils, loggers
import launchpad as lp
from acme import specs
from acme import wrappers
from acme.agents.jax import dqn
from acme.jax import networks as networks_lib
from acme.jax import utils
import gym
import haiku as hk
import jax

logger_dict = collections.defaultdict(loggers.InMemoryLogger)

def logger_factory(
        name: str,
        steps_key: Optional[str] = None,
        task_id: Optional[int] = None,
) -> loggers.Logger:
    del steps_key, task_id
    return logger_dict[name]


def env_factory(seed):
    del seed
    """Loads the gym environment."""
    # Internal logic.
    env = gym.make('CartPole-v0')

    env = wrappers.GymWrapper(env)

    # Make the environment output single-precision floats.
    # We use this because most TPUs only work with float32.
    env = wrappers.SinglePrecisionWrapper(env)

    return env


def network_factory(environment_spec: specs.EnvironmentSpec) -> dqn.DQNNetworks:
    """Creates networks for training DQN on Atari."""

    def network(inputs):
        model = hk.Sequential([
            hk.Linear(128), jax.nn.relu,
            hk.Linear(256), jax.nn.relu,
            hk.Linear(environment_spec.actions.num_values)
        ])
        return model(inputs)

    network_hk = hk.without_apply_rng(hk.transform(network))
    obs = utils.add_batch_dim(utils.zeros_like(environment_spec.observations))
    network = networks_lib.FeedForwardNetwork(
        init=lambda rng: network_hk.init(rng, obs), apply=network_hk.apply)
    typed_network = networks_lib.non_stochastic_network_to_typed(network)
    return dqn.DQNNetworks(policy_network=typed_network)

def build_experiment_config():
    """Builds DQN experiment config which can be executed in different ways."""
    # Create an environment, grab the spec, and use it to create networks.
    # Construct the agent.
    config = dqn.DQNConfig(
        discount=0.99,
        learning_rate=1e-4,
        n_step=1,
        target_update_period=100,
        min_replay_size=1000,
        max_replay_size=1_000_000,
        samples_per_insert=1.0,
        batch_size=64)
    loss_fn = losses.QLearning(
        discount=config.discount, max_abs_reward=1.)

    dqn_builder = dqn.DQNBuilder(config, loss_fn=loss_fn, actor_backend='gpu')

    return experiments.ExperimentConfig(
        builder=dqn_builder,
        environment_factory=env_factory,
        network_factory=network_factory,
        seed=0,
        max_num_actor_steps=100_000)

def main(_):
    experiment_config = build_experiment_config()
    program = experiments.make_distributed_experiment(
        experiment=experiment_config,
        num_actors=8 if lp_utils.is_local_run() else 128)
    lp.launch(program, xm_resources=lp_utils.make_xm_docker_resources(program))


if __name__ == '__main__':
  app.run(main)