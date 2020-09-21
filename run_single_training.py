from rayleigh_benard_environment import RayleighBenardEnvironment
import sympy
import argparse
from tensorforce import Agent, Runner

import tensorflow as tf
tf.get_logger().setLevel('ERROR')

parser = argparse.ArgumentParser()
parser.add_argument("--num-episodes", type = int, required = True)

args = parser.parse_args()

num_episodes = args.num_episodes


savefolder = "saver_data"
agent_name = "RB_ppo_agent_single"

#environment configuration
num_dt_between_actions = 10
max_episode_timesteps = 1000
num_state_points = 80
num_actions = 20

x, y, t = sympy.symbols('x,y,t', real=True)

#RB config
RB_config = {
    'N': (100, 250),
    'Ra': 1000.,
    "Pr": 0.7,
    'dt': 0.01,
    'filename': 'RB100',
    'conv': 1,
    'modplot': 100,
    'modsave': 50,
        'bcT': (sympy.sin((t+x)), 0),
    'family': 'C',
    'quad': 'GC'
    }


env = RayleighBenardEnvironment(
    num_dt_between_actions = num_dt_between_actions,
    max_episode_timesteps = max_episode_timesteps,
    num_state_points = num_state_points,
    num_actions = num_actions,
    RB_config = RB_config)

network = [dict(type='dense', size=512), dict(type='dense', size=512)]
agent = Agent.create(
    # Agent + Environment
    agent='ppo', environment=env,
    #max_episode_timesteps=nb_actuations,
    # TODO: nb_actuations could be specified by Environment.max_episode_timesteps() if it makes sense...
    # Network
    network=network,
    # Optimization
    batch_size=20, learning_rate=1e-3, subsampling_fraction=0.2, optimization_steps=25,
    # Reward estimation
    likelihood_ratio_clipping=0.2, estimate_terminal=True,  # ???
    # TODO: gae_lambda=0.97 doesn't currently exist
    # Critic
    critic_network=network,
    critic_optimizer=dict(
        type='multi_step', num_steps=5,
        optimizer=dict(type='adam', learning_rate=1e-3)
    ),
    # Regularization
    entropy_regularization=0.01,
    # TensorFlow etc
    parallel_interactions=1
    )

runner = Runner(agent = agent, environment=env)
runner.run(num_episodes = num_episodes)
