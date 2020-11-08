from tensorforce.agents import Agent
from tensorforce.execution import ParallelRunner, Runner
from simulation_base.rayleigh_benard_environment import RayleighBenardEnvironment, MAX_EPISODE_TIMESTEPS

import os

env = RayleighBenardEnvironment()

network = [dict(type='dense', size=512), dict(type='dense', size=512)]

agent = Agent.create(
    # Agent + Environment
    agent='ppo', environment=example_environment, max_episode_timesteps=MAX_EPISODE_TIMESTEPS,
    # TODO: nb_actuations could be specified by Environment.max_episode_timesteps() if it makes sense...
    # Network
    network=network,
    # Optimization
    batch_size=20, learning_rate=1e-3, subsampling_fraction=0.2, optimization_steps=25,
    # Reward estimation
    likelihood_ratio_clipping=0.2,
    estimate_terminal=True,
    # TensorFlow etc
    parallel_interactions=number_servers, #in the case of use_best_model = True
    saver=dict(directory=os.path.join(os.getcwd(), datafolder))
)

runner = Runner(agent=agent, environment=env)
runner.run(
    num_episodes=1
    )