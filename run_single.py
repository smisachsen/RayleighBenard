from simulation_base.rayleigh_benard_environment import *

from tensorforce import Agent,Runner
import os


env = RayleighBenardEnvironment()

network = [dict(type='dense', size=512), dict(type='dense', size=512)]

agent = Agent.create(
    # Agent + Environment
    agent='ppo',
    environment=env,
    # TODO: nb_actuations could be specified by Environment.max_episode_timesteps() if it makes sense...
    # Network
    network=network,
    # Optimization
    batch_size=20,
    learning_rate=1e-3,
    discount=0.999,
    entropy_regularization=0.01,
    saver=dict(directory=os.path.join(os.getcwd(), 'saver_data')),
)

runner = Runner(
    agent=agent, environment=env
    )
runner.run(num_episodes=100)
