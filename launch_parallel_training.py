import argparse
import os
import sys
import csv
import socket
import numpy as np
import os

from tensorforce.agents import Agent
from tensorforce.execution import ParallelRunner

from simulation_base.rayleigh_benard_environment import RayleighBenardEnvironment, MAX_EPISODE_TIMESTEPS
from RemoteEnvironmentClient import RemoteEnvironmentClient


agent_dir = "agent_save"
agent_filename = "ppo_agent"


ap = argparse.ArgumentParser()
ap.add_argument("-n", "--number-servers", required=True, help="number of servers to spawn", type=int)
ap.add_argument("-p", "--ports-start", required=True, help="the start of the range of ports to use", type=int)
ap.add_argument("-t", "--host", default="None", help="the host; default is local host; string either internet domain or IPv4", type=str)

args = vars(ap.parse_args())

number_servers = args["number_servers"]
ports_start = args["ports_start"]
host = args["host"]

if host == 'None':
    host = socket.gethostname()

example_environment = RayleighBenardEnvironment()
use_best_model = True

environments = []
for crrt_simu in range(number_servers):
    environments.append(RemoteEnvironmentClient(
        example_environment, verbose=0, port=ports_start + crrt_simu, host=host,
        timing_print=(crrt_simu == 0)
    ))

# if use_best_model:
#     evaluation_environment = environments.pop()
# else:
#     evaluation_environment = None

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
    saver=dict(directory=os.path.join(os.getcwd(), 'saver_data'))
)

runner = ParallelRunner(agent=agent, environments=environments, save_best_agent=os.path.join(os.getcwd() + "saved_agents"))

cwd = os.getcwd()
evaluation_folder = "env_" + str(number_servers - 1)
sys.path.append(cwd + evaluation_folder)
# out_drag_file = open("avg_drag.txt", "w")

runner.run(
    num_episodes=200, sync_episodes=True
    )
runner.close()

