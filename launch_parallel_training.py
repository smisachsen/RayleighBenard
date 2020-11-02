import argparse
import os
import sys
import csv
import socket
import numpy as np

from tensorforce import Agent,Runner

from simulation_base.rayleigh_benard_environment import RayleighBenardEnvironment, NUM_DT_BETWEEN_ACTIONS
from RemoteEnvironmentClient import RemoteEnvironmentClient


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
# use_best_model = True

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
    agent='ppo',
    environment=example_environment,
    # TODO: nb_actuations could be specified by Environment.max_episode_timesteps() if it makes sense...
    # Network
    network=network,
    # Optimization
    batch_size=20,
    learning_rate=1e-3,
    discount=0.999,
    entropy_regularization=0.01,
    parallel_interactions=number_servers,
    saver=dict(directory=os.path.join(os.getcwd(), 'saver_data')),
)

runner = Runner(
    agent=agent, environments=environments, num_parallel=number_servers,
    evaluation=True
    )

cwd = os.getcwd()
evaluation_folder = "env_" + str(number_servers - 1)
sys.path.append(cwd + evaluation_folder)
# out_drag_file = open("avg_drag.txt", "w")

runner.run(
    num_episodes=5, sync_episodes=True,
)
# out_drag_file.close()
runner.close()
print("running done")
