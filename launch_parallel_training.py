import argparse
import os
import sys
import csv
import socket
import numpy as np
import os
import json

from datetime import datetime
from tensorforce.agents import Agent
from tensorforce.execution import ParallelRunner, Runner

from simulation_base.rayleigh_benard_environment import RayleighBenardEnvironment, MAX_EPISODE_TIMESTEPS
from RemoteEnvironmentClient import RemoteEnvironmentClient

num_episodes = 64

start=datetime.now()
datetime_start = now.strftime("%H:%M:%S")

datafolder = os.path.join(os.getcwd(), "data")

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

cwd = os.getcwd()
evaluation_folder = "env_" + str(number_servers - 1)
sys.path.append(cwd + evaluation_folder)

runner = ParallelRunner(agent=agent, environments=environments)
runner.run(
    num_episodes=num_episodes, sync_episodes=True
    )


end=datetime.now()
datetime_end = now.strftime("%H:%M:%S")


data = dict()
data["datetime_start"] = datetime_start
data["datetime_end"] = datetime_end
data["num_episodes"] = num_episodes
data["rewards"] = runner.episode_rewards

#dump rewards to datafolder
outfile = os.path.join(datafolder, "results.json")
with open(outfile, "w") as file:
    json.dump(data, file)


#run single evaluation and save to datafolder
env = RayleighBenardEnvironment()
single_runner = Runner(agent=agent, environment=env)
single_runner.run(num_episodes=1, evaluation=True)


env.RB.save_to_file(folderpath=datafolder) 

runner.close()
single_runner.close()



