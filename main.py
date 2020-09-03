import sympy
import argparse
from tensorforce import Agent, Runner

from rayleigh_benard_environment import RayleighBenardEnvironment
from optimal_agent_tuning import *
from config import *

"""
#### SETUP Environment ####
"""


parser = argparse.ArgumentParser(description = "Set num_episodes, max_episode_timesteps")
parser.add_argument("NUM_EPISODES", help = "number of episodes to run",
    type = int)
parser.add_argument("MAX_EPISODE_TIMESTEPS", help = "number of timesteps per episode before termination",
    type = int)

args = parser.parse_args()

envs = [RayleighBenardEnvironment(
    num_dt_between_actions = NUM_DT_BETWEEN_ACTIONS,
    max_episode_timesteps = args.MAX_EPISODE_TIMESTEPS,
    num_state_points = NUM_STATE_POINTS,
    num_actions = NUM_ACTIONS,
    RB_config = RB_CONFIG) for _ in range(NUM_ENVS)]

"""
#### SETUP AGENT ####
"""
run_agents(AGENTS_JSON, num_episodes = args.NUM_EPISODES,
    agent_names = AGENT_NAMES, environments = envs,
    output = True, save_directory = "data/")

"""
RUN AGENT
"""
