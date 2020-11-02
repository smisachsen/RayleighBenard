from rayleigh_benard_environment import RayleighBenardEnvironment
import numpy as np
from config import *

env = RayleighBenardEnvironment(num_dt_between_actions = NUM_DT_BETWEEN_ACTIONS,
    max_episode_timesteps = 200,
    num_state_points = NUM_STATE_POINTS,
    num_actions = 5,
    RB_config = RB_CONFIG)

done = False
states = env.reset()
actions = np.ones(5)*1.2
actions[0] = 0
i = 0
while not done:
    states, done, reward = env.execute(actions)
    print(i, done, reward)
    i+=1

env.RB.save_to_file()
