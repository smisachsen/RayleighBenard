import sympy

NUM_DT_BETWEEN_ACTIONS = 10
MAX_EPISODE_TIMESTEPS = 10
NUM_STATE_POINTS = 20
NUM_ACTIONS = 10
NUM_ENVS = 1
NUM_EPISODES = 1
AGENTS_JSON = ["agents/ppo.json"]
AGENT_NAMES = ["ppo_agent"]

x, y, tt = sympy.symbols('x,y,t', real=True)

RB_CONFIG = {
    'N': (100, 250),
    'Ra': 1000.,
    "Pr": 0.7,
    'dt': 0.01,
    'filename': 'RB100',
    'conv': 1,
    'modplot': 100,
    'modsave': 50,
        'bcT': (sympy.sin((tt+x)), 0),
    'family': 'C',
    'quad': 'GC'
}
