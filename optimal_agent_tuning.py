import sys

from tensorforce import Runner, Agent

def run_agents(agent_json_list, num_episodes, agent_names, environment = None,
    environments = None, output = False, save_directory = None):

    env = environment if environment else environments[0]
    for agent_json, agent_name in zip(agent_json_list, agent_names):
        agent = Agent.create(agent_json, environment = env)

        if output:
            print(f"Running agent: {agent_name}")
        _run_agent(agent, num_episodes, environment = environment,
            environments = environments, agent_name = agent_name,
            save_directory = save_directory)

def _run_agent(agent, num_episodes, agent_name, environment = None, environments = None,
    save_directory = None):
    if environment is not None:
        runner = Runner(agent = agent, environment = environment)

    elif environments is not None:
        runner = Runner(agent = agent, environments = environments,
            num_parallel = len(environments))

    else:
        print("environment or environments must be != None")
        sys.exit()

    directory = save_directory


    runner.run(num_episodes = num_episodes)
    agent.save(directory = save_directory, filename = agent_name)
