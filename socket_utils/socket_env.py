import socket
import time
from socket_utils.echo_server import EchoServer

from tensorforce.environments import Environment


class Client(Environment):

    def __init__(self, environment, port, host, verbose):
        self.port = port
        self.host = host
        self.environment = environment
        self.socket = socket.socket()
        self.socket.connect((host, port))
        self.verbose = verbose

        super().__init__()

        print('Connected to {}:{}'.format(self.host, self.port))


    def communicate_socket(self, request, data):
        to_send = EchoServer.encode_message(request, data)
        self.socket.send(to_send)

        received_msg = self.socket.recv(100_000)
        request, data = EchoServer.decode_message(received_msg)

        return (request, data)

    def states(self):
        return self.environment.states()

    def actions(self):
        return self.environment.actions()

    def max_episode_timesteps(self):
        return self.environment.max_episode_timesteps()

    def reset(self):
        print("reseting")
        _ = self.communicate_socket("RESET", 1)
        _, init_state = self.communicate_socket("STATE", 1)

        return init_state

    def execute(self, actions):
        self.communicate_socket("CONTROL", actions)
        self.communicate_socket("EVOLVE", 1)

        # obtain the next state
        _, next_state = self.communicate_socket("STATE", 1)

        # check if terminal
        _, terminal = self.communicate_socket("TERMINAL", 1)

        # get the reward
        _, reward = self.communicate_socket("REWARD", 1)

        # print("execute performed; state, terminal, reward:")
        # print(next_state)
        # print(terminal)
        # print(reward)

        return (next_state, terminal, reward)
