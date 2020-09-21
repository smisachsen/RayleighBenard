import socket

from socket_utils.echo_server import EchoServer

class Server(EchoServer):
    def __init__(self, tensorforce_environment, host, port, verbose):
        self.host = host
        self.port = port
        self.verbose = verbose
        self.tensorforce_environment = tensorforce_environment


        self.socket_instance = socket.socket()
        self.socket_instance.bind((host, port))
        self.socket_instance.listen(1)

        self.state = None
        self.terminal = False
        self.reward = None

        EchoServer.__init__(self)

        connection = None
        self.done = False
        while not self.done:
            if connection is None:
                connection, self.address = self.socket_instance.accept()
                print('Got connection from {}'.format(self.address))

            else:
                message = connection.recv(100_000)
                response = self.handle_message(message)
                connection.send(response)

        self.socket.close()

    def SIMULATION_FINISHED(self, data):
        assert data == 1
        self.done = True
        print(f"ending simulation for {self.address}")


    def RESET(self, data):
        self.state = self.tensorforce_environment.reset()
        return(1)  # went fine

    def STATE(self, data):
        return(self.state)

    def TERMINAL(self, data):
        return(self.terminal)

    def REWARD(self, data):
        return(self.reward)

    def CONTROL(self, data):
        self.actions = data
        return(1)  # went fine

    def EVOLVE(self, data):
        self.state, self.terminal, self.reward = self.tensorforce_environment.execute(self.actions)
        return(1)  # went fine
