from socket_utils.socket_server import Server

def check_free_port(host, port, verbose=True):
    """Check if a given port is available."""
    sock = socket.socket()
    try:
        sock.bind((host, port))
        sock.close()
        print("host {} on port {} is AVAIL".format(host, port))
        return(True)
    except:
        print("host {} on port {} is BUSY".format(host, port))
        sock.close()
        return(False)

def launch_server(host, port, verbose, env):
    Server(tensorforce_environment = env, host=host, port=port, verbose=verbose)
