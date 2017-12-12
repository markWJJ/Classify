from xmlrpc.server import SimpleXMLRPCServer
from xmlrpc.server import SimpleXMLRPCRequestHandler
import logging
logging.basicConfig(level=logging.INFO)
#_logger=logging.getLogger("server")
from Model.BayesDemo import Bayes

# Restrict to a particular path.
class RequestHandler(SimpleXMLRPCRequestHandler):
    rpc_paths = ('/RPC2',)

# Create server
server = SimpleXMLRPCServer(("192.168.100.253", 18112),
                            requestHandler=RequestHandler)
server.register_introspection_functions()

# Register pow() function; this will use the value of
# pow.__name__ as the name, which is just 'pow'.
# server.register_function(pow)

# Register a function under a different name


def getBayesClassifer(sentence):
    bayes=Bayes()
    res=bayes.perdict(sentence)
    return res
server.register_function(getBayesClassifer, 'getBayesClassifer')

# _logger.info("server open ")
print("server open")
# Run the server's main loop
server.serve_forever()
