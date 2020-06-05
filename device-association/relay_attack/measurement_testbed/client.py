import numpy
import logging
import utils.ssl
from server import PacketType
from twisted.protocols import basic
from twisted.internet import protocol, ssl, reactor

class ClientRelayTest(basic.LineOnlyReceiver):
    
    def __init__(self, factory):
        self.factory = factory
        self.handle_methods = {
            PacketType.Relay_Test: self.handle_relay_test
        }
    
    def connectionMade(self):
        logging.info("client connection established")
        self.sendLine(self.create_init_msg())
    
    def lineReceived(self, data):
        packet_type = data[0:1]
        func = self.handle_methods.get(packet_type)
        func()
    
    def create_init_msg(self):
        return b''.join([
            PacketType.Relay_Init,
            "{:50}".format(self.factory.testbed),
            "{:03}".format(self.factory.position)
        ])
    
    def handle_relay_test(self, voltage_start_range=100, voltage_end_range=500):
        data = numpy.random.randint(voltage_start_range, voltage_end_range, self.factory.test_length)
        self.sendLine(b''.join([PacketType.Relay_Test, data.tostring()]))
    
class ClientRelayTestFactory(protocol.ClientFactory):
    
    def __init__(self, test_length, testbed, position):
        self.test_length = test_length
        self.testbed = testbed
        self.position = position
    
    def buildProtocol(self, addr):
        return ClientRelayTest(self)
    
    def clientConnectionLost(self, transport, reason):
        logging.debug("client connection lost")
        logging.debug(reason.getErrorMessage())
        reactor.crash() # important to rerun latency tests
    
    def clientConnectionFailed(self, transport, reason):
        logging.debug("client connection failed")
        logging.debug(reason.getErrorMessage())
        reactor.crash() # important to rerun latency tests
    
def start_tcp(hostname, port, test_length, testbed, position):
    logging.basicConfig(filename="relay-attack.log", level=logging.DEBUG)
    logging.info("############### start TCP")
    factory = ClientRelayTestFactory(test_length, testbed, position)
    reactor.connectTCP(hostname, port, factory)
    reactor.run()
    
def start_tls(hostname, tls_hostname, port, test_length, testbed, position):
    logging.basicConfig(filename="relay-attack.log", level=logging.DEBUG)
    factory = ClientRelayTestFactory(test_length, testbed, position)
    certData = utils.ssl.get_public_cert()
    authority = ssl.Certificate.loadPEM(certData)
    logging.info("############### start TLS")
    logging.info(authority)
    logging.info("hostname: " + hostname)
    logging.info("tls hostname: " + tls_hostname)
    options = ssl.optionsForClientTLS(tls_hostname, authority)
    reactor.connectSSL(hostname, port, factory, options)
    reactor.run(installSignalHandlers=False)
    