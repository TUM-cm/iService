import os
import sys
import time
import logging
import utils.ssl
from enum import Enum
from twisted.python import log
from twisted.protocols import basic
from utils.serializer import DillSerializer
from twisted.internet import reactor, protocol, ssl

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

class PacketType(bytes, Enum):
    Relay_Init = b'\x01'
    Relay_Test = b'\x02'
    Relay_End = b'\x03'

class ServerRelayTest(basic.LineOnlyReceiver):
    
    def __init__(self, factory):
        self.factory = factory
        self.start_time = None
        self.handle_methods = {
            PacketType.Relay_Init: self.handle_relay_init,
            PacketType.Relay_Test: self.handle_relay_test
        }
    
    def connectionMade(self):
        logging.info("server connection established")
    
    def lineReceived(self, data):
        packet_type = data[:1]
        func = self.handle_methods.get(packet_type)
        if func != None:
            func(data)
    
    def handle_relay_init(self, data):
        logging.info("handle relay init")
        testbed = data[1:51].strip()
        self.position = int(data[51:])
        logging.debug("testbed: " + testbed)
        logging.debug("position: " + str(self.position))
        self.data_path = os.path.join(__location__, "measurements", testbed)
        self.serializer = DillSerializer(self.data_path)
        if os.path.isfile(self.data_path):
            self.latency = self.serializer.deserialize()
        else:
            self.latency = dict()
        self.latency[self.position] = list()
        self.test_rounds = self.factory.test_rounds
        self.send_request()
    
    def handle_relay_test(self, _):
        duration = time.time() - self.start_time
        self.latency[self.position].append(duration)
        self.send_request()
    
    def send_request(self):
        #logging.debug("test round: " + str(self.test_rounds))
        if self.test_rounds == 0:
            logging.debug("duration: " + str(self.latency[self.position]))
            logging.debug("zero duration: " + str(self.latency[self.position].count(0)))
            logging.debug("negative duration: " + str(sum(num < 0 for num in self.latency[self.position])))
            self.serializer.serialize(self.latency)
            logging.debug("close connection")
            self.sendLine(b''.join([PacketType.Relay_End]))
            self.transport.loseConnection()
        else:
            self.start_time = time.time()
            self.test_rounds -= 1
            self.sendLine(b''.join([PacketType.Relay_Test]))
            
class ServerRelayTestFactory(protocol.Factory):
    
    def __init__(self, test_rounds):
        self.test_rounds = test_rounds
    
    def buildProtocol(self, addr):
        return ServerRelayTest(self)
    
    def clientConnectionLost(self, transport, reason):
        logging.info("server connection lost")
        logging.debug(reason.getErrorMessage())
        reactor.stop()
    
    def clientConnectionFailed(self, transport, reason):
        logging.info("server connection failed")
        logging.debug(reason.getErrorMessage())
        reactor.stop()
    
def start_tcp(port, test_rounds):
    log.startLogging(sys.stdout)
    factory = ServerRelayTestFactory(test_rounds)
    reactor.listenTCP(port, factory)
    reactor.run()
    
def start_tls(port, test_rounds):
    log.startLogging(sys.stdout)
    factory = ServerRelayTestFactory(test_rounds)
    certData = utils.ssl.get_server_cert()
    certificate = ssl.PrivateCertificate.loadPEM(certData)
    logging.info(certificate)
    reactor.listenSSL(port, factory, certificate.options())
    reactor.run()
