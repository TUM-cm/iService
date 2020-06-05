import pickle
import logging
import utils.ssl
import coupling.utils.misc as misc
from twisted.protocols import basic
from twisted.internet import reactor, protocol, ssl
import coupling.light_grouping_pattern.light_analysis as light_analysis
from coupling.device_grouping.online.coupling_protocol import PacketType

class ClientCouplingProtocol(basic.LineOnlyReceiver):
    
    def __init__(self, factory):
        self.factory = factory
        self.handle_methods = {
            PacketType.Query_Raw_Light: self.handle_query_raw_light,
            PacketType.Query_Pattern_Light: self.handle_query_pattern_light,
            PacketType.Query_Raw_WiFi: self.handle_query_raw_wifi,
            PacketType.Query_Raw_BLE: self.handle_query_raw_ble
        }
    
    def lineReceived(self, data):
        paket_type = data[0:1]
        data_period = float(data[1:])
        func = self.handle_methods.get(paket_type)
        func(data_period)
    
    def handle_query_raw_light(self, data_period):
        logging.debug("query raw light: " + self.factory.get_mac())
        voltage, _ = self.factory.data_provider.get_light_data(data_period)
        packet = self.create_pickle_packet(PacketType.Response_Raw_Light, voltage)
        logging.debug("response raw light: " + self.factory.get_mac())
        self.sendLine(packet)
    
    def handle_query_pattern_light(self, data_period, rounds=10):
        logging.debug("query light pattern: " + self.factory.get_mac())
        for _ in range(rounds):
            try:
                light_signal, light_signal_time = self.factory.data_provider.get_light_data(data_period)
                light_pattern_duration, light_pattern = light_analysis.detect_cycle_by_sequence(
                                                            light_signal, light_signal_time)
                if misc.valid_light_pattern(light_pattern_duration):
                    break
            except:
                pass
        # Create packet for each data type and merge into one packet
        packet_duration = self.create_pickle_packet(PacketType.Response_Light_Pattern_Duration, light_pattern_duration)
        packet_voltage = self.create_pickle_packet(PacketType.Response_Light_Pattern_Signal, light_pattern)
        packet = self.create_pickle_packet(PacketType.Response_Light_Pattern, [packet_duration, packet_voltage])        
        logging.debug("response light pattern: " + self.factory.get_mac())
        self.sendLine(packet)
    
    def handle_query_raw_wifi(self, data_period):
        logging.debug("query raw WiFi: " + self.factory.get_mac())
        wifi = self.factory.data_provider.get_wifi_data(data_period)
        packet = self.create_pickle_packet(PacketType.Response_Raw_WiFi, wifi)
        logging.debug("response raw WiFi: " + self.factory.get_mac())
        self.sendLine(packet)
    
    def handle_query_raw_ble(self, data_period):
        logging.debug("query raw BLE: " + self.factory.get_mac())
        ble = self.factory.data_provider.get_ble_data(data_period)
        packet = self.create_pickle_packet(PacketType.Response_Raw_BLE, ble)
        logging.debug("response raw BLE: " + self.factory.get_mac())
        self.sendLine(packet)
    
    def create_pickle_packet(self, message_type, data):
        return b''.join([
                message_type,
                pickle.dumps(data)
            ])
        
class ClientCouplingFactory(protocol.ClientFactory):
    
    def __init__(self, data_provider):
        self.transport = None
        self.data_provider = data_provider
    
    def buildProtocol(self, addr):
        logging.info("addr: " + str(addr))
        return ClientCouplingProtocol(self)
    
    def startedConnecting(self, connector):
        self.transport = connector.transport
    
    def set_mac(self, mac):
        self.mac = mac
    
    def get_mac(self):
        return self.mac
    
class ClientController:
    
    def __init__(self, server_ip, server_port,
                 data_provider, coupling_groundtruth):
        self.server_ip = server_ip
        self.server_port = server_port
        self.coupling_groundtruth = coupling_groundtruth
        self.factory = ClientCouplingFactory(data_provider)
        certData = utils.ssl.get_public_cert()
        authority = ssl.Certificate.loadPEM(certData)
        self.options = ssl.optionsForClientTLS(self.server_ip, authority)
    
    def start(self):
        self.connect = reactor.connectSSL(self.server_ip,
                                          self.server_port,
                                          self.factory,
                                          self.options)
    
    def stop(self):
        self.connect.disconnect()
