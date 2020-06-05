import pickle
import logging
import utils.ssl
import threading
import utils.times as times
from twisted.protocols import basic
from collections import defaultdict
from coupling.utils.misc import StopWatch
from utils.nested_dict import nested_dict
from coupling.utils.misc import AtomicCounter
from coupling.utils.coupling_data import Client
from twisted.internet import reactor, protocol, ssl
from coupling.utils.coupling_data import LightPatternType
from coupling.device_grouping.online.coupling_protocol import PacketType
from coupling.device_grouping.online.machine_learning_features import BasicFeatures
from coupling.device_grouping.online.machine_learning_features import TsFreshFeatures

class ServerCouplingProtocol(basic.LineOnlyReceiver):
    
    def __init__(self, factory, rooms, evaluation_coupling, evaluation_runtime):
        self.rooms = rooms
        self.mac = None
        self.client = Client()
        self.factory = factory
        self.evaluation_coupling = evaluation_coupling
        self.evaluation_runtime = evaluation_runtime
        self.stop_watch = StopWatch()
        self.stop_watch_query_data = StopWatch()
        self.stop_watch_query_raw_light = StopWatch()
        self.stop_watch_query_pattern_light = StopWatch()
        self.stop_watch_query_raw_wifi = StopWatch()
        self.stop_watch_query_raw_ble = StopWatch()
        self.stop_watch_coupling = StopWatch()
        self.handle_methods = {
            PacketType.Response_Raw_Light: self.handle_raw_light_response,
            PacketType.Response_Light_Pattern: self.handle_light_pattern_response,
            PacketType.Response_Raw_WiFi: self.handle_raw_wifi_response,
            PacketType.Response_Raw_BLE: self.handle_raw_ble_response
        }
    
    def connectionMade(self):
        self.mac = self.get_mac()
        logging.info("connection made: " + str(self.mac))
        self.query_client_data()
    
    def query_client_data(self):
        logging.info("### query client data: " + str(self.mac))
        self.stop_watch_query_data.start()
        self.stop_watch_query_raw_light.start()
        self.sendLine(self.create_query_packet(PacketType.Query_Raw_Light, self.factory.data_period_coupling))
        self.stop_watch_query_pattern_light.start()
        self.sendLine(self.create_query_packet(PacketType.Query_Pattern_Light, self.factory.data_period_coupling))
        self.stop_watch_query_raw_wifi.start()
        self.sendLine(self.create_query_packet(PacketType.Query_Raw_WiFi, self.factory.data_period_localization))
        self.stop_watch_query_raw_ble.start()
        self.sendLine(self.create_query_packet(PacketType.Query_Raw_BLE, self.factory.data_period_localization))
    
    def create_query_packet(self, packet_type, data_period):
        return b''.join([
            packet_type,
            str(data_period)
        ])
    
    def dataReceived(self, data):
        if self.delimiter in data:
            raw_data = data.split(self.delimiter)
            self.client.data_buffer.append(raw_data[0])
            line = ''.join(self.client.data_buffer)
            self.client.reset_data_buffer()
            self.lineReceived(self.mac, line)
        else:
            self.client.data_buffer.append(data)
    
    def lineReceived(self, mac, data):
        packet_type = data[0]
        func = self.handle_methods.get(packet_type)
        data = pickle.loads(data[1:])
        func(mac, data)
        if self.check_complete_client_data(self.client):
            logging.info("check complete data: " + mac)
            self.stop_watch_query_data.stop()
            self.timer_query_data = threading.Timer(self.factory.frequency_coupling, self.query_client_data)
            self.timer_query_data.start()
            thread = threading.Thread(target=self.perform_coupling, args=(mac, self.client,))
            self.factory.processing_clients.append(thread)
            thread.start()
            self.client = Client() # reset all data
    
    def get_basic_all_features(self, data, _):
        feature = self.factory.basic_features.compute(data)
        return feature.reshape(1, -1)
    
    def get_basic_selected_features(self, data, _):
        feature = self.factory.basic_features.compute_selected_features(data)
        return feature.reshape(1, -1)
    
    def get_tsfresh_selected_features(self, data, features_to_extract):
        return self.factory.tsfresh_features.extract_selected_features(
            data, features_to_extract, True)
    
    def perform_coupling(self, mac, client):
        logging.info("### perform coupling: " + mac)
        self.stop_watch_coupling.start()
        time = str(times.get_current_time())
        for room_id, room in self.rooms.iteritems():
            logging.info("Signal similarity: " + mac + ", room: " + str(room_id))
            coupling_signal_similarity = self.coupling_via_signal_similarity(
                room.light_signal, client.light_signal, self.factory.equalize_method)
            
            logging.info("Signal pattern duration: " + mac + ", room: " + str(room_id))
            pattern_len = room.light_pattern_duration.shape[1]
            coupling_signal_pattern_duration = self.coupling_via_signal_patterns(
                room.light_pattern_duration, client.light_pattern_duration,
                pattern_len, LightPatternType.Duration, self.factory.equalize_method)
            
            logging.info("Signal pattern: " + mac + ", room: " + str(room_id))
            coupling_signal_pattern = self.coupling_via_signal_patterns(
                room.light_pattern, client.light_pattern,
                pattern_len, LightPatternType.Signal, self.factory.equalize_method)
            
            logging.info("machine learning - basic all features: " + mac + ", room: " + str(room_id))
            coupling_ml_basic_all = self.coupling_via_machine_learning(
                room.coupling_classifier_basic_all_features,
                client.light_signal, self.get_basic_all_features)
            
            logging.info("machine learning - basic selected features: " + mac + ", room: " + str(room_id))
            coupling_ml_basic_selected = self.coupling_via_machine_learning(
                room.coupling_classifier_basic_selected_features,
                client.light_signal, self.get_basic_selected_features)
            
            logging.info("machine learning - tsfresh selected features: " + mac + ", room: " + str(room_id))
            coupling_ml_tsfresh_selected = self.coupling_via_machine_learning(
                room.coupling_classifier_tsfresh_selected_features,
                client.light_signal, self.get_tsfresh_selected_features,
                room.tsfresh_selected_features)
            
            logging.info("BLE localization: " + mac + ", room: " + str(room_id))
            loc_random_ble, loc_filtering_ble, loc_svm_ble, loc_random_forest_ble = self.localization(
                room.ble_localization, room.ble_scan, client.ble_scan)
            
            logging.info("WiFi localization: " + mac + ", room: " + str(room_id))
            loc_random_wifi, loc_filtering_wifi, loc_svm_wifi, loc_random_forest_wifi = self.localization(
                room.wifi_localization, room.wifi_scan, client.wifi_scan)
            
            self.evaluation_coupling[mac]["signal similarity"][time].append((room_id, coupling_signal_similarity[0], coupling_signal_similarity[1]))
            self.evaluation_coupling[mac]["signal pattern"][time].append((room_id, coupling_signal_pattern[0], coupling_signal_pattern[1]))
            self.evaluation_coupling[mac]["signal pattern duration"][time].append((room_id, coupling_signal_pattern_duration[0], coupling_signal_pattern_duration[1]))
            self.evaluation_coupling[mac]["ml basic all features"][time].append((room_id, coupling_ml_basic_all[0], coupling_ml_basic_all[1]))
            self.evaluation_coupling[mac]["ml basic selected features"][time].append((room_id, coupling_ml_basic_selected[0], coupling_ml_basic_selected[1]))
            self.evaluation_coupling[mac]["ml tsfresh selected features"][time].append((room_id, coupling_ml_tsfresh_selected[0], coupling_ml_tsfresh_selected[1]))
            self.evaluation_coupling[mac]["loc random BLE"][time].append((room_id, loc_random_ble[0], loc_random_ble[1]))
            self.evaluation_coupling[mac]["loc filtering BLE"][time].append((room_id, loc_filtering_ble[0], loc_filtering_ble[1]))
            self.evaluation_coupling[mac]["loc SVM BLE"][time].append((room_id, loc_svm_ble[0], loc_svm_ble[1]))
            self.evaluation_coupling[mac]["loc Random Forest BLE"][time].append((room_id, loc_random_forest_ble[0], loc_random_forest_ble[1]))
            self.evaluation_coupling[mac]["loc random WiFi"][time].append((room_id, loc_random_wifi[0], loc_random_wifi[1]))
            self.evaluation_coupling[mac]["loc filtering WiFi"][time].append((room_id, loc_filtering_wifi[0], loc_filtering_wifi[1]))
            self.evaluation_coupling[mac]["loc SVM WiFi"][time].append((room_id, loc_svm_wifi[0], loc_svm_wifi[1]))
            self.evaluation_coupling[mac]["loc Random Forest WiFi"][time].append((room_id, loc_random_forest_wifi[0], loc_random_forest_wifi[1]))
        
        self.evaluation_runtime["time coupling"].append(self.stop_watch_coupling.get_elapsed_time())
        self.evaluation_runtime["time query data"].append(self.stop_watch_query_data.get_elapsed_time())
        self.evaluation_runtime["time query raw light"].append(self.stop_watch_query_raw_light.get_elapsed_time())
        self.evaluation_runtime["time query pattern light"].append(self.stop_watch_query_pattern_light.get_elapsed_time())
        self.evaluation_runtime["time query raw wifi"].append(self.stop_watch_query_raw_wifi.get_elapsed_time())
        self.evaluation_runtime["time query raw ble"].append(self.stop_watch_query_raw_ble.get_elapsed_time())
        
        logging.info("### end perform coupling: " + mac)
    
    def __generic_localization(self, room_localizer, localization_method, room_scan, client_scan):
        feature = room_localizer.features.compute(
            room_scan, client_scan, room_localizer.features.imputing_values)
        return bool(localization_method(feature))
    
    def localization(self, room_localizer, room_scan, client_scan):
        self.stop_watch.start()
        random = self.__generic_localization(
            room_localizer, room_localizer.reasoning_random.predict, room_scan, client_scan)
        random_time = self.stop_watch.get_elapsed_time()
        self.stop_watch.start()
        filtering = self.__generic_localization(
            room_localizer, room_localizer.reasoning_filtering.predict, room_scan, client_scan)
        filtering_time = self.stop_watch.get_elapsed_time()
        self.stop_watch.start()
        svm = self.__generic_localization(
            room_localizer, room_localizer.reasoning_machine_learning.predict_svm, room_scan, client_scan)
        svm_time = self.stop_watch.get_elapsed_time()
        self.stop_watch.start()
        random_forest = self.__generic_localization(
            room_localizer, room_localizer.reasoning_machine_learning.predict_random_forest, room_scan, client_scan)
        random_forest_time = self.stop_watch.get_elapsed_time()
        return (random, random_time), (filtering, filtering_time), (svm, svm_time), (random_forest, random_forest_time)
    
    def coupling_via_signal_similarity(self, room_light_data, client_light_data, equalize_method):
        self.stop_watch.start()
        similarity = self.factory.coupling_compare_method(
            room_light_data, client_light_data, equalize_method)
        result = (similarity >= self.factory.coupling_similarity_threshold)
        return result, self.stop_watch.get_elapsed_time()
    
    def coupling_via_signal_patterns(self, room_light_pattern, client_light_pattern,
                                     pattern_len, coupling_type, equalize_method):
        self.stop_watch.start()
        counter = 0
        match_result = False
        found_matching_pattern = 0
        max_counter = pattern_len * 10
        for search_signal in client_light_pattern:
            for collection_signal in room_light_pattern:
                if coupling_type == LightPatternType.Signal:
                    min_data_series = min([len(search_signal), len(collection_signal)])
                    search_signal = search_signal[:min_data_series]
                    collection_signal = collection_signal[:min_data_series]        
                similarity = self.factory.coupling_compare_method(
                    search_signal, collection_signal, equalize_method)
                counter += 1
                if similarity > self.factory.coupling_similarity_threshold:
                    found_matching_pattern += 1
                    break
            if found_matching_pattern == pattern_len:
                match_result = True
                break
            elif counter > max_counter:
                match_result = False
                break    
        return match_result, self.stop_watch.get_elapsed_time()
    
    def coupling_via_machine_learning(self, room_classifier, client_light_data,
                                      callback_create_feature, features_to_extract=None):
        self.stop_watch.start()
        feature = callback_create_feature(client_light_data, features_to_extract)
        result = room_classifier.predict(feature)
        return bool(result), self.stop_watch.get_elapsed_time()
    
    def handle_raw_light_response(self, mac, data):
        logging.debug("raw light response: " + mac)
        self.client.light_signal = data
        self.stop_watch_query_raw_light.stop()
    
    def handle_light_pattern_response(self, mac, data):
        logging.debug("light pattern response: " + mac)
        packet_type0 = data[0][0]
        packet_type1 = data[1][0]
        data0 = pickle.loads(data[0][1:])
        data1 = pickle.loads(data[1][1:])
        if packet_type0 == PacketType.Response_Light_Pattern_Duration and \
            packet_type1 == PacketType.Response_Light_Pattern_Signal:
            self.client.light_pattern_duration = data0
            self.client.light_pattern = data1
        else:
            self.client.light_pattern_duration = data1
            self.client.light_pattern = data0
        self.stop_watch_query_pattern_light.stop()
    
    def handle_raw_wifi_response(self, mac, data):
        logging.debug("raw WiFi response: " + mac)
        self.client.wifi_scan = data
        self.stop_watch_query_raw_wifi.stop()
    
    def handle_raw_ble_response(self, mac, data):
        logging.debug("raw BLE response: " + mac)
        self.client.ble_scan = data
        self.stop_watch_query_raw_ble.stop()
    
    def connectionLost(self, _):
        self.timer_query_data.cancel()
        logging.info("connection lost: " + self.mac)
        if self.factory.connected_clients.decrement() == 0:
            logging.info("all connections lost, wait for completion")
            for client in self.factory.processing_clients:
                client.join()
            logging.info("processing completed")
            self.factory.evaluate_callback(
                self.evaluation_coupling, self.evaluation_runtime)
            self.factory.stop_reactor_callback()
        else:
            logging.info("connected clients: " + str(self.factory.connected_clients.get()))
    
    def check_complete_client_data(self, client):
        data = list()
        data.append(client.light_signal is None)
        data.append(client.light_pattern_duration is None)
        data.append(client.light_pattern is None)
        data.append(client.wifi_scan is None)
        data.append(client.ble_scan is None)    
        return not (data.count(True) > 0)
    
    def get_mac(self):
        ip, port = self.transport.client
        if "127.0.0.1" in ip:
            return self.factory.access_point.get_mac(port)
        else:
            return self.factory.access_point.get_mac(ip)
     
class ServerCouplingFactory(protocol.Factory):
    
    def __init__(self, access_point,
                 data_period_coupling, coupling_compare_method,
                 coupling_similarity_threshold, equalize_method,
                 data_period_localization,
                 num_clients, rooms, frequency_coupling,
                 stop_reactor_callback, evaluate_callback):
        self.processing_clients = list()
        self.connected_clients = AtomicCounter(num_clients)
        self.evaluation_coupling = nested_dict(3, list)
        self.evaluation_runtime = defaultdict(list)
        self.rooms = rooms
        self.frequency_coupling = frequency_coupling
        self.access_point = access_point
        self.data_period_coupling = data_period_coupling
        self.coupling_compare_method = coupling_compare_method
        self.coupling_similarity_threshold = coupling_similarity_threshold
        self.equalize_method = equalize_method
        self.basic_features = BasicFeatures()
        self.tsfresh_features = TsFreshFeatures()
        self.data_period_localization = data_period_localization
        self.stop_reactor_callback = stop_reactor_callback
        self.evaluate_callback = evaluate_callback
    
    def buildProtocol(self, addr):
        logging.debug("Server address: " + str(addr))
        return ServerCouplingProtocol(
            self, self.rooms, self.evaluation_coupling, self.evaluation_runtime)
    
class ServerController:
    
    def __init__(self, server_port, access_point,
                 data_period_coupling, coupling_compare_method,
                 coupling_similarity_threshold, equalize_method,
                 data_period_localization,
                 num_clients, rooms, frequency_coupling,
                 stop_reactor_callback, evaluate_callback):
        
        certData = utils.ssl.get_server_cert()
        self.port = server_port
        self.certificate = ssl.PrivateCertificate.loadPEM(certData)
        self.server_coupling_factory = ServerCouplingFactory(access_point,
                                                             data_period_coupling, coupling_compare_method,
                                                             coupling_similarity_threshold, equalize_method,
                                                             data_period_localization,
                                                             num_clients, rooms, frequency_coupling,
                                                             stop_reactor_callback, evaluate_callback)
    
    def start(self):
        self.port = reactor.listenSSL(
            self.port, self.server_coupling_factory, self.certificate.options())
    
    def stop(self):
        #self.port.stopListening()
        self.port.loseConnection() # stop accepting connection
        self.port.connectionLost(reason=None) # cleanup socket
