import numpy
import pickle
import logging
import utils.ssl
import itertools
import coupling.utils.misc as misc
from twisted.protocols import basic
from coupling.tvgl.TVGL import TVGL
from coupling.utils.misc import StopWatch
from coupling.utils.coupling_data import Client
from twisted.internet import reactor, protocol, ssl
from coupling.utils.coupling_data import LightPatternType
from coupling.utils.coupling_data import StaticEvaluationResult
from coupling.localization.localization_server import Localization
from coupling.localization.localization_server import LocalizationFeatures
from coupling.device_grouping.online.coupling_protocol import PacketType
from coupling.device_grouping.online.machine_learning_features import BasicFeatures
from coupling.device_grouping.online.machine_learning_features import TsFreshFeatures
import coupling.device_grouping.online.coupling_static_simulator as coupling_simulator

class ServerCouplingProtocol(basic.LineOnlyReceiver):
    
    def __init__(self, factory, clients):
        self.factory = factory
        self.clients = clients
        self.handle_methods = {
            PacketType.Response_Raw_Light: self.handle_raw_light_response,
            PacketType.Response_Light_Pattern: self.handle_light_pattern_response,
            PacketType.Response_Raw_WiFi: self.handle_raw_wifi_response,
            PacketType.Response_Raw_BLE: self.handle_raw_ble_response
        }
    
    def connectionMade(self):
        mac = self.get_mac()
        logging.info("connection made: " + str(mac))
        if mac not in self.clients:
            self.clients[mac] = Client()
        self.query_client_data()
    
    def query_client_data(self):
        logging.info("query client data: " + str(self.get_mac()))
        self.factory.stop_watch_query_data.start()
        self.factory.stop_watch_query_raw_light.start()
        self.sendLine(self.create_query_packet(PacketType.Query_Raw_Light, self.factory.data_period_coupling))
        self.factory.stop_watch_query_pattern_light.start()
        self.sendLine(self.create_query_packet(PacketType.Query_Pattern_Light, self.factory.data_period_coupling))
        self.factory.stop_watch_query_raw_wifi.start()
        self.sendLine(self.create_query_packet(PacketType.Query_Raw_WiFi, self.factory.data_period_localization))
        self.factory.stop_watch_query_raw_ble.start()
        self.sendLine(self.create_query_packet(PacketType.Query_Raw_BLE, self.factory.data_period_localization))
    
    def create_query_packet(self, packet_type, data_period):
        return b''.join([
            packet_type,
            str(data_period)
        ])
    
    def get_basic_all_features(self, data):
        feature = self.factory.basic_features.compute(data)
        return feature.reshape(1, -1)
    
    def get_basic_selected_features(self, data):
        feature = self.factory.basic_features.compute_selected_features(data)
        return feature.reshape(1, -1)
    
    def get_tsfresh_all_features(self, data):
        return self.factory.tsfresh_features.extract_selected_features(
            data, self.factory.tsfresh_features_to_extract_all, True)
        
    def get_tsfresh_selected_features(self, data):
        return self.factory.tsfresh_features.extract_selected_features(
            data, self.factory.tsfresh_features_to_extract_selected, True)
    
    def lineReceived(self, mac, data):
        self.clients[mac].reset_data_buffer()
        packet_type = data[0]
        func = self.handle_methods.get(packet_type)
        data = pickle.loads(data[1:])
        func(mac, data)
        if self.check_complete_client_data():
            self.factory.stop_watch_query_data.stop()
            for i in range(coupling_simulator.evaluation_rounds):
                logging.info("round: " + str(i))
                self.factory.stop_watch_coupling.start()
                
                logging.info("BLE localization")
                client_data = [(client_mac, self.clients[client_mac].ble_scan) for client_mac in self.clients]
                loc_random_ble, loc_filtering_ble, loc_svm_ble, loc_random_forest_ble = self.localization(
                    self.factory.ble_localization, client_data)
                logging.info("---------------------------")
                 
                logging.info("WiFi localization")
                client_data = [(client_mac, self.clients[client_mac].wifi_scan) for client_mac in self.clients]
                loc_random_wifi, loc_filtering_wifi, loc_svm_wifi, loc_random_forest_wifi = self.localization(
                    self.factory.wifi_localization, client_data)
                logging.info("---------------------------")
                 
                logging.info("light pattern duration")
                coupling_signal_pattern_duration = self.coupling_via_signal_patterns(
                    LightPatternType.Duration, self.factory.equalize_method)
                logging.info("---------------------------")
                 
                logging.info("light pattern signal")
                coupling_signal_pattern_signal = self.coupling_via_signal_patterns(
                    LightPatternType.Signal, self.factory.equalize_method)
                logging.info("------------------------")
                 
                logging.info("light similarity")
                coupling_signal_similarity = self.coupling_via_signal_similarity(self.factory.equalize_method)
                logging.info("------------------------")
                 
                logging.info("machine learning - basic all features")
                coupling_ml_basic_all = self.coupling_via_machine_learning(
                    self.factory.coupling_classifier_basic_all_features, self.get_basic_all_features)
                logging.info("---------------------------")
                  
                logging.info("machine learning - basic selected features")
                coupling_ml_basic_selected = self.coupling_via_machine_learning(
                    self.factory.coupling_classifier_basic_selected_features, self.get_basic_selected_features)
                logging.info("---------------------------")
                
                #logging.info("machine learning - tsfresh all features")
                #coupling_ml_tsfresh_all = self.coupling_via_machine_learning(
                #    self.factory.coupling_classifier_tsfresh_all_features, self.get_tsfresh_all_features)
                #logging.info("---------------------------")
                
                logging.info("machine learning - tsfresh selected features")
                coupling_ml_tsfresh_selected = self.coupling_via_machine_learning(
                    self.factory.coupling_classifier_tsfresh_selected_features, self.get_tsfresh_selected_features)
                logging.info("---------------------------")
                self.factory.stop_watch_coupling.stop()
                
                #logging.info("TVGL")
                #coupling_tvgl = self.coupling_via_tvgl()
                #logging.info("---------------------------")
                coupling_tvgl = None
                coupling_ml_tsfresh_all = None
                
                evaluation_result = StaticEvaluationResult(self.factory.stop_watch_query_data.get_elapsed_time(), self.factory.stop_watch_query_raw_light.get_elapsed_time(),
                                                           self.factory.stop_watch_query_pattern_light.get_elapsed_time(), self.factory.stop_watch_query_raw_wifi.get_elapsed_time(),
                                                           self.factory.stop_watch_query_raw_ble.get_elapsed_time(), self.factory.stop_watch_coupling.get_elapsed_time(),
                                                           loc_random_wifi, loc_filtering_wifi, loc_svm_wifi, loc_random_forest_wifi,
                                                           loc_random_ble, loc_filtering_ble, loc_svm_ble, loc_random_forest_ble,
                                                           coupling_signal_pattern_duration, coupling_signal_pattern_signal,  coupling_signal_similarity,
                                                           coupling_ml_basic_all, coupling_ml_basic_selected,
                                                           coupling_ml_tsfresh_all, coupling_ml_tsfresh_selected,
                                                           coupling_tvgl)
                coupling_simulator.add_evaluation_data(evaluation_result)
                logging.info("----")
            self.stop_coupling_simulation()
    
    def dataReceived(self, data):
        mac = self.get_mac()
        if self.delimiter in data:
            data = data.split(self.delimiter)[0]
            self.clients[mac].data_buffer.append(data)
            self.lineReceived(mac, ''.join(self.clients[mac].data_buffer))
        else:
            self.clients[mac].data_buffer.append(data)
    
    def handle_raw_light_response(self, mac, data):
        logging.debug("raw light response: " + mac)
        self.clients[mac].light_signal = data
        self.factory.stop_watch_query_raw_light.stop()
    
    def handle_light_pattern_response(self, mac, data):
        logging.debug("light pattern response: " + mac)
        packet_type0 = data[0][0]
        packet_type1 = data[1][0]
        data0 = pickle.loads(data[0][1:])
        data1 = pickle.loads(data[1][1:])
        if packet_type0 == PacketType.Response_Light_Pattern_Duration and \
            packet_type1 == PacketType.Response_Light_Pattern_Signal:
            self.clients[mac].light_pattern_duration = data0
            self.clients[mac].light_pattern = data1
        else:
            self.clients[mac].light_pattern_duration = data1
            self.clients[mac].light_pattern = data0
        self.factory.stop_watch_query_pattern_light.stop()
    
    def handle_raw_wifi_response(self, mac, data):
        logging.debug("raw wifi response: " + mac)
        self.clients[mac].wifi_scan = data
        self.factory.stop_watch_query_raw_wifi.stop()
    
    def handle_raw_ble_response(self, mac, data):
        logging.debug("raw ble response: " + mac)
        self.clients[mac].ble_scan = data
        self.factory.stop_watch_query_raw_ble.stop()
    
    def check_complete_client_data(self):
        light_signal = []
        light_pattern_duration = []
        light_pattern = []
        wifi_scan = []
        ble_scan = []
        for client in self.clients.values():
            light_signal.append(client.light_signal is None)
            light_pattern_duration.append(client.light_pattern_duration is None)
            light_pattern.append(client.light_pattern is None)
            wifi_scan.append(client.wifi_scan is None)
            ble_scan.append(client.ble_scan is None)
        if light_signal.count(False) == self.factory.access_point.get_num_connected_clients() \
            and light_pattern_duration.count(False) == self.factory.access_point.get_num_connected_clients() \
            and light_pattern.count(False) == self.factory.access_point.get_num_connected_clients() \
            and wifi_scan.count(False) == self.factory.access_point.get_num_connected_clients() \
            and ble_scan.count(False) == self.factory.access_point.get_num_connected_clients():
            logging.debug("all client data")
            return True
        else:
            logging.debug("missing client data")
            return False
    
    def __generic_localization(self, localizer, localization_method, client_data):
        accept_clients = set()
        reject_clients = set()
        for client_mac, client_scan in client_data:
            results = dict()
            for position, db_scan in localizer.features.scans.iteritems():
                feature = localizer.features.compute(
                    client_scan, db_scan, localizer.features.imputing_values)
                results[position] = localization_method(feature)
            results = localizer.in_area(results, self.factory.localization_pos_in_area,
                                        self.factory.localization_pos_out_area)
            accept_clients.add(client_mac) if results else reject_clients.add(client_mac)
        return accept_clients, reject_clients
    
    def localization(self, localizer, client_data):
        logging.info("### Random")
        self.factory.stop_watch.start()
        random = self.__generic_localization(localizer, localizer.reasoning_random.predict, client_data)
        random = self.factory.evaluate_callback(random[0], random[1], self.factory.stop_watch.get_elapsed_time())
        logging.info("### Filtering")
        self.factory.stop_watch.start()
        filtering = self.__generic_localization(localizer, localizer.reasoning_filtering.predict, client_data)
        filtering = self.factory.evaluate_callback(filtering[0], filtering[1], self.factory.stop_watch.get_elapsed_time())    
        logging.info("### SVM")
        self.factory.stop_watch.start()
        svm = self.__generic_localization(localizer, localizer.reasoning_machine_learning.predict_svm, client_data)
        svm = self.factory.evaluate_callback(svm[0], svm[1], self.factory.stop_watch.get_elapsed_time())
        logging.info("### Random Forest")
        self.factory.stop_watch.start()
        random_forest = self.__generic_localization(
            localizer, localizer.reasoning_machine_learning.predict_random_forest, client_data)
        random_forest = self.factory.evaluate_callback(
            random_forest[0], random_forest[1], self.factory.stop_watch.get_elapsed_time())
        return random, filtering, svm, random_forest
    
    def coupling_via_signal_similarity(self, equalize_method):
        self.factory.stop_watch.start()
        accept_clients = set()
        reject_clients = set()
        import time
        for client_mac1, client_mac2 in itertools.combinations(self.clients.keys(), 2):
            print(self.factory.coupling_compare_method)
            print(len(self.clients[client_mac1].light_signal))
            print(len(self.clients[client_mac2].light_signal))
            print(equalize_method)
            start = time.time()
            similarity = self.factory.coupling_compare_method(
                                    self.clients[client_mac1].light_signal,
                                    self.clients[client_mac2].light_signal,
                                    equalize_method)
            print(time.time() - start)
            print("---")
            if similarity < self.factory.coupling_similarity_threshold:
                reject_clients.update([client_mac1, client_mac2])
            else:
                accept_clients.update([client_mac1, client_mac2])
        reject_clients -= accept_clients
        return self.factory.evaluate_callback(accept_clients, reject_clients,
                                              self.factory.stop_watch.get_elapsed_time())
    
    def coupling_via_signal_patterns(self, coupling_type, equalize_method):
        self.factory.stop_watch.start()
        accept_clients = set()
        reject_clients = set()
        for client_mac1, client_mac2 in itertools.combinations(self.clients.keys(), 2):
            if coupling_type == LightPatternType.Duration:
                client1_signal = self.clients[client_mac1].light_pattern_duration
                client2_signal = self.clients[client_mac2].light_pattern_duration
            elif coupling_type == LightPatternType.Signal:
                client1_signal = self.clients[client_mac1].light_pattern
                client2_signal = self.clients[client_mac2].light_pattern
            run_search = True
            match_result = False
            if coupling_type == LightPatternType.Duration:
                if len(client1_signal[0]) != len(client2_signal[0]):
                    run_search = False
            if run_search:
                match_result = self.__find_matching_patterns(client_mac1, client1_signal, client2_signal,
                                                             coupling_type, equalize_method)
            if match_result:
                accept_clients.update([client_mac1, client_mac2])
            else:
                reject_clients.update([client_mac1, client_mac2])
        reject_clients -= accept_clients    
        return self.factory.evaluate_callback(accept_clients, reject_clients,
                                              self.factory.stop_watch.get_elapsed_time())
    
    def __find_matching_patterns(self, client_mac1, client1_signal, client2_signal,
                                 coupling_type, equalize_method):
        counter = 0
        match_result = False
        found_matching_pattern = 0
        pattern_len = self.clients[client_mac1].light_pattern_duration.shape[1]
        max_counter = pattern_len * 10
        for search_signal in client1_signal:
            for collection_signal in client2_signal:
                if coupling_type == LightPatternType.Signal:
                    min_data_series = min([len(search_signal), len(collection_signal)])
                    search_signal = search_signal[:min_data_series]
                    collection_signal = collection_signal[:min_data_series]        
                similarity = self.factory.coupling_compare_method(search_signal,
                                                                  collection_signal,
                                                                  equalize_method)
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
        return match_result
    
    def coupling_via_machine_learning(self, classifier, callback_create_feature):
        self.factory.stop_watch.start()
        accept_clients = set()
        reject_clients = set()
        for client_mac in self.clients.keys():
            client_light_data = self.clients[client_mac].light_signal
            logging.debug("length light data: " + str(len(client_light_data)))
            feature = callback_create_feature(client_light_data)
            logging.debug("feature shape: " + str(feature.shape))
            result = classifier.predict(feature)
            if result == 1.0:
                accept_clients.add(client_mac)
            else:
                reject_clients.add(client_mac)
        return self.factory.evaluate_callback(accept_clients, reject_clients,
                                              self.factory.stop_watch.get_elapsed_time())
    
    def coupling_via_tvgl(self):
        self.factory.stop_watch.start()
        clients = []
        client_light_data = []
        for client_mac in self.clients.keys():
            clients.append(client_mac)
            light_signal = self.clients[client_mac].light_signal
            client_light_data.append(light_signal.reshape(-1,1))
        clients = numpy.array(clients)
        client_light_data = numpy.concatenate(client_light_data, axis=1)
        length_of_slice = client_light_data.shape[1]
        thetaSet = TVGL(client_light_data, length_of_slice, lamb=2.5, beta=12, indexOfPenalty=-1)
        thetaSet = [theta for theta in thetaSet if misc.matrix_is_diag(theta)]
        thetaSum = numpy.sum(thetaSet, axis=0)
        thetaValues = thetaSum[thetaSum != 0]
        divider = numpy.mean(thetaValues)
        accept_clients = set(clients[thetaValues < divider])
        reject_clients = set(clients[thetaValues > divider])
        return self.factory.evaluate_callback(
            accept_clients, reject_clients, self.factory.stop_watch.get_elapsed_time())
    
    def get_mac(self):
        ip, port = self.transport.client
        if "127.0.0.1" in ip:
            return self.factory.access_point.get_mac(port)
        else:
            return self.factory.access_point.get_mac(ip)
    
    def stop_coupling_simulation(self):
        self.factory.stop_reactor_callback()
    
class ServerCouplingFactory(protocol.Factory):
    
    def __init__(self, access_point,
                 data_period_coupling, coupling_compare_method, coupling_similarity_threshold, equalize_method,
                 data_period_localization, localization_pos_in_area, localization_pos_out_area, path_wifi_scans, path_ble_scans,
                 coupling_classifier_basic_all_features, coupling_classifier_basic_selected_features,
                 coupling_classifier_tsfresh_all_features, coupling_classifier_tsfresh_selected_features,
                 tsfresh_features_to_extract_all, tsfresh_features_to_extract_selected,
                 evaluate_callback, stop_reactor_callback):
        
        self.clients = dict()
        self.stop_watch = StopWatch()
        self.stop_watch_query_data = StopWatch()
        self.stop_watch_query_raw_light = StopWatch()
        self.stop_watch_query_pattern_light = StopWatch()
        self.stop_watch_query_raw_wifi = StopWatch()
        self.stop_watch_query_raw_ble = StopWatch()
        self.stop_watch_coupling = StopWatch()
        self.access_point = access_point
        
        self.data_period_coupling = data_period_coupling
        self.coupling_compare_method = coupling_compare_method
        self.coupling_similarity_threshold = coupling_similarity_threshold
        self.equalize_method = equalize_method
        
        self.basic_features = BasicFeatures()
        self.tsfresh_features = TsFreshFeatures()
        self.tsfresh_features_to_extract_all = tsfresh_features_to_extract_all
        self.tsfresh_features_to_extract_selected = tsfresh_features_to_extract_selected
        self.coupling_classifier_basic_all_features = coupling_classifier_basic_all_features
        self.coupling_classifier_basic_selected_features = coupling_classifier_basic_selected_features
        self.coupling_classifier_tsfresh_all_features = coupling_classifier_tsfresh_all_features
        self.coupling_classifier_tsfresh_selected_features = coupling_classifier_tsfresh_selected_features
        
        self.data_period_localization = data_period_localization
        self.localization_pos_in_area = localization_pos_in_area
        self.localization_pos_out_area = localization_pos_out_area
        wifi_features = LocalizationFeatures.from_single_room(path_wifi_scans, localization_pos_in_area)
        ble_features = LocalizationFeatures.from_single_room(path_ble_scans, localization_pos_in_area)
        self.wifi_localization = Localization(wifi_features)
        self.ble_localization = Localization(ble_features)
    
        self.evaluate_callback = evaluate_callback
        self.stop_reactor_callback = stop_reactor_callback
    
    def buildProtocol(self, addr):
        logging.debug("Server address: " + str(addr))
        return ServerCouplingProtocol(self, self.clients)
    
class ServerController:
    
    def __init__(self, server_port, access_point,
                 data_period_coupling, coupling_compare_method, coupling_similarity_threshold, equalize_method,
                 data_period_localization, localization_pos_in_area, localization_pos_out_area, path_wifi_scans, path_ble_scans,
                 coupling_classifier_basic_all_features, coupling_classifier_basic_selected_features,
                 coupling_classifier_tsfresh_all_features, coupling_classifier_tsfresh_selected_features,
                 tsfresh_features_to_extract_all, tsfresh_features_to_extract_selected,
                 evaluate_callback, stop_reactor_callback):
        
        certData = utils.ssl.get_server_cert()
        self.port = server_port
        self.certificate = ssl.PrivateCertificate.loadPEM(certData)
        self.server_coupling_factory = ServerCouplingFactory(access_point,
                                                             data_period_coupling, coupling_compare_method, coupling_similarity_threshold, equalize_method,
                                                             data_period_localization, localization_pos_in_area, localization_pos_out_area, path_wifi_scans, path_ble_scans,
                                                             coupling_classifier_basic_all_features, coupling_classifier_basic_selected_features,
                                                             coupling_classifier_tsfresh_all_features, coupling_classifier_tsfresh_selected_features,
                                                             tsfresh_features_to_extract_all, tsfresh_features_to_extract_selected,
                                                             evaluate_callback, stop_reactor_callback)
        
    def start(self):
        self.port = reactor.listenSSL(self.port,
                                      self.server_coupling_factory,
                                      self.certificate.options())
    
    def stop(self):
        #self.port.stopListening()
        self.port.loseConnection() # stop accepting connection
        self.port.connectionLost(reason=None) # cleanup socket
