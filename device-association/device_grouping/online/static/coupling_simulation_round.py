import os
import sys

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
module_path = os.path.join(__location__, "..", "..", "..")
sys.path.append(module_path)

import glob
import numpy
import logging
from mock import MagicMock
from itertools import cycle
import tsfresh.feature_extraction
from twisted.internet import reactor
from utils.serializer import DillSerializer
from coupling.utils.misc import create_random_mac
from coupling.utils.access_point import AccessPoint
from coupling.utils.coupling_data import StaticCouplingResult
import coupling.light_grouping_pattern.light_analysis as light_analysis
import coupling.localization.localization_server as localization_server
from coupling.device_grouping.online.machine_learning_features import Classifier
from coupling.device_grouping.online.static.coupling_server import ServerController
from coupling.device_grouping.online.static.coupling_client import ClientController
from coupling.device_grouping.online.machine_learning_features import BasicFeatures
from coupling.device_grouping.online.machine_learning_features import TsFreshFeatures
import coupling.device_grouping.online.coupling_static_simulator as coupling_simulator
from coupling.device_grouping.online.static.coupling_data_provider import CouplingDataProvider

# global data
clients = list()
mac_mapping = dict()

class SimulationData:
    
    def __init__(self, server_ip, server_port, num_clients, num_reject_clients, len_light_pattern,
                 data_period_coupling, coupling_compare_method, coupling_similarity_threshold, equalize_method,
                 data_period_ml_train, path_ml_train_data, coupling_ml_classifier,
                 data_period_localization, localization_pos_in_area, path_wifi_scans, path_ble_scans, rounds=0):
        
        self.server_ip = server_ip
        self.server_port = int(server_port)
        self.num_clients = int(num_clients)
        self.num_reject_clients = int(num_reject_clients)
        self.len_light_pattern = int(len_light_pattern)
        self.light_signal, self.light_signal_time = light_analysis.load_light_pattern(self.len_light_pattern)
        
        self.data_period_coupling = float(data_period_coupling)
        self.str_coupling_compare_method = coupling_compare_method
        self.coupling_compare_method = coupling_simulator.coupling_compare_methods[self.str_coupling_compare_method]
        
        self.coupling_similarity_threshold = float(coupling_similarity_threshold)
        self.str_equalize_method = equalize_method
        self.equalize_method = coupling_simulator.equalize_methods[self.str_equalize_method]
        
        self.data_period_ml_train = float(data_period_ml_train)
        combined_raw_feature_data = glob.glob(os.path.join(path_ml_train_data, "combined-*-raw-feature-data"))[0]
        combined_raw_feature_data = DillSerializer(combined_raw_feature_data).deserialize()
        X_basic = combined_raw_feature_data[self.data_period_ml_train][rounds].X_basic
        y_basic = combined_raw_feature_data[self.data_period_ml_train][rounds].y_basic
        X_basic_all_features, X_basic_selected_features = self.process_ml_features(
            X_basic, y_basic, "basic-features", self.data_period_ml_train)
        assert len(X_basic_all_features) == len(y_basic)
        assert len(X_basic_selected_features) == len(y_basic)
        
        X_tsfresh = combined_raw_feature_data[self.data_period_ml_train][rounds].X_tsfresh
        y_tsfresh = combined_raw_feature_data[self.data_period_ml_train][rounds].y_tsfresh
        tsfresh_features_to_extract_selected = os.path.join(__location__, "..", "tsfresh-features-to-be-extracted")
        self.tsfresh_features_to_extract_selected = DillSerializer(tsfresh_features_to_extract_selected).deserialize()
        X_tsfresh_all_features, X_tsfresh_selected_features = self.process_ml_features(
            X_tsfresh, y_tsfresh, "tsfresh-features", self.data_period_ml_train, self.tsfresh_features_to_extract_selected)
        self.tsfresh_features_to_extract_all = tsfresh.feature_extraction.settings.from_columns(X_tsfresh_all_features)
        assert len(X_tsfresh_all_features) == len(y_tsfresh)
        assert len(X_tsfresh_selected_features) == len(y_tsfresh)
        
        self.str_coupling_ml_classifier = coupling_ml_classifier
        clf_type = coupling_simulator.coupling_ml_classifiers[self.str_coupling_ml_classifier]
        
        self.coupling_classifier_basic_all_features = Classifier.get_clf(clf_type)
        self.coupling_classifier_basic_all_features = self.coupling_classifier_basic_all_features.fit(
            X_basic_all_features, y_basic)
        self.coupling_classifier_basic_selected_features = Classifier.get_clf(clf_type)
        self.coupling_classifier_basic_selected_features = self.coupling_classifier_basic_selected_features.fit(
            X_basic_selected_features, y_basic)
        logging.info("basic all features shape: " + str(X_basic_all_features.shape))
        logging.info("basic selected features shape: " + str(X_basic_selected_features.shape))
        logging.info("truth shape: " + str(y_basic.shape))
        
        self.coupling_classifier_tsfresh_all_features = Classifier.get_clf(clf_type)
        self.coupling_classifier_tsfresh_all_features = self.coupling_classifier_tsfresh_all_features.fit(
            X_tsfresh_all_features, y_tsfresh)
        self.coupling_classifier_tsfresh_selected_features = Classifier.get_clf(clf_type)
        self.coupling_classifier_tsfresh_selected_features = self.coupling_classifier_tsfresh_selected_features.fit(
            X_tsfresh_selected_features, y_tsfresh)
        logging.info("tsfresh all features shape: " + str(X_tsfresh_all_features.shape))
        logging.info("tsfresh selected features shape: " + str(X_tsfresh_selected_features.shape))
        logging.info("truth shape: " + str(y_tsfresh.shape))
        
        self.data_period_localization = float(data_period_localization)
        self.localization_pos_in_area = map(int, localization_pos_in_area.strip('[]').split(','))
        self.wifi_scans = localization_server.load_data(path_wifi_scans)
        self.ble_scans = localization_server.load_data(path_ble_scans)
        self.path_wifi_scans = path_wifi_scans
        self.path_ble_scans = path_ble_scans
        self.localization_pos_out_area = [pos for pos in self.wifi_scans.keys() if pos not in self.localization_pos_in_area]
    
    def process_ml_features(self, X, y, filename, data_period_ml_train, tsfresh_features_to_extract_selected=None):
        logging.info("process ml features: " + filename)
        data_exists = False
        file_exists = os.path.isfile(os.path.join(__location__, filename))
        if file_exists:
            X_features = DillSerializer(os.path.join(__location__, filename)).deserialize()
            if data_period_ml_train in X_features:
                X_all_features = X_features[data_period_ml_train][0]
                X_selected_features = X_features[data_period_ml_train][1]
                data_exists = True
        if not data_exists:
            if tsfresh_features_to_extract_selected:
                tsfresh_features = TsFreshFeatures()
                X_all_features = tsfresh_features.extract(X, y)
                X_selected_features = tsfresh_features.extract_selected_features(
                                            X, tsfresh_features_to_extract_selected)
            else:
                basic_features = BasicFeatures()
                X_all_features = basic_features.extract(X)
                X_selected_features = basic_features.extract_selected_features(X)
            if not file_exists:
                X_features = {data_period_ml_train: (X_all_features, X_selected_features)}
            else:
                X_features[data_period_ml_train] = (X_all_features, X_selected_features)
            DillSerializer(os.path.join(__location__, filename)).serialize(X_features)
        return X_all_features, X_selected_features

def evaluate_callback(accept_clients, reject_clients, runtime):
    accept_clients = list(accept_clients)
    reject_clients = list(reject_clients)
    groundtruth_accept_clients = [client.factory.get_mac() for client in clients if client.coupling_groundtruth]
    groundtruth_reject_clients = [client.factory.get_mac() for client in clients if not client.coupling_groundtruth]
    return StaticCouplingResult(accept_clients, reject_clients,
                                groundtruth_accept_clients, groundtruth_reject_clients,
                                runtime, mac_mapping)

def stop_reactor_callback():
    logging.info("stop reactor")
    reactor.stop()
    
def get_mac(identifier):
    if identifier not in mac_mapping:
        mac_mapping[identifier] = create_random_mac()
        for client in clients:
            if client.factory.transport:
                remote = client.factory.transport.getHost()
                if identifier == remote.host or identifier == remote.port:
                    client.factory.set_mac(mac_mapping[identifier])
                    break
    return mac_mapping[identifier]

def run(parameter):
    access_point = AccessPoint()
    access_point.deny_hosts = MagicMock()
    access_point.get_mac = MagicMock(side_effect=get_mac)
    access_point.get_num_connected_clients = MagicMock(return_value=parameter.num_clients)
    localization_pos_in_iter = cycle(parameter.localization_pos_in_area)
    localization_pos_out_iter = cycle(parameter.localization_pos_out_area)
    
    for _ in range(parameter.num_clients-parameter.num_reject_clients): # accept client
        localization_pos_in = next(localization_pos_in_iter)
        coupling_data_provider = CouplingDataProvider(parameter.light_signal, parameter.light_signal_time,
                                                      parameter.wifi_scans[localization_pos_in],
                                                      parameter.ble_scans[localization_pos_in])
        clients.append(ClientController(parameter.server_ip, parameter.server_port,
                                        coupling_data_provider, True))
    
    datalen = len(parameter.light_signal)
    mean = parameter.light_signal.mean()
    std = parameter.light_signal.std()
    #light_signal_random, light_signal_random_time = light_analysis.load_random_light_signal()
    for _ in range(parameter.num_reject_clients): # reject client
        localization_pos_out = next(localization_pos_out_iter)
        light_signal_random = numpy.random.normal(mean, std, datalen)
        #coupling_data_provider = CouplingDataProvider(light_signal_random, light_signal_random_time,
        #                                              parameter.wifi_scans[localization_pos_out], 
        #                                              parameter.ble_scans[localization_pos_out])
        coupling_data_provider = CouplingDataProvider(light_signal_random, parameter.light_signal_time,
                                                      parameter.wifi_scans[localization_pos_out], 
                                                      parameter.ble_scans[localization_pos_out])
        clients.append(ClientController(parameter.server_ip, parameter.server_port,
                                        coupling_data_provider, False))
    
    server = ServerController(parameter.server_port, access_point,
                              parameter.data_period_coupling, parameter.coupling_compare_method,
                              parameter.coupling_similarity_threshold, parameter.equalize_method,
                              parameter.data_period_localization, parameter.localization_pos_in_area,
                              parameter.localization_pos_out_area, parameter.path_wifi_scans, parameter.path_ble_scans,
                              parameter.coupling_classifier_basic_all_features,
                              parameter.coupling_classifier_basic_selected_features,
                              parameter.coupling_classifier_tsfresh_all_features,
                              parameter.coupling_classifier_tsfresh_selected_features,
                              parameter.tsfresh_features_to_extract_all, parameter.tsfresh_features_to_extract_selected,
                              evaluate_callback, stop_reactor_callback)
    server.start()
    for client in clients:
        client.start()
    logging.info("run server and clients")
    reactor.run()
    
def test():
    testbed = "vm" # server, vm
    server_ip = "localhost"
    server_port = 1026
    num_clients = 10
    num_reject_clients = 0
    len_light_pattern = 8
    data_period_coupling = 0.07
    coupling_compare_method = "pearson"
    coupling_similarity_threshold = 0.7
    equalize_method = "dtw"
    data_period_ml_train = 0.05
    coupling_ml_classifier = "Random Forest"
    path_ml_train_data = os.path.join(__location__, "..", "ml-train-data", testbed)
    data_period_localization = 5
    localization_pos_in_area = str(coupling_simulator.localization_pos_in_area)
    fingerprint_directory = os.path.join(__location__, "..", "..", "localization", "data")
    path_wifi_scans = os.path.join(fingerprint_directory, "wifi-fingerprints")
    path_ble_scans = os.path.join(fingerprint_directory, "bluetooth-fingerprints")
    parameter = SimulationData(server_ip, server_port,
                               num_clients, num_reject_clients, len_light_pattern,
                               data_period_coupling, coupling_compare_method, coupling_similarity_threshold, equalize_method,
                               data_period_ml_train, path_ml_train_data, coupling_ml_classifier,
                               data_period_localization, localization_pos_in_area, path_wifi_scans, path_ble_scans)
    run(parameter)
    
def evaluation():
    parameter = SimulationData(sys.argv[1], sys.argv[2], sys.argv[3],
                               sys.argv[4], sys.argv[5], sys.argv[6],
                               sys.argv[7], sys.argv[8], sys.argv[9],
                               sys.argv[10], sys.argv[11], sys.argv[12],
                               sys.argv[13], sys.argv[14], sys.argv[15],
                               sys.argv[16])
    run(parameter)
    
if __name__ == "__main__":
    logging.basicConfig(filename="static-coupling-simulation.log", level=logging.DEBUG)
    #test()
    evaluation()
    