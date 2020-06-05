from __future__ import division
import os
import glob
import numpy
import random
import logging
import itertools
import coupling.utils.misc as misc
from collections import defaultdict
from utils.serializer import DillSerializer
from coupling.localization.localization_server import Localization
import coupling.light_grouping_pattern.light_analysis as light_analysis
from coupling.localization.localization_server import LocalizationFeatures
from coupling.device_grouping.online.machine_learning_features import Classifier
from coupling.device_grouping.online.machine_learning_features import BasicFeatures
from coupling.device_grouping.online.machine_learning_features import TsFreshFeatures
import coupling.device_grouping.online.coupling_dynamic_simulator as coupling_simulator
from coupling.device_grouping.offline.sampling_time import get_pattern_max_sampling_period
from coupling.device_grouping.online.static.coupling_data_provider import CouplingDataProvider

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

class ClientData:
    
    def __init__(self, light_signal, light_signal_time, wifi_scan, ble_scan):
        self.light_signal = light_signal
        self.light_signal_time = light_signal_time
        self.wifi_scan = wifi_scan
        self.ble_scan = ble_scan
    
class RoomData:
    
    def __init__(self, light_signal, light_pattern, light_pattern_duration,
                 coupling_classifier_basic_all_features,
                 coupling_classifier_basic_selected_features,
                 coupling_classifier_tsfresh_selected_features,
                 tsfresh_selected_features,
                 wifi_scan, ble_scan,
                 wifi_features, ble_features):
        
        self.light_signal = light_signal
        self.light_pattern = light_pattern
        self.light_pattern_duration = light_pattern_duration
        self.wifi_scan = wifi_scan
        self.ble_scan = ble_scan
        self.wifi_localization = Localization(wifi_features)
        self.ble_localization = Localization(ble_features)
        self.tsfresh_selected_features = tsfresh_selected_features
        self.coupling_classifier_basic_all_features = coupling_classifier_basic_all_features
        self.coupling_classifier_basic_selected_features = coupling_classifier_basic_selected_features
        self.coupling_classifier_tsfresh_selected_features = coupling_classifier_tsfresh_selected_features
    
class CouplingTestbed:
    
    def __init__(self, num_users, num_rooms, simulation_duration,
                 data_period_ml_train, path_ml_train_data, coupling_ml_classifier,
                 path_localization_data, localization_room_to_pos,
                 intra_room_distance=2, inter_room_distance=3, room_step_size=2,
                 len_light_patterns=range(2, 11, 2), check_similarity_runtime=False):
        
        self.stay_durations = numpy.random.multinomial(
            simulation_duration, numpy.ones(num_rooms)/num_rooms)        
        self.rooms, self.room_distribution = self.select_rooms(num_rooms)
        self.room_distances = self.calculate_room_distances(
            self.rooms, inter_room_distance, intra_room_distance, room_step_size)
        self.user_routes = self.create_user_routes(num_users, self.stay_durations, self.room_distribution)
        
        path_ble_scans = os.path.join(path_localization_data, "bluetooth-fingerprints")
        ble_scans, ble_features = LocalizationFeatures.from_multiple_rooms(
            path_ble_scans, localization_room_to_pos)
        
        path_wifi_scans = os.path.join(path_localization_data, "wifi-fingerprints")
        wifi_scans, wifi_features = LocalizationFeatures.from_multiple_rooms(
            path_wifi_scans, localization_room_to_pos)
        
        light_data = self.create_light_data(
            len_light_patterns, get_pattern_max_sampling_period(), check_similarity_runtime)
        tsfresh_selected_features, coupling_classifiers = self.create_ml_coupling(
            coupling_ml_classifier, path_ml_train_data, data_period_ml_train, len_light_patterns)
        
        self.room_data = dict()
        self.client_data = dict()
        len_light_patterns = sorted(light_data.keys())
        iter_len_light_patterns = itertools.cycle(len_light_patterns)
        loc_rooms = sorted(ble_scans.keys())
        iter_loc_rooms = itertools.cycle(loc_rooms)
        
        for room_id in self.rooms:
            loc_room_id = next(iter_loc_rooms)
            len_light_pattern = next(iter_len_light_patterns)
            
            raw_light_signal = light_data[len_light_pattern][0]
            raw_light_signal_time = light_data[len_light_pattern][1]
            light_signal = light_data[len_light_pattern][2]
            light_pattern = light_data[len_light_pattern][3]
            light_pattern_duration = light_data[len_light_pattern][4]
            
            coupling_classifier_basic_all_features = coupling_classifiers[len_light_pattern][0]
            coupling_classifier_basic_selected_features = coupling_classifiers[len_light_pattern][1]
            coupling_classifier_tsfresh_selected_features = coupling_classifiers[len_light_pattern][2]
            
            self.room_data[room_id] = RoomData(light_signal, light_pattern, light_pattern_duration,
                                               coupling_classifier_basic_all_features,
                                               coupling_classifier_basic_selected_features,
                                               coupling_classifier_tsfresh_selected_features,
                                               tsfresh_selected_features,
                                               wifi_scans[loc_room_id], ble_scans[loc_room_id],
                                               wifi_features[loc_room_id], ble_features[loc_room_id])
            
            self.client_data[room_id] = ClientData(raw_light_signal, raw_light_signal_time,
                                                   wifi_scans[loc_room_id], ble_scans[loc_room_id])
    
    def create_light_data(self, len_light_patterns, sampling_period, check_runtime):
        light_signal_data = dict()
        for len_light_pattern in len_light_patterns:
            raw_light_signal, raw_light_signal_time = light_analysis.load_light_pattern(len_light_pattern)
            coupling_data_provider = CouplingDataProvider(
                raw_light_signal, raw_light_signal_time, None, None)
            signal_pattern_found = False
            while not signal_pattern_found:
                light_signal, light_signal_time = coupling_data_provider.get_light_data(sampling_period)
                try:
                    light_pattern_duration, light_pattern = light_analysis.detect_cycle_by_sequence(
                        light_signal, light_signal_time)
                    signal_pattern_found = misc.valid_light_pattern(
                        light_pattern_duration, len_light_pattern)
                except:
                    pass
            light_signal_data[len_light_pattern] = (raw_light_signal, raw_light_signal_time,
                                                    light_signal, light_pattern, light_pattern_duration)
        
        if check_runtime:
            import time
            import coupling.utils.vector_similarity as vector_similarity
            for len_light_pattern1, len_light_pattern2 in itertools.combinations(light_signal_data, 2):
                signal1 = light_signal_data[len_light_pattern1][2]
                signal2 = light_signal_data[len_light_pattern2][2]
                start = time.time()
                vector_similarity.pearson(signal1, signal2, vector_similarity.equalize_methods.dtw)
                end = time.time()
                print("signal 1: ", len(signal1))     
                print("signal 2: ", len(signal2))
                print("duration: ", end - start)
            import sys
            sys.exit(0)
        return light_signal_data
    
    def create_ml_coupling(self, coupling_ml_classifier, path_ml_train_data,
                           data_period_ml_train, len_light_patterns, rounds=0):
        tsfresh_selected_features = os.path.join(__location__, "..", "tsfresh-features-to-be-extracted")
        tsfresh_selected_features = DillSerializer(tsfresh_selected_features).deserialize()
        single_raw_feature_data = glob.glob(os.path.join(path_ml_train_data, "single-*-raw-feature-data"))[0]
        single_raw_feature_data = DillSerializer(single_raw_feature_data).deserialize()
        coupling_classifiers = dict()
        for len_light_pattern in len_light_patterns:    
            logging.info("ml len light pattern: " + str(len_light_pattern))
            logging.info("data period ml train: " + str(data_period_ml_train))
            logging.info("create features basic")
            X_basic = single_raw_feature_data[len_light_pattern][data_period_ml_train][rounds].X_basic
            y_basic = single_raw_feature_data[len_light_pattern][data_period_ml_train][rounds].y_basic
            X_basic_all_features, X_basic_selected_features = self.process_ml_features(
                X_basic, y_basic, "basic-features", len_light_pattern)
            assert len(X_basic_all_features) == len(y_basic)
            assert len(X_basic_selected_features) == len(y_basic)
            logging.info("create features tsfresh")
            X_tsfresh = single_raw_feature_data[len_light_pattern][data_period_ml_train][rounds].X_tsfresh
            y_tsfresh = single_raw_feature_data[len_light_pattern][data_period_ml_train][rounds].y_tsfresh
            _, X_tsfresh_selected_features = self.process_ml_features(
                X_tsfresh, y_tsfresh, "tsfresh-features", len_light_pattern, tsfresh_selected_features)
            assert len(X_tsfresh_selected_features) == len(y_tsfresh)
            clf_type = coupling_simulator.coupling_ml_classifiers[coupling_ml_classifier]
            logging.info("create classifier basic all and selected")
            coupling_classifier_basic_all_features = Classifier.get_clf(clf_type)
            coupling_classifier_basic_all_features = coupling_classifier_basic_all_features.fit(
                X_basic_all_features, y_basic)
            coupling_classifier_basic_selected_features = Classifier.get_clf(clf_type)
            coupling_classifier_basic_selected_features = coupling_classifier_basic_selected_features.fit(
                X_basic_selected_features, y_basic)
            logging.info("create classifier tsfresh selected")
            coupling_classifier_tsfresh_selected_features = Classifier.get_clf(clf_type)
            coupling_classifier_tsfresh_selected_features = coupling_classifier_tsfresh_selected_features.fit(
                X_tsfresh_selected_features, y_tsfresh)
            coupling_classifiers[len_light_pattern] = (coupling_classifier_basic_all_features,
                                                       coupling_classifier_basic_selected_features,
                                                       coupling_classifier_tsfresh_selected_features)
        return tsfresh_selected_features, coupling_classifiers
    
    def process_ml_features(self, X, y, filename, len_light_pattern, tsfresh_selected_features=None):
        data_exists = False
        file_exists = os.path.isfile(os.path.join(__location__, filename))
        if file_exists:
            X_features = DillSerializer(os.path.join(__location__, filename)).deserialize()
            if len_light_pattern in X_features:
                X_all_features = X_features[len_light_pattern][0]
                X_selected_features = X_features[len_light_pattern][1]
                data_exists = True    
        if not data_exists:
            if tsfresh_selected_features:
                tsfresh_features = TsFreshFeatures()
                X_all_features = tsfresh_features.extract(X, y)
                X_selected_features = tsfresh_features.extract_selected_features(
                                            X, tsfresh_selected_features)
            else:
                basic_features = BasicFeatures()
                X_all_features = basic_features.extract(X)
                X_selected_features = basic_features.extract_selected_features(X)
            if not file_exists:
                X_features = {len_light_pattern: (X_all_features, X_selected_features)}
            else:
                X_features[len_light_pattern] = (X_all_features, X_selected_features)
            DillSerializer(os.path.join(__location__, filename)).serialize(X_features)
        return X_all_features, X_selected_features
    
    def select_rooms(self, num_rooms, duplicates=True):
        if duplicates:
            room_distribution = [random.choice(range(num_rooms)) for _ in range(num_rooms)]
        else:
            room_distribution = random.sample(range(num_rooms), num_rooms)
        return set(room_distribution), room_distribution
    
    def calculate_room_distances(self, rooms, inter_room_distance, intra_room_distance, room_step_size):
        room_distances = defaultdict(dict)
        for start_room, end_room in itertools.combinations(rooms, 2):
            key1 = start_room
            key2 = end_room
            distance = 0
            same_side = (start_room % 2 == 0) and (end_room % 2 == 0) or \
                        (start_room % 2 != 0) and (end_room % 2 != 0)
            if not same_side:
                distance += inter_room_distance
                if end_room > start_room and end_room % 2 != 0:
                    start_room += 1
                elif end_room > start_room and end_room % 2 == 0:
                    start_room -= 1
            room_steps = (end_room - start_room) / room_step_size
            distance += room_steps * intra_room_distance
            room_distances[key1][key2] = distance
            room_distances[key2][key1] = distance
        # user stays within room
        for room in rooms:
            room_distances[room][room] = 0
        return room_distances
    
    def create_user_routes(self, num_users, stay_durations, room_distribution):
        if len(stay_durations) == 1:
            group_sizes = [random.choice(range(num_users, num_users+1)) for _ in stay_durations]
        else:
            group_sizes = [random.choice(range(1, num_users+1)) for _ in stay_durations]
        group_clients = [random.sample(range(num_users), group_size) for group_size in group_sizes]                
        assert len(stay_durations) == len(room_distribution) == len(group_clients)
        testbed = zip(stay_durations, room_distribution, group_clients)
        user_routes = dict()
        users = set([group for time_group in testbed for group in time_group[2]])
        for user in users:
            user_route = [(time_group[0], time_group[1]) for time_group in testbed if user in time_group[2]]
            user_routes[user] = user_route
        return user_routes
    
    def get_user_routes(self):
        return self.user_routes
    
    def get_room_distances(self):
        return self.room_distances
    
    def get_client_data(self):
        return self.client_data
    
    def get_room_data(self):
        return self.room_data
    
def main():
    testbed = "vm"
    simulation_duration = 180 # seconds
    num_users = 1
    num_rooms = 10
    data_period_ml_train = 0.08
    coupling_ml_classifier = "Random Forest"
    path_ml_train_data = os.path.join(__location__, "..", "ml-train-data", testbed)
    path_localization_data = os.path.join(__location__, "..", "..", "localization", "data")
    localization_room_to_pos = str(coupling_simulator.localization_room_to_pos)
    CouplingTestbed(int(num_users), int(num_rooms), int(simulation_duration),
                    float(data_period_ml_train), path_ml_train_data, coupling_ml_classifier,
                    path_localization_data, localization_room_to_pos)
    
if __name__ == "__main__":
    main()
    