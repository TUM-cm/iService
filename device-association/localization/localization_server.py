from __future__ import division
import os
import math
import numpy
import random
import itertools
import scipy.stats
import scipy.spatial
from enum import Enum
from sklearn import svm
import coupling.utils.misc # for plt settings
from collections import Counter
import matplotlib.pyplot as plt
from collections import defaultdict
from utils.serializer import JsonSerializer
from sklearn.ensemble import RandomForestClassifier

s_to_ms = 1e3
__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

class Index(Enum):
    BSSI = 0
    RSSI = 1

def load_data(path_scans):
    data = JsonSerializer(path_scans).deserialize()
    database_fingerprint = dict()
    for position, scan in data.items():
        fingerprint = scan["entries"]
        bssi = numpy.array([entry["mac"].upper() for entry in fingerprint])
        rssi = numpy.array([entry["rssi"] for entry in fingerprint])
        nonzero = numpy.nonzero(rssi)
        bssi = bssi[nonzero]
        rssi = rssi[nonzero]
        assert len(bssi) == len(rssi)
        database_fingerprint[int(position)] = (bssi, rssi)
    return database_fingerprint

def get_rssi(x, y):
    bssi_x = x[Index.BSSI.value]
    bssi_y = y[Index.BSSI.value]        
    rssi_x = x[Index.RSSI.value]
    rssi_y = y[Index.RSSI.value]
    ap_overlap = numpy.intersect1d(bssi_x, bssi_y)
    rssi_x_ret = numpy.zeros(shape=(len(ap_overlap)), dtype=numpy.int)
    rssi_y_ret = numpy.zeros(shape=(len(ap_overlap)), dtype=numpy.int)
    for i, ap in enumerate(ap_overlap):
        ap_idx = numpy.where(bssi_x == ap)
        rssi = rssi_x[ap_idx]
        rssi_x_ret[i] = numpy.median(rssi)
        ap_idx = numpy.where(bssi_y == ap)
        rssi = rssi_y[ap_idx]
        rssi_y_ret[i] = numpy.median(rssi)
    return rssi_x_ret, rssi_y_ret

class DatasetStatistics:
    
    def __init__(self, path_wifi_fingerprints, path_ble_fingerprints):
        self.wifi_fingerprints = load_data(path_wifi_fingerprints)
        self.ble_fingerprints = load_data(path_ble_fingerprints)
        self.total_num_ap, self.ap_num_per_scan = self.get_occurrences(self.wifi_fingerprints)
        self.total_num_beacon, self.beacon_num_per_scan = self.get_occurrences(self.ble_fingerprints)
        self.wifi_unique_total, self.wifi_unique_ratio = self.get_unique_ratio(self.wifi_fingerprints)
        self.ble_unique_total, self.ble_unique_ratio = self.get_unique_ratio(self.ble_fingerprints)
    
    def pprint(self, sampled_wifi_entries, sampled_ble_entries):
        print("### WiFi")
        print("total num AP: ", self.total_num_ap)
        print("AP per scan: ", self.ap_num_per_scan)
        print("Total unique per scan:", numpy.mean(self.wifi_unique_total))
        print("Ratio unique per scan: ", [round(entry*100, 2) for entry in self.wifi_unique_ratio])
        print("Ratio sampled APs:", round(100*numpy.median([sampled_wifi_entries/total for total in self.wifi_unique_total]), 2))
        print("### BLE")
        print("total num beacon: ", self.total_num_beacon)
        print("beacon per scan: ", self.beacon_num_per_scan)
        print("Total unique per scan:", numpy.mean(self.ble_unique_total))
        print("Ratio unique per scan: ", [round(entry*100, 2) for entry in self.ble_unique_ratio])
        print("Ratio sampled beacons:", round(100*numpy.mean([sampled_ble_entries/total for total in self.ble_unique_total]),2))
    
    def get_unique_ratio(self, fingerprints):
        unique_total = list()
        unique_ratio = list()
        for fingerprint in fingerprints.values():
            unique_bssi = set()
            assert len(fingerprint[Index.BSSI.value]) == len(fingerprint[Index.RSSI.value])
            for mac in fingerprint[Index.BSSI.value]:
                unique_bssi.add(mac) # without duplicates
            all_scan_values = len(fingerprint[Index.BSSI.value])
            unique_scan = len(unique_bssi)
            #duplicates = all_scan_values - scan_without_duplicates
            #duplicates_ratio.append(duplicates / all_scan_values)
            unique_ratio.append(unique_scan / all_scan_values)
            unique_total.append(unique_scan)
        return unique_total, unique_ratio
    
    def get_occurrences(self, fingerprints):
        bssi = set()
        bssi_per_scan = list()
        for fingerprint in fingerprints.values():
            per_scan = set()
            for mac in fingerprint[Index.BSSI.value]:
                per_scan.add(mac)
                bssi.add(mac)
            bssi_per_scan.append(len(per_scan))
        return len(bssi), (numpy.min(bssi_per_scan),
            numpy.max(bssi_per_scan), numpy.mean(bssi_per_scan))

class LocalizationFeatures:
    
    @classmethod
    def from_single_room(cls, path_scans, pos_in_area):
        scans = load_data(path_scans)
        pos_out_area = list(set(scans.keys()).difference(pos_in_area))
        cls.scans = scans
        cls.pos_in_area = pos_in_area
        cls.pos_out_area = pos_out_area
        return cls(scans, pos_in_area, pos_out_area)
    
    @classmethod
    def from_multiple_rooms(cls, path_scans, map_room_to_pos):
        features = dict()
        room_scans = dict()
        scans = load_data(path_scans)
        for room_id, positions in map_room_to_pos.items():
            pos_in_area = positions
            pos_out_area = list()
            for other_room_id in map_room_to_pos:
                if room_id != other_room_id:
                    pos_out_area.extend(map_room_to_pos[other_room_id])            
            bssid = numpy.concatenate([scans[pos][0] for pos in positions])
            rssi = numpy.concatenate([scans[pos][1] for pos in positions])
            room_scans[room_id] = (bssid, rssi)
            features[room_id] = cls(scans, pos_in_area, pos_out_area)
        return room_scans, features
    
    def __init__(self, scans, pos_in_area, pos_out_area, len_combination=2, num_features=10):
        self.num_features = num_features
        self.imputing_values = self.ImputingValues(scans, self)
        self.groundtruth = dict()
        for pos_in in pos_in_area:
            self.groundtruth[pos_in] = 1
        for pos_out in pos_out_area:
            self.groundtruth[pos_out] = 0
        positions = pos_in_area + pos_out_area
        data_len = len(list(itertools.combinations(positions, len_combination)))
        self.X = numpy.empty(shape=(data_len, num_features))
        self.y = numpy.empty(shape=(data_len, 1))
        for i, (position1, position2) in enumerate(itertools.combinations(positions, len_combination)):
            pos1_data = scans[position1]
            pos2_data = scans[position2]
            feature = self.compute(pos1_data, pos2_data, self.imputing_values)
            within_area = self.groundtruth[position1] & self.groundtruth[position2]
            self.X[i] = feature
            self.y[i] = within_area
        self.y = self.y.ravel()
    
    class ImputingValues:
        def __init__(self, data, features, idx=0):
            # mean value over all non-nan values
            data_len = len(data) * len(data)
            spearman = numpy.empty(shape=(data_len))
            pearson = numpy.empty(shape=(data_len))
            manhattan = numpy.empty(shape=(data_len))
            euclidean = numpy.empty(shape=(data_len))
            for scan1 in data.values():
                for scan2 in data.values():
                    overlap = features.overlap(scan1, scan2)
                    rssi_scan1, rssi_scan2 = get_rssi(scan1, scan2)
                    manhattan[idx] = features.manhattan_distance(rssi_scan1, rssi_scan2, overlap)
                    euclidean[idx] = features.euclidean_distance(rssi_scan1, rssi_scan2, overlap)
                    spearman[idx] = features.spearman_correlation(rssi_scan1, rssi_scan2)
                    pearson[idx] = features.pearson_correlation(rssi_scan1, rssi_scan2)
                    idx += 1
            self.mean_spearman = numpy.mean(spearman[~numpy.isnan(spearman)])
            self.mean_pearson = numpy.mean(pearson[~numpy.isnan(pearson)])
            self.mean_manhattan = numpy.mean(manhattan[~numpy.isnan(manhattan)])
            self.mean_euclidean = numpy.mean(euclidean[~numpy.isnan(euclidean)])
    
    def get_groundtruth(self, position):
        return self.groundtruth[position]
    
    def compute(self, scan1, scan2, imputing_values):
        return self.__compute(scan1, scan2, imputing_values, self.num_features)
    
    def __compute(self, scan1, scan2, imputing_values, num_features):
        feature = numpy.empty(shape=(num_features))
        overlap = self.overlap(scan1, scan2)
        feature[0] = overlap
        feature[1] = self.union(scan1, scan2)
        feature[2] = self.jaccard_distance(scan1, scan2)
        feature[3] = self.non_overlap(scan1, scan2)
        feature[4] = self.share_top_ap(scan1, scan2)
        feature[5] = self.share_range_ap(scan1, scan2)
        rssi_scan1, rssi_scan2 = get_rssi(scan1, scan2)
        feature[6] = self.get_value(self.spearman_correlation, imputing_values.mean_spearman,
                                    rssi_scan1, rssi_scan2)
        feature[7] = self.get_value(self.pearson_correlation, imputing_values.mean_pearson,
                                    rssi_scan1, rssi_scan2)
        feature[8] = self.get_value(self.manhattan_distance, imputing_values.mean_manhattan,
                                    rssi_scan1, rssi_scan2, overlap)
        feature[9] = self.get_value(self.euclidean_distance, imputing_values.mean_euclidean,
                                    rssi_scan1, rssi_scan2, overlap)
        return feature
    
    def get_value(self, method, imputing_value, *arg):
        value = None
        try:
            if len(arg[0]) > 0:
                if len(arg) == 2:
                    value = method(arg[0], arg[1])
                elif len(arg) == 3:
                    value = method(arg[0], arg[1], arg[2])
            else:
                value = imputing_value
        except:
            value = imputing_value
        if math.isnan(value) or value == None:
            value = imputing_value
        return value
    
    # AP presence
    def overlap(self, x, y): # raw count of overlapping routers
        return len(numpy.intersect1d(x[Index.BSSI.value], y[Index.BSSI.value]))
    
    def union(self, x, y): # size of the union of two lists
        return len(numpy.union1d(x[Index.BSSI.value], y[Index.BSSI.value]))
    
    # range: 0-1
    def jaccard_distance(self, x, y): # ratio between size of intersection and size of union of two lists
        bssi_x = x[Index.BSSI.value]
        bssi_y = y[Index.BSSI.value]
        union = len(numpy.union1d(bssi_x, bssi_y))
        intersection = len(numpy.intersect1d(bssi_x, bssi_y))
        jaccard = (union - intersection) / union
        return jaccard
    
    def non_overlap(self, x, y): # non-overlap: raw count of non-overlapping routers (size of union minus size of overlap)
        return self.union(x, y) - self.overlap(x, y)
    
    # RSSI of overlapping routers
    # RSSI range: -100 to 0 dBm, closer to 0 is higher strength
    def spearman_correlation(self, x, y):
        if len(x) > 2 and len(y) > 2:
            return scipy.stats.spearmanr(x, y).correlation
        else:
            return numpy.nan
        
    def pearson_correlation(self, x, y):
        if len(x) > 2 and len(y) > 2:
            return scipy.stats.pearsonr(x, y)[0]
        else:
            return numpy.nan
    
    def manhattan_distance(self, x, y, overlap):
        value = scipy.spatial.distance.cityblock(x, y)
        if math.isnan(value) or overlap == 0:
            return value
        else:
            return value / overlap
    
    def euclidean_distance(self, x, y, overlap):
        value = scipy.spatial.distance.euclidean(x, y)
        if math.isnan(value) or overlap == 0:
            return value
        else:
            return value / overlap
    
    # AP presence + RSSI
    def share_top_ap(self, x, y):
        max_idx_x = numpy.argmax(x[Index.RSSI.value])
        max_idx_y = numpy.argmax(y[Index.RSSI.value])
        max_ap_x = x[Index.BSSI.value][max_idx_x]
        max_ap_y = y[Index.BSSI.value][max_idx_y]
        return int(max_ap_x == max_ap_y)
    
    def share_range_ap(self, x, y, rssi_range=6): #top AP +/- 6db
        # positive if at least one overlapping access point in the lists
        # of routers of A and B within 6dB from the top router
        max_rssi_x = x[Index.RSSI.value].max()
        max_rssi_y = y[Index.RSSI.value].max()
        if max_rssi_x >= max_rssi_y:
            bottom_rssi = max_rssi_x - rssi_range
        else:
            bottom_rssi = max_rssi_y - rssi_range
        rssi_idx = numpy.where(x[Index.RSSI.value] >= bottom_rssi)[0]
        bssi_x = x[Index.BSSI.value][rssi_idx]
        rssi_idx = numpy.where(y[Index.RSSI.value] >= bottom_rssi)[0]
        bssi_y = y[Index.BSSI.value][rssi_idx]
        ap_overlap = numpy.intersect1d(bssi_x, bssi_y)
        return int(len(ap_overlap) > 0)
    
class ReasoningRandom:
    
    def predict(self, _):
        return int(random.getrandbits(1))
    
class ReasoningFiltering:
    
    def __init__(self, features):
        self.features = features
    
    def predict(self, feature):
        feature_distance = defaultdict(list)
        for X, y in zip(self.features.X, self.features.y):
            distance = scipy.spatial.distance.cosine(feature, X)
            feature_distance[y].append(distance)
        distance_in_area = numpy.mean(feature_distance[1])
        distance_out_area = numpy.mean(feature_distance[0])
        return int(distance_in_area < distance_out_area)
    
class ReasoningMachineLearning:
    
    def __init__(self, features):
        self.features = features
        self.rfc = RandomForestClassifier()
        self.rfc.fit(self.features.X, self.features.y)
        self.svm = svm.SVC()
        self.svm.fit(self.features.X, self.features.y)
    
    def predict_svm(self, feature):
        # calculate feature between scan1 and scan2
        # predict feature the class: in (1), out (0)
        feature = feature.reshape(1, -1)
        return int(self.svm.predict(feature))
    
    def predict_random_forest(self, feature):
        feature = feature.reshape(1, -1)
        return int(self.rfc.predict(feature))
    
class Localization:
    
    def __init__(self, features):
        self.features = features
        self.reasoning_random = ReasoningRandom()
        self.reasoning_filtering = ReasoningFiltering(self.features)
        self.reasoning_machine_learning = ReasoningMachineLearning(self.features)
    
    def evaluate(self, datalen):
        #from sklearn.model_selection import KFold
        #coupling_data_provider = CouplingDataProvider(None, None, parameter.wifi_scans[localization_pos_in], parameter.ble_scans[localization_pos_in])
        result_svm = dict()
        result_random = dict()
        result_filtering = dict()
        result_random_forest = dict()
        for test_position, test_scan in self.features.scans.items():
            svm = dict()
            random = dict()
            filtering = dict()
            random_forest = dict()
            ap_bssi = test_scan[0][:datalen]
            ap_rssi = test_scan[1][:datalen]
            test_scan = (ap_bssi, ap_rssi)
            for position, scan in self.features.scans.items():
                ap_bssi = scan[0][:datalen]
                ap_rssi = scan[1][:datalen]
                scan = (ap_bssi, ap_rssi)
                feature = self.features.compute(test_scan, scan, self.features.imputing_values)
                random[position] = self.reasoning_random.predict(None)
                filtering[position] = self.reasoning_filtering.predict(feature)
                svm[position] = self.reasoning_machine_learning.predict_svm(feature)
                random_forest[position] = self.reasoning_machine_learning.predict_random_forest(feature)
            result_svm[test_position] = self.in_area(svm, self.features.pos_in_area, self.features.pos_out_area)
            result_random[test_position] = self.in_area(random, self.features.pos_in_area, self.features.pos_out_area)
            result_filtering[test_position] = self.in_area(filtering, self.features.pos_in_area, self.features.pos_out_area)
            result_random_forest[test_position] = self.in_area(random_forest, self.features.pos_in_area, self.features.pos_out_area)
        print("random")
        self.accuracy(result_random, self.features.pos_in_area, self.features.pos_out_area)
        print("filtering")
        self.accuracy(result_filtering, self.features.pos_in_area, self.features.pos_out_area)
        print("svm")
        self.accuracy(result_svm, self.features.pos_in_area, self.features.pos_out_area)
        print("random forest")
        self.accuracy(result_random_forest, self.features.pos_in_area, self.features.pos_out_area)
    
    def in_area(self, prediction, pos_in_area, pos_out_area):
        predict_in_area = [prediction[pos] for pos in pos_in_area]
        predict_out_area = [prediction[pos] for pos in pos_out_area]
        positions = len(pos_in_area) + len(pos_out_area)
        return (sum(predict_in_area) / positions) > (sum(predict_out_area) / positions)
    
    # input predict: {pos_1: result={True|False}, pos_2: ... }
    def accuracy(self, predict, pos_in_area, pos_out_area):
        num_in_area = len(pos_in_area)
        num_out_area = len(pos_out_area)
        predict_in_area = [predict[pos] for pos in pos_in_area]
        predict_out_area = [predict[pos] for pos in pos_out_area]
        accuracy_in_area = Counter(predict_in_area)[True] / num_in_area
        accuracy_out_area = Counter(predict_out_area)[False] / num_out_area
        print ((accuracy_in_area + accuracy_out_area) / 2)
    
def offline_localization(path_ble_scans, path_wifi_scans):
    pos_in_area = [1, 2, 3, 4, 5]
    #from coupling.device_grouping.online.static.coupling_data_provider import CouplingDataProvider
    #from coupling.device_grouping.online.dynamic.coupling_data_provider import CouplingDataProvider
    #measurements_to_rooms = {1:[1,2,3,4,5], 2:[6,7,8,9], 3:[10,11], 4:[12,13], 5:[14,15,16,17,18,19],
    #                         6:[20,21,22,23], 7:[24,25,26,27], 8:[28,29,30,31], 9:[32,33,34,35], 10:[36,37,38,39]}
    
    for sampling_period in [2, 5, 10, 15, 20, 25, 30]:
        wifi_datalen = int(round(get_wifi_entries(sampling_period)))
        ble_datalen = int(round(get_ble_entries(sampling_period)))
        print("sampling period:", sampling_period)
        
        #print("WiFi")
        #wifi_features = LocalizationFeatures.from_multiple_rooms(path_wifi_scans, measurements_to_rooms)
        wifi_features = LocalizationFeatures.from_single_room(path_wifi_scans, pos_in_area)
        wifi_localization = Localization(wifi_features)
        wifi_localization.evaluate(wifi_datalen)
        
        print("BLE")
        #ble_features = LocalizationFeatures.from_multiple_rooms(path_ble_scans, measurements_to_rooms)
        ble_features = LocalizationFeatures.from_single_room(path_ble_scans, pos_in_area)
        ble_localization = Localization(ble_features)
        ble_localization.evaluate(ble_datalen)
        
def num_entries_per_ms(filename):
    entries_per_ms = list()
    data_path = os.path.join(__location__, "data", filename)
    fingerprints = JsonSerializer(data_path).deserialize()
    for scan in fingerprints.values():
        fingerprint = scan["entries"]
        bssi = numpy.array([entry["mac"] for entry in fingerprint])
        rssi = numpy.array([entry["rssi"] for entry in fingerprint])
        assert len(bssi) == len(rssi)
        duration = int(scan["stopTimestamp"]) - int(scan["startTimestamp"]) # ms
        entries_per_ms.append(len(bssi) / duration)
    return numpy.mean(entries_per_ms)

def num_encounters(path_ble_scans):
    dummy = load_data(path_ble_scans)
    len_bssi = list()
    for bssi, _ in dummy.values():
        len_bssi.append(len(bssi))
    print(numpy.mean(len_bssi))

def print_num_entries(duration):
    ble_entries_ms = num_entries_per_ms("bluetooth-fingerprints")
    wifi_entries_ms = num_entries_per_ms("wifi-fingerprints")
    print("BLE entries/s:", (duration * s_to_ms) * ble_entries_ms)
    print("Wi-Fi entries(s:", (duration * s_to_ms) * wifi_entries_ms)

def get_ble_entries(duration):
    ble_entries_ms = num_entries_per_ms("bluetooth-fingerprints")
    return (duration * s_to_ms) * ble_entries_ms

def get_wifi_entries(duration):
    wifi_entries_ms = num_entries_per_ms("wifi-fingerprints")
    return (duration * s_to_ms) * wifi_entries_ms

def plot_sampled_entries():
    for data_type, get_entries in [("wifi", get_wifi_entries), ("ble", get_ble_entries)]:
        periods = range(1, 31)
        entries = [get_entries(period) for period in periods]
        fig, ax = plt.subplots()
        ax.plot(periods, entries)
        #ax.axvline(entries[sampling_period-1], linestyle="--")
        ax.grid()
        ax.set_xlabel("Sampling period (s)")
        ax.set_ylabel("Sampling entries")
        ax.set_xticks(periods)
        #ax.set_xticklabels(periods)
        plot_format = "pdf"
        plt.show()
        filename = "localization-sampling-entries-" + data_type + "." + plot_format
        fig.savefig(filename, plot_format=plot_format, bbox_inches="tight")
        plt.close(fig)
    
def main():
    path_ble_scans = os.path.join(__location__, "data", "bluetooth-fingerprints")
    path_wifi_scans = os.path.join(__location__, "data", "wifi-fingerprints")
    
    sampling_period = 5
    #plot_sampled_entries()
    
    print("Data entries within time frame")
    print_num_entries(sampling_period)
    
    sampled_wifi_entries = get_wifi_entries(sampling_period)
    sampled_ble_entries = get_ble_entries(sampling_period)
    
    print("Dataset statistics")
    DatasetStatistics(path_wifi_scans, path_ble_scans).pprint(sampled_wifi_entries, sampled_ble_entries)
    
    print("Offline localization")
    offline_localization(path_ble_scans, path_wifi_scans)
    
if __name__ == '__main__':
    main()
    