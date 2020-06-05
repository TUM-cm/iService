from __future__ import division
import numpy
import random
from enum import Enum
from coupling.utils.misc import create_random_mac
from light_timing.time_analysis import get_duration_monotonic
from coupling.localization.localization_server import num_entries_per_ms

class CouplingDataProvider:
    
    class DataType(Enum):
        LIGHT = 0
        WIFI = 1
        BLE = 2
    
    def __init__(self, single_client_data):
        self.random_light_signal_time = single_client_data.light_signal_time
        light_signal = single_client_data.light_signal
        data_len = len(self.random_light_signal_time)
        self.random_light_signal = CouplingDataProvider.get_random_data(
            data_len, min_val=numpy.min(light_signal), max_val=numpy.max(light_signal))
        rssi = single_client_data.ble_scan[1]
        data_len = len(rssi)
        random_ble_mac = numpy.array([create_random_mac() for _ in range(data_len)])
        random_ble_rssi = CouplingDataProvider.get_random_data(
            data_len, min_val=numpy.min(rssi), max_val=numpy.max(rssi))
        self.random_ble_scan = (random_ble_mac, random_ble_rssi)
        rssi = single_client_data.wifi_scan[1]
        data_len = len(rssi)
        random_wifi_mac = numpy.array([create_random_mac() for _ in range(data_len)])
        random_wifi_rssi = CouplingDataProvider.get_random_data(
            data_len, min_val=numpy.min(rssi), max_val=numpy.max(rssi))
        self.random_wifi_scan = (random_wifi_mac, random_wifi_rssi)
        self.ms_to_s = int(1e3)
        self.ns_to_s = int(1e9)
        self.sampling_per_time_unit = dict()
        self.__add_light_sampling(self.sampling_per_time_unit)
        self.__add_wifi_sampling(self.sampling_per_time_unit)
        self.__add_ble_sampling(self.sampling_per_time_unit)
    
    @staticmethod
    def get_random_data(datalen, mean=0, std=1, min_val=None, max_val=None):    
        if min_val and max_val:
            return numpy.random.randint(min_val, max_val, datalen)
        else:
            return numpy.random.normal(mean, std, size=datalen)
    
    def set_random_signal(self):
        self.light_signal =  self.random_light_signal
        self.light_signal_time = self.random_light_signal_time
        self.wifi_scan = self.random_wifi_scan
        self.ble_scan = self.random_ble_scan
    
    def set_signal(self, client_data):
        self.light_signal = client_data.light_signal
        self.light_signal_time = client_data.light_signal_time
        self.wifi_scan = client_data.wifi_scan
        self.ble_scan = client_data.ble_scan
    
    def __add_light_sampling(self, sampling_per_time_unit):  
        durations = get_duration_monotonic("time_data_speed")
        data_entries = numpy.mean([duration[0] for duration in durations])
        time_interval = numpy.mean([duration[2] for duration in durations])
        light_sampling_per_time_unit = (data_entries / time_interval) * self.ns_to_s # entries per s
        sampling_per_time_unit[CouplingDataProvider.DataType.LIGHT] = int(light_sampling_per_time_unit)
    
    def __add_wifi_sampling(self, sampling_per_time_unit):
        sampling_per_time_unit[CouplingDataProvider.DataType.WIFI] = \
            int(num_entries_per_ms("wifi-fingerprints") * self.ms_to_s)
    
    def __add_ble_sampling(self, sampling_per_time_unit):
        sampling_per_time_unit[CouplingDataProvider.DataType.BLE] = \
            int(num_entries_per_ms("bluetooth-fingerprints") * self.ms_to_s)
    
    def __get_sensing_range(self, data_period, data_type):
        sensing_data_entries = data_period * self.sampling_per_time_unit[data_type]
        sensing_data_entries = int(round(sensing_data_entries + 0.5))
        if data_type == CouplingDataProvider.DataType.LIGHT:
            data_len = len(self.light_signal)
        elif data_type == CouplingDataProvider.DataType.WIFI:
            data_len = len(self.wifi_scan[0])
        elif data_type == CouplingDataProvider.DataType.BLE:
            data_len = len(self.ble_scan[0])
        start_range = data_len - sensing_data_entries
        start_position = random.choice(range(start_range))
        end_position = start_position + sensing_data_entries
        return start_position, end_position
    
    def get_light_data(self, data_period):
        start, end = self.__get_sensing_range(data_period, CouplingDataProvider.DataType.LIGHT)
        return self.light_signal[start:end], self.light_signal_time[start:end]
    
    def get_wifi_data(self, data_period):
        start, end = self.__get_sensing_range(data_period, CouplingDataProvider.DataType.WIFI)
        ap_mac = self.wifi_scan[0][start:end]
        ap_rssi = self.wifi_scan[1][start:end]
        assert len(ap_mac) == len(ap_rssi)
        return (ap_mac, ap_rssi)
    
    def get_ble_data(self, data_period):
        start, end = self.__get_sensing_range(data_period, CouplingDataProvider.DataType.BLE)
        ap_mac = self.ble_scan[0][start:end]
        ap_rssi = self.ble_scan[1][start:end]
        assert len(ap_mac) == len(ap_rssi)
        return (ap_mac, ap_rssi)
