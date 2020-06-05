from __future__ import division
import numpy
import random
from enum import Enum
from light_timing.time_analysis import get_duration_monotonic
from coupling.localization.localization_server import num_entries_per_ms

class CouplingDataProvider:
    
    class DataType(Enum):
        LIGHT = 0
        WIFI = 1
        BLE = 2
    
    def __init__(self, light_signal, light_signal_time, wifi_scan, ble_scan):
        self.light_signal = light_signal
        self.light_signal_time = light_signal_time
        self.wifi_scan = wifi_scan
        self.ble_scan = ble_scan
        self.ms_to_s = int(1e3)
        self.ns_to_s = int(1e9)
        self.sampling_entries_per_second = dict()
        self.__add_light_sampling(self.sampling_entries_per_second)
        self.__add_wifi_sampling(self.sampling_entries_per_second)
        self.__add_ble_sampling(self.sampling_entries_per_second)
    
    def __add_light_sampling(self, sampling_entries_per_second):  
        durations = get_duration_monotonic("time_data_speed")
        data_entries = numpy.mean([duration[0] for duration in durations])
        time_interval = numpy.mean([duration[2] for duration in durations])
        light_sampling_per_time_unit = (data_entries / time_interval) * self.ns_to_s
        sampling_entries_per_second[CouplingDataProvider.DataType.LIGHT] = int(light_sampling_per_time_unit)
    
    def __add_wifi_sampling(self, sampling_entries_per_second):
        sampling_entries_per_second[CouplingDataProvider.DataType.WIFI] = \
            int(num_entries_per_ms("wifi-fingerprints") * self.ms_to_s)
    
    def __add_ble_sampling(self, sampling_entries_per_second):
        sampling_entries_per_second[CouplingDataProvider.DataType.BLE] = \
            int(num_entries_per_ms("bluetooth-fingerprints") * self.ms_to_s)
    
    def __get_sensing_range(self, data_period_second, data_type):
        sensing_data_entries = data_period_second * self.sampling_entries_per_second[data_type]
        sensing_data_entries = int(round(sensing_data_entries + 0.5))
        if data_type == CouplingDataProvider.DataType.LIGHT:
            data_len = len(self.light_signal)
        elif data_type == CouplingDataProvider.DataType.WIFI:
            data_len = len(self.wifi_scan[0])
        elif data_type == CouplingDataProvider.DataType.BLE:
            data_len = len(self.ble_scan[0])
        start_range = data_len - sensing_data_entries
        start_signal = random.choice(range(start_range))
        end_signal = start_signal + sensing_data_entries
        return start_signal, end_signal
    
    def get_light_data(self, data_period_second):
        start, end = self.__get_sensing_range(data_period_second, CouplingDataProvider.DataType.LIGHT)
        return self.light_signal[start:end], self.light_signal_time[start:end]
    
    def get_wifi_data(self, data_period_second):
        start, end = self.__get_sensing_range(data_period_second, CouplingDataProvider.DataType.WIFI)
        ap_mac = self.wifi_scan[0][start:end]
        ap_rssi = self.wifi_scan[1][start:end]
        assert len(ap_mac) == len(ap_rssi)
        return (ap_mac, ap_rssi)
    
    def get_ble_data(self, data_period_second):
        start, end = self.__get_sensing_range(data_period_second, CouplingDataProvider.DataType.BLE)
        ap_mac = self.ble_scan[0][start:end]
        ap_rssi = self.ble_scan[1][start:end]
        assert len(ap_mac) == len(ap_rssi)
        return (ap_mac, ap_rssi)
