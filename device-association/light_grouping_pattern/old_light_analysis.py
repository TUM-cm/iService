import os
import time
import numpy
import datetime
import collections
import matplotlib.pyplot as plt
import receive_light.decoding.smoothing as smoothing
from utils.serializer import JsonSerializer

def load_data(path="./data"):
    light_data = {}
    for f in os.listdir(path):
        print(f)
        full_path = os.path.join(path, f)
        flux, time = load_data_object(full_path)
        mode = f.split("_")[-1]
        time_on_off = f.split("_")[-2]
        light_data[time_on_off + ":" + mode] = (flux, time)    
    return collections.OrderedDict(sorted(light_data.items(),
                                          key=lambda entry: float(entry[0].split(":")[0]),
                                          reverse=True))

def load_data_object(full_path):
    raw_data = JsonSerializer(full_path).deserialize()
    flux = [entry["flux"] for entry in raw_data]
    if "." in raw_data[0]["time"]:
        time_format = '%H:%M:%S.%f'
    else:
        time_format = '%H:%M:%S'
    time = [datetime.datetime.strptime(entry["time"], time_format) for entry in raw_data]
    return flux, time

def plot(data):
    idx = 0
    data_len = len(data)
    plot_len = (data_len/8)*2
    plt.subplots_adjust(hspace=.35)
    for key, value in data.items():
        idx +=1
        plt.subplot(plot_len, 1, idx)
        plt.title(key)
        flux = value[0]
        time = value[1]
        binary = value[2]
        plt.plot(time, flux)
        idx += 1
        plt.subplot(plot_len, 1, idx)
        plt.step(time, binary)
        if plot_len == idx:
            idx = 0
            plt.show()

def analyze_data_batch():
    data = load_data()
    for key, value in data.items():
        signal = value[0]
        time = value[1]
        binary = smoothing.simple(signal)
        data[key] = (signal, time, binary)
    plot(data)
    
def total_seconds(x):
    return x.total_seconds()

def data_from_native_android():
    path = "./test_data/native_android"
    os.listdir(path)
    for f in os.listdir(path):
        full_path = os.path.join(path, f)
        raw_data = JsonSerializer(full_path).deserialize()
        voltage = [entry["flux"] for entry in raw_data]
        voltage_time = [datetime.datetime.fromtimestamp(entry["time"]/1e3) for entry in raw_data]
        voltage = numpy.asarray(voltage)
        voltage_time = numpy.asarray(voltage_time)
        #duration = get_duration(voltage, voltage_time)
        #get_total_seconds = numpy.vectorize(total_seconds, otypes=[numpy.double])
        #duration = get_total_seconds(duration)
        #print f
        #print duration
        #print statistics.get_summary(duration[1:])
        #print "--------------"