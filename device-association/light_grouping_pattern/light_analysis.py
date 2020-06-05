from __future__ import division
import os
import numpy
import scipy.signal
import scipy.spatial
import matplotlib.pyplot as plt
from utils.serializer import NumpySerializer
import receive_light.decoding.smoothing as smoothing
from utils.sliding_window import sliding_window_array

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

def load_random_light_signal(concatenate=True):
    light_signal, light_signal_time = __load_data(
        "random_light_signal",
        "random_light_signal_time")
    if concatenate:
        light_signal = numpy.concatenate(light_signal)
        light_signal_time = numpy.concatenate(light_signal_time)
    return light_signal, light_signal_time

def __concatenate_light_signals(light_signal, light_signal_time):
    light_signal_time = light_signal_time.astype(numpy.int64)
    
    last_relative_time_per_frame = light_signal_time[:,-1]
    last_duration_per_frame = last_relative_time_per_frame - light_signal_time[:,-2]
    last_relative_time_per_frame = last_relative_time_per_frame + last_duration_per_frame
        
    time_offset = numpy.r_[0, last_relative_time_per_frame[:-1]]
    time_offset_per_frame = numpy.cumsum(time_offset).reshape(-1,1)
    light_signal_time = light_signal_time + time_offset_per_frame
    
    return numpy.concatenate(light_signal), numpy.concatenate(light_signal_time)

def load_light_pattern(len_light_pattern, concatenate=True, dist=10):
    light_signal, light_signal_time = __load_data(
        "voltage_%d_pattern_dist_%d" % (len_light_pattern, dist),
        "time_%d_pattern_dist_%d" % (len_light_pattern, dist))
    if concatenate:
        return __concatenate_light_signals(light_signal, light_signal_time)
    else:
        return light_signal, light_signal_time

def __load_data(f_voltage, f_time, path=os.path.join(__location__, "signal-data")):
    voltage = NumpySerializer(os.path.join(path, f_voltage)).deserialize()
    voltage_time = NumpySerializer(os.path.join(path, f_time)).deserialize()
    return voltage, voltage_time

def load_light_pattern_test(path_light_signal, path_light_signal_time, concatenate=True):
    light_signal, light_signal_time = __load_data(path_light_signal, path_light_signal_time)
    if concatenate:
        return __concatenate_light_signals(light_signal, light_signal_time)
    else:
        return light_signal, light_signal_time
    
def join_data(ser_voltage, ser_voltage_time):
    num_array = len(ser_voltage[0])
    num_elements = num_array * len(ser_voltage)
    voltage = numpy.empty(num_elements, dtype=numpy.int)
    voltage_time = numpy.empty(num_elements, dtype=numpy.int64)
    start = 0
    end = num_array
    for volt, volt_time in zip(ser_voltage, ser_voltage_time):
        voltage[start:end] = volt
        voltage_time[start:end] = volt_time
        start += num_array
        end += num_array
    return voltage, voltage_time

def periodogram(ser_voltage, sampling_time=2e-5):
    # f=1/s, sampling every 20 ns
    freq = 1.0 / sampling_time
    x, y = scipy.signal.periodogram(ser_voltage[0], freq)
    #x, y = scipy.signal.welch(ser_voltage[0], freq)
    max_idx = numpy.where(y > 100)
    print(1.0 / x[max_idx])
    plt.plot(1.0/x, y)
    plt.show()

# http://www.cbcity.de/die-fft-mit-python-einfach-erklaert
# https://stackoverflow.com/questions/11205037/detect-period-of-unknown-source/11210226#11210226
def fft(ser_voltage, ser_voltage_time, plot=False):
    data = ser_voltage[0]
    time = ser_voltage_time[0]
    
    # test data: on 10.000 > 0.01 s, off 5.000 > 0.005 s
    #hann = np.hanning(len(data)) # sampling window, if signal is not fully periodic
    #fft = np.fft.fft(hann * data)
    fft = numpy.fft.fft(data)
    N = len(fft) // 2 # only half fft signal due to signal mirroring
    fft = 2.0 * numpy.abs(fft[:N]) / N # take real value and adjust value range (2/N)
    
    dt = time[1] - time[0]
    dt = (dt * 1000) / 1000000000 # convert from relative time to ns to s
    fa = 1.0 / dt # f=1/s, s=1/f
    X = numpy.linspace(0, fa/2, N, endpoint=True)[1:]
    
    fft = fft[1:]
    Xp = 1.0 / X # convert from frequency to period, reciprocal of sampling rate
    #freq = numpy.fft.fftfreq(N, d=timstep)
    #Xp = 1.0/freq[1:]
    
    threshold = 50
    idx_max = numpy.where(fft > threshold)[0]
    print("Frequency peaks")
    print("Time (s): ", Xp[idx_max])
    print("Frequency: ", fft[idx_max])
    print("Time ratio: ", Xp[idx_max].max() / Xp[idx_max].min())
    if plot:
        _, axes = plt.subplots(2)
        axes[0].plot(data)
        axes[1].plot(Xp, fft)
        axes[1].scatter(Xp[idx_max], fft[idx_max], color="red")
        plt.ylabel('Amplitude ($Unit$)')
        plt.xlabel('Period ($s$)')
        plt.xticks(numpy.arange(min(Xp), 0.31, 0.01))
        plt.show()

def get_sequence(voltage, voltage_time, sliding_window=True, window_ratio=0.1):
    # For larger data frame of voltages, compute threshold per window more robust
    # Threshold mid range most robust against outliers over large data frames
    if sliding_window:
        window_size = int(window_ratio * len(voltage))
        size = stepsize = window_size
        voltage_window = sliding_window_array(voltage, size, stepsize)
        voltage_on_off_sequence = numpy.apply_along_axis(
            smoothing.simple_threshold, -1, voltage_window).ravel()
    else:
        voltage_on_off_sequence = smoothing.simple_threshold(voltage)
    volt_changes = numpy.diff(voltage_on_off_sequence)
    volt_changes_idx = numpy.where(numpy.logical_or(volt_changes==1, volt_changes==-1))[0]
    volt_changes_idx = numpy.hstack([0, volt_changes_idx])
    as_strided = numpy.lib.stride_tricks.as_strided
    on_off_view = as_strided(volt_changes_idx,
                             (volt_changes_idx.shape[0]-1,2),
                             volt_changes_idx.strides*2)
    duration = numpy.diff(voltage_time[on_off_view]).ravel()
    volt_on_off = voltage_on_off_sequence[volt_changes_idx[1:]]
    return volt_on_off, duration, volt_changes_idx # [1:] without zero

def get_unique_sequence_parts(volt_on_off, duration):
    join_values = list(zip(volt_on_off, duration))
    join_values = numpy.array(join_values, dtype=numpy.dtype('int,int'))
    unique, counts = numpy.unique(join_values, return_counts=True)
    value_view = numpy.array(unique.tolist())
    return numpy.c_[value_view, counts]

def merge_signal(signal_parts, tolerance_similarity=0.05):
    result = scipy.spatial.distance.cdist(signal_parts[:,1].reshape(signal_parts[:,1].shape[0],-1),
                                          signal_parts[:,1].reshape(signal_parts[:,1].shape[0],-1),
                                          lambda u, v: numpy.allclose(u, v, rtol=tolerance_similarity))
    numpy.fill_diagonal(result, 0)
    merge_row_idx = numpy.where(result==1)[0]
    if len(merge_row_idx) > 0 and len(merge_row_idx) % 2 == 0:    
        inverse_mask = numpy.ones(len(signal_parts), numpy.bool)
        inverse_mask[merge_row_idx] = 0
        no_signal_merge = signal_parts[inverse_mask]
        no_signal_merge = no_signal_merge[:,[0,1]]
        merge_row_idx = merge_row_idx.reshape(-1, 2)
        signal_merge = signal_parts[merge_row_idx]
        duration_seq = numpy.floor(numpy.sum(signal_merge[:,:,1] * signal_merge[:,:,2], axis=1) /
                                   numpy.sum(signal_merge[:,:,2], axis=1)).astype(numpy.int)
        state = signal_merge[:,0][:,0]
        signal_merge = numpy.c_[state, duration_seq]
        sequence = numpy.r_[no_signal_merge, signal_merge]
    else:
        sequence = signal_parts[:,[0,1]]
    return sequence

def split_cycles_by_idx(voltage, idx_local_minima):
    return numpy.split(voltage, idx_local_minima)[1:-1]

def split_cycles_by_view(voltage, idx_local_minima):
    as_strided = numpy.lib.stride_tricks.as_strided
    cycle_view = as_strided(idx_local_minima, (idx_local_minima.shape[0]-1,2), idx_local_minima.strides*2)
    cycles = [voltage[start:stop] for start, stop in cycle_view]
    return cycles

def detect_cycle_by_sequence(
        ser_voltage, ser_voltage_time, mean_threshold=0.8, mult_std_diff=4, plot=False, timing=False):
    
    # Single data frame for binary signal or per window calculation
    if len(ser_voltage.shape) > 1:
        ser_voltage, ser_voltage_time = join_data(ser_voltage, ser_voltage_time) 
    else:
        ser_voltage_time = ser_voltage_time.astype(numpy.int64)
    
    volt_on_off, duration, volt_on_off_idx = get_sequence(ser_voltage, ser_voltage_time)
    
    signal_parts = get_unique_sequence_parts(volt_on_off, duration)
    mean_occurrence = signal_parts[:,2].mean() * mean_threshold
    signal_parts = signal_parts[signal_parts[:,2] > mean_occurrence]
    
    off_parts_idx = numpy.where(signal_parts[:,0]==0)
    off_parts = signal_parts[off_parts_idx]
    off_parts = merge_signal(off_parts)
    
    on_parts_idx = numpy.where(signal_parts[:,0]==1)
    on_parts = signal_parts[on_parts_idx]
    on_parts = merge_signal(on_parts)
    
    sequence = numpy.vstack((off_parts, on_parts))
    sequence = sequence[:,1]
    sequence_len = len(sequence)
    
    # With sequence length, identify on off patterns, start with on
    on_off_view = sliding_window_array(volt_on_off, sequence_len)
    duration_view = sliding_window_array(duration, sequence_len)
    # sequence_len+1 important to include last phase, or overlapping 
    on_off_idx_view = sliding_window_array(volt_on_off_idx, sequence_len+1)
    
    target_sequence_idx = numpy.where(on_off_view[:,0] == 1)
    duration_view = duration_view[target_sequence_idx]
    on_off_idx_view = on_off_idx_view[target_sequence_idx]
    
    sequence = sequence.reshape(-1, sequence.shape[0])
    result = scipy.spatial.distance.cdist(duration_view, sequence, lambda u, v: abs(u.mean() - v.mean()))
    selection = (abs(result - result.mean()) < mult_std_diff * result.std()).ravel()
    
    duration_candidates = duration_view[selection]
    cycles_on_off = on_off_idx_view[selection]
    cycles_on_off = numpy.c_[cycles_on_off[:,0], cycles_on_off[:,-1]]
    # diverse length of voltage, use list as container
    voltage_candidates = [ser_voltage[range_idx[0]:range_idx[1]] for range_idx in cycles_on_off]
    if timing:
        voltage_time_candidates = [ser_voltage_time[range_idx[0]:range_idx[1]] for range_idx in cycles_on_off]
    
    if plot:
        plot_elements = (sequence_len//2) * 2
        _, axes = plt.subplots(plot_elements)
        for i in range(plot_elements):
            axes[i].plot(voltage_candidates[i])
        plt.show()
    
    if timing:
        return duration_candidates, voltage_candidates, voltage_time_candidates
    else:
        return duration_candidates, voltage_candidates

def detect_cycle_by_min_correlation(voltage, multiple_autocorrelation=False):
    freqs = numpy.fft.rfft(voltage)
    if multiple_autocorrelation:
        auto1 = freqs * numpy.conj(freqs)
        auto2 = auto1 * numpy.conj(auto1)
        autocorr = numpy.fft.irfft(auto2)
    else:
        autocorr = numpy.fft.irfft(freqs * numpy.conj(freqs))
    idx_local_minima = scipy.signal.argrelmin(autocorr)[0]
    print(split_cycles_by_idx(voltage, idx_local_minima))
    plot_voltage_cycles(voltage, idx_local_minima, autocorr)

# https://dsp.stackexchange.com/questions/386/autocorrelation-in-audio-analysis
def detect_cycle_by_max_min_correlation(voltage, filtering=True, multiple_autocorrelation=False):
    # filter noisy data for improved local minima via sliding window or
    # apply two times auto-correlation without using local minima via sliding window
    if filtering:
        window_len = int(numpy.floor(len(voltage) * 0.2))
        print(window_len)
        if window_len % 2 == 0:
            window_len += 1
        voltage = scipy.signal.savgol_filter(voltage, window_len, 1)
    freqs = numpy.fft.rfft(voltage)
    if multiple_autocorrelation:
        auto1 = freqs * numpy.conj(freqs)
        auto2 = auto1 * numpy.conj(auto1)
        autocorr = numpy.fft.irfft(auto2)
    else:
        autocorr = numpy.fft.irfft(freqs * numpy.conj(freqs))
    
    idx_local_maxima = scipy.signal.argrelmax(autocorr)[0]
    mean_distance_local_maxima = int(numpy.floor(numpy.sum(numpy.diff(idx_local_maxima[:-1])) /
                                                 len(idx_local_maxima)-1))
    #mean_distance_local_maxima = int(numpy.diff(idx_local_maxima).mean())
    start = idx_local_maxima
    stop = idx_local_maxima + mean_distance_local_maxima
    idx_local_minima = [numpy.argmin(voltage[r1:r2]) for r1, r2 in zip(start, stop)]
    # adapt indices of local minimum to global data frame
    idx_local_minima += idx_local_maxima
    
    cycles = split_cycles_by_idx(voltage, idx_local_minima)
    print(cycles)
    cycle_len = [cycle.shape[0] for cycle in cycles]
    num_resample = min(cycle_len)
    if num_resample > 0:
        cycles = [scipy.signal.resample(cycle, num_resample) for cycle in cycles]
    plot_voltage_cycles(voltage, idx_local_minima, autocorr)
    
def plot_voltage_cycles(voltage, idx_local_minima, autocorr):
    _, axes = plt.subplots(2)
    axes[0].plot(voltage)
    for minimum in idx_local_minima:
        axes[0].axvline(minimum, color="green")
    axes[1].plot(autocorr)
    plt.show()

def cycles(patterns, pattern_dist=None):
    for pattern in patterns:
        pattern_len = len(pattern)
        if pattern_dist:
            ser_voltage, ser_voltage_time = load_light_pattern_test(
                "voltage_%d_pattern_dist_%d" % (pattern_len, pattern_dist),
                "time_%d_pattern_dist_%d" % (pattern_len, pattern_dist))
        else:
            ser_voltage, ser_voltage_time = load_light_pattern_test(
                "voltage_%d_pattern" % pattern_len,
                "time_%d_pattern" % pattern_len)
            
        
            
        duration_candidates, _ = detect_cycle_by_sequence(ser_voltage, ser_voltage_time, plot=False)
        print(pattern_len)
        print(numpy.array(pattern) // 1000)
        print(len(duration_candidates))
        print(len(duration_candidates[0]))
        print(duration_candidates)
        print("---")
    
def testing_cycles():
    print("without pattern distance")
    patterns = [[2154557, 3812485],
                [4522300, 1292425, 4423046, 4333883],
                [2359963,3573843,6033328,5444867,2516832,7802597],
                [5709660,7919600,5757815,1385375,6440888,1196877,5746486,1501844],
                [2067857,6899412,1958155,2278422,7073935,7307621,3257578,1118105,6392651,5437882]]
    cycles(patterns)
    
    print("with pattern distance")
    patterns = [[4359176,3454139],
                [1486229,4424049,3645294,3089004],
                [1605289,3454733,4299809,2056194,1267983,1446395],
                [3804509,3821279,4357781,4917474,1478387,3346444,2018115,1492463],
                [3232186,2815303,1391152,2091222,3645319,1098794,4938733,3542548,1633644,2399618]]
    pattern_dist = 10
    cycles(patterns, pattern_dist)
    
if __name__ == "__main__":
    testing_cycles()
    