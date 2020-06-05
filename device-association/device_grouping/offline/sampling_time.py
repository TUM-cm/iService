import os
import numpy
import subprocess
import matplotlib.pyplot as plt
import coupling.utils.misc as misc
from collections import OrderedDict
from collections import defaultdict
from utils.serializer import DillSerializer
import coupling.light_grouping_pattern.light_analysis as light_analysis
from coupling.device_grouping.online.static.coupling_data_provider import CouplingDataProvider

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

'''
clock_interval = sensing_interval / 1000; (ns > us)
rtdm_timer_start(&timer, 0, sensing_interval, RTDM_TIMERMODE_RELATIVE);

nanosecs_rel_t     interval
https://xenomai.org/documentation/xenomai-2.6/html/api/group__rtdmtimer.html#ga429ca4935762583edb6e1ebc955fe958

time unit: ns
sampling interval: 20.000 ns (ReceiveLightControl) > 20 us
'''

sampling_times_ms = {2: (36.21536375947993, 39.33066386782231),
                     4: (32.79744937993234, 42.25825208568204),
                     6: (33.827774647887345, 45.10369953051642),
                     8: (47.87238547886316, 73.06837783615957),
                     10: (56.16000988264367, 74.88001317685824)}

def get_pattern_max_sampling_period(conversion_ms_to_s=1e3): # return time in seconds
    return round(max([max(entry) for entry in sampling_times_ms.values()])/conversion_ms_to_s, 3)

def plot_sampling_times(
        path_light_pattern_data, path_sampling_time_signal_patterns, result_path, plot_format, pos_factor=1.02):
    
    def autolabel(ax, rects):
        if rects.errorbar is None:
            for rect in rects:
                height = rect.get_height()
                ax.text(rect.get_x() + rect.get_width()/2., pos_factor*height, "%.1f" % height, ha="center", va="bottom")        
        else:
            _, _, barlinecols = rects.errorbar
            for err_segment, rect in zip(barlinecols[0].get_segments(), rects):
                height = err_segment[1][1]  # Use height of error bar
                value = rect.get_height()
                ax.text(rect.get_x() + rect.get_width() / 2, pos_factor*height, "%.1f" % value, ha='center', va='bottom')
    
    light_pattern_data = DillSerializer(path_light_pattern_data).deserialize()
    duration_signal_pattern = {key: value.signal_duration for key, value in light_pattern_data.items()}
    sampling_time_signal_pattern = DillSerializer(path_sampling_time_signal_patterns).deserialize()
    
    print("Average duration of sampling patterns (ms)")
    temp_signal_duration = {key: round(value, 2) for key, value in duration_signal_pattern.items()}
    print(temp_signal_duration)
    print("Sampling time per light pattern (ms)")
    temp_mean_sampling_time = {key: round(numpy.mean(value), 2) for key, value in sampling_time_signal_pattern.items()}
    print("mean:", temp_mean_sampling_time)
    print("std:", {key: round(numpy.std(value), 2) for key, value in sampling_time_signal_pattern.items()})
    ratio = {length_signal_pattern: round(temp_mean_sampling_time[length_signal_pattern]/temp_signal_duration[length_signal_pattern], 2) for length_signal_pattern in temp_signal_duration}
    print("ratio:", ratio)
    print("mean ratio:", round(numpy.mean(list(ratio.values())),2))
    
    print("Sampling range (ms)")
    print([(len_light_pattern, numpy.min(sampling_times), numpy.max(sampling_times))
           for len_light_pattern, sampling_times in sampling_time_signal_pattern.items()])
    
    mean_sampling_time =  [numpy.mean(entry) for entry in sampling_time_signal_pattern.values()]
    std_sampling_time = [numpy.std(entry) for entry in sampling_time_signal_pattern.values()]
    
    width = 0.35
    x = numpy.arange(len(duration_signal_pattern))
    fig, ax = plt.subplots()
    rects1 = ax.bar(x, duration_signal_pattern.values(), width, fill=False, edgecolor="black",
                    hatch="///", error_kw=dict(capsize=5, capthick=3, lw=3))
    rects2 = ax.bar(x + width, mean_sampling_time, width, yerr=std_sampling_time, fill=False,
                    edgecolor="black", hatch="xxx", error_kw=dict(capsize=5, capthick=3, lw=3))
    ax.set_xticks(x + width/2)
    ax.set_xticklabels(duration_signal_pattern.keys())
    ax.set_xlabel("Length of light pattern")
    ax.set_ylabel("Period (ms)")
    ax.legend((rects1[0], rects2[0]), ("Period of light pattern", "Sampling period of light pattern"),
              bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
    ax.grid(True)
    autolabel(ax, rects1)
    autolabel(ax, rects2)
    
    ax.set_ylim(top=numpy.max(mean_sampling_time+std_sampling_time)*1.25)
    fig.set_figwidth(2.2*fig.get_figwidth())
    #plt.show()
    filename = "sampling-time-light-pattern." + plot_format
    filepath = os.path.join(result_path, filename)
    plt.savefig(filepath, format=plot_format, bbox_inches='tight')
    plt.close(fig)

def sampling_time_light_patterns(
        path_light_pattern_data, path_sampling_time_signal_patterns,
        conversion_ms_to_s, test_rounds=10, scaling_factor=0.05):
    
    light_pattern_data = DillSerializer(path_light_pattern_data).deserialize()
    sampling_times = dict()
    light_pattern_lengths = dict()
    duration_signal_len = {key: value.signal_duration for key, value in light_pattern_data.items()}
    for test_round in range(test_rounds):
        print("round: ", test_round+1)
        sampling_time = dict()
        light_pattern_length = dict()
        for len_light_pattern, min_sampling_duration in duration_signal_len.items():
            print("len light pattern: ", len_light_pattern)
            min_sampling_duration = min_sampling_duration / conversion_ms_to_s
            sampling_duration = min_sampling_duration
            base_sampling_duration = min_sampling_duration
            sampling_time_too_small = True
            while sampling_time_too_small:
                light_pattern, light_pattern_time = light_analysis.load_light_pattern(len_light_pattern)
                coupling_data_provider = CouplingDataProvider(light_pattern, light_pattern_time, None, None)
                voltage, voltage_time = coupling_data_provider.get_light_data(sampling_duration)
                try:
                    light_pattern_duration, raw_light_pattern = light_analysis.detect_cycle_by_sequence(voltage, voltage_time)
                    if misc.valid_light_pattern(light_pattern_duration, len_light_pattern):
                        assert voltage.shape[0] == voltage_time.shape[0]
                        length_signal_patterns = list(map(len, raw_light_pattern))
                        # to select a single light pattern as length references makes no sense, because we don't know
                        # where the light pattern begins and thereby we have to compare multiple light patterns between
                        # light bulb and mobile device: sum(signal patterns) or compare the raw voltage signal
                        #voltage_signal_length = numpy.max(length_signal_patterns)
                        voltage_signal_length = sum(length_signal_patterns)
                        voltage_signal_length = voltage.shape[0]
                        sampling_time_too_small = False
                    else:
                        sampling_duration += scaling_factor * base_sampling_duration
                except ValueError:
                    sampling_duration += scaling_factor * base_sampling_duration
                    pass
            light_pattern_length[len_light_pattern] = voltage_signal_length
            sampling_time[len_light_pattern] = sampling_duration * conversion_ms_to_s # convert s to ms
        sampling_times[test_round] = sampling_time
        light_pattern_lengths[test_round] = light_pattern_length
    
    # resort after length of signal pattern, append from multiple runs
    sampling_time_signal_pattern = defaultdict(list)
    length_light_patterns = defaultdict(list)
    for results_sampling_time, results_signal_pattern_length in zip(sampling_times.values(), light_pattern_lengths.values()):
        for light_pattern in results_sampling_time.keys():
            sampling_time_signal_pattern[light_pattern].append(results_sampling_time[light_pattern])
            length_light_patterns[light_pattern].append(results_signal_pattern_length[light_pattern])
    # sort after length of light pattern
    sampling_time_signal_pattern = OrderedDict(sorted(sampling_time_signal_pattern.items()))
    DillSerializer(path_sampling_time_signal_patterns).serialize(sampling_time_signal_pattern)

def convert_from_svg_to_emf(path_svg_file, inkscape_path="C://Program Files//Inkscape//inkscape.exe"):
    path_emf_file = path_svg_file[:path_svg_file.rfind(".")+1] + "emf"
    subprocess.call([inkscape_path, path_svg_file, "--export-emf", path_emf_file])
    
def plot_light_signals(path_light_pattern_data, conversion_us_to_ms, result_path, plot_format, plot_signal=0):
    print("plot light signals")
    light_patterns = DillSerializer(path_light_pattern_data).deserialize()
    fig, axarr = plt.subplots(len(light_patterns), sharex=True)
    second_axarr = list()
    for i, len_light_pattern in enumerate(light_patterns):
        light_pattern = light_patterns[len_light_pattern]
        signal = light_pattern.signals[plot_signal]
        time_signal = light_pattern.time_signals[plot_signal]
        relative_time_ms = (time_signal - time_signal[0]) / conversion_us_to_ms
        print("len light pattern:", len_light_pattern)
        print("Relative time (ms):", relative_time_ms[-1])
        axarr[i].plot(relative_time_ms, signal)
        axarr[i].set_yticks([min(signal), max(signal)])
        righty = axarr[i].twinx()
        righty.set_yticks([0.5])
        righty.set_yticklabels([len_light_pattern])
        righty.tick_params(axis="both", which="both", length=0)
        second_axarr.append(righty)
    axarr[-1].set_xlabel("Signal time (ms)")
    axarr[-1].set_xticks(numpy.arange(relative_time_ms[-1], step=5))
    ax = fig.add_subplot(111, frameon=False)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_ylabel("Voltage signal (mV)", labelpad=60)
    ax2 = ax.twinx()
    ax2.set_yticks([])
    ax2.set_xticks([])
    ax2.set_ylabel("Length of light pattern", labelpad=50)
    fig.subplots_adjust(hspace=0.7)
    plt.savefig(os.path.join(result_path, "random-light-pattern." + plot_format), format=plot_format, bbox_inches='tight')
    
    # clear plot to keep only the signal patterns
    for ax in axarr:
        ax.axis("off")
    for ax in second_axarr:
        ax.axis("off")
    ax.axis("off")
    ax2.axis("off")
    
    svgfile = os.path.join(result_path, "random-light-pattern.svg")
    plt.savefig(svgfile, format="svg", bbox_inches='tight')
    convert_from_svg_to_emf(svgfile)
    #plt.show()
    plt.close(fig)
    
class LightPatternData:
    
    def __init__(self, signals, time_signals, signal_duration):
        self.signals = signals
        self.time_signals = time_signals
        self.signal_duration = signal_duration
    
def timing_light_patterns(path_light_pattern_data, conversion_us_to_ms):
    light_pattern_data = dict()
    for len_light_pattern in range(2, 11, 2):
        # Use all data to detect light pattern
        light_signal, light_signal_time = light_analysis.load_light_pattern(len_light_pattern)
        _, light_patterns, light_pattern_times = light_analysis.detect_cycle_by_sequence(
            light_signal, light_signal_time, timing=True)
        signal_duration = [(time[-1] - time[0]) for time in light_pattern_times if time[-1] > time[0]]
        signal_duration = numpy.mean([duration / conversion_us_to_ms for duration in signal_duration])
        light_pattern_data[len_light_pattern] = LightPatternData(light_patterns, light_pattern_times, signal_duration)
    DillSerializer(path_light_pattern_data).serialize(light_pattern_data)
    
def create_dir(dirpath):
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)
    
def main():
    plot_format = "pdf"
    conversion_us_to_ms = 1e3
    conversion_ms_to_s = 1e3
    
    raw_result_path = os.path.join(__location__, "raw-results", "sampling-time")
    result_path = os.path.join(__location__, "results", "sampling-time")
    
    path_light_pattern_data = os.path.join(raw_result_path, "light-pattern-data")
    path_sampling_time_signal_patterns = os.path.join(raw_result_path, "sampling-time-light-patterns")
    
    create_dir(raw_result_path)
    create_dir(result_path)
    
    if not os.path.exists(path_light_pattern_data):
        timing_light_patterns(path_light_pattern_data, conversion_us_to_ms)
    
    if not os.path.exists(path_sampling_time_signal_patterns):
        sampling_time_light_patterns(
            path_light_pattern_data, path_sampling_time_signal_patterns, conversion_ms_to_s)
    
    plot_light_signals(
        path_light_pattern_data, conversion_us_to_ms, result_path, plot_format)
    plot_sampling_times(
        path_light_pattern_data, path_sampling_time_signal_patterns, result_path, plot_format)
    
if __name__ == '__main__':
    main()
    