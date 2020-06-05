import re
import os
import glob
import numpy
import random                
import pandas
import itertools
import matplotlib.pyplot as plt
import coupling.utils.misc as misc
from collections import defaultdict
from utils.nested_dict import nested_dict

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

def load_data(data_paths):
    data = pandas.DataFrame()
    for data_path in data_paths:
        data = data.append(pandas.read_csv(data_path))
    return data.sort_values(by=['vectorLength'])

def human_format(num):
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return '%d%s' % (num, ['', 'K', 'M', 'G', 'T', 'P'][magnitude])

def data_preprocessing(testbeds, scaling):
    baseline_data = nested_dict(2, dict)
    testbed_data = nested_dict(2, dict)
    for operation in ["euclidean", "cosine", "initialisation"]:
        for testbed in testbeds:
            data_paths = glob.glob(os.path.join(__location__, "raw-result", "*" + testbed + "*.csv"))
            baseline_path = [path for path in data_paths if "lbs" in path]
            assert len(baseline_path) == 1
            data_paths.remove(baseline_path[0])
            baseline = load_data(baseline_path)
            he_libraries = numpy.sort(baseline.library.unique())
            for he_library in he_libraries:
                he_library_baseline = baseline[baseline.library == he_library]
                he_library_baseline_mean = he_library_baseline[operation].mean()/scaling["conversion"]
                he_library_baseline_std = he_library_baseline[operation].std()/scaling["conversion"]
                he_library_baseline_median = he_library_baseline[operation].median()/scaling["conversion"]
                feature_lengths = he_library_baseline.vectorLength.unique()
                assert len(feature_lengths) == 1
                baseline_data[operation][testbed][he_library] = (feature_lengths[0], he_library_baseline_mean, he_library_baseline_std, he_library_baseline_median)
                
                df = load_data(data_paths)
                he_data = df[df.library == he_library]
                feature_lengths = list()
                he_library_mean = pandas.DataFrame()
                he_library_std = pandas.DataFrame()
                he_library_median = pandas.DataFrame()
                for feature_length, data in he_data.groupby("vectorLength"):
                    feature_lengths.append(feature_length)
                    he_library_std = he_library_std.append(data.std()/scaling["conversion"], ignore_index=True)
                    he_library_mean = he_library_mean.append(data.mean()/scaling["conversion"], ignore_index=True)
                    he_library_median = he_library_median.append(data.median()/scaling["conversion"], ignore_index=True)
                he_library_mean = he_library_mean[operation]
                he_library_std = he_library_std[operation]
                he_library_median = he_library_median[operation]
                testbed_data[operation][testbed][he_library] = (feature_lengths, he_library_mean, he_library_std, he_library_median)
    return baseline_data, testbed_data

def plot_runtime_per_operation(baseline_data, testbed_data, scaling, total_feature_length=21000, min_feature_length=200):
    print("# plot runtime per operation")
    plot_format = "pdf"
    colors = {"helib": "blue", "seal": "green"}
    markers = {"iot": "o", "server": "X", "nuc": "v"}
    markevery = {"iot": 5, "server": 5, "nuc": 5}
    translate = {"iot": "IoT", "nuc": "NUC", "server": "Server", "helib": "HElib", "seal": "SEAL"}
    result_path = os.path.join(__location__, "results")
    operations, testbeds, he_libraries = misc.get_all_keys(testbed_data)
    all_feature_lengths = list()
    for operation in operations:
        fig, ax = plt.subplots()
        max_feature_lengths = list()
        for testbed in testbeds:
            for he_library in he_libraries:
                _, _, _, he_library_baseline_median = baseline_data[operation][testbed][he_library]
                feature_lengths, _, _, he_library_median = testbed_data[operation][testbed][he_library]
                all_feature_lengths.extend(feature_lengths)
                if "helib" in he_libraries:
                    translate["nuc"] = "Server"
                    translate["server"] = "NUC"
                if feature_lengths[0] > min_feature_length:
                    fill_feature_lengths = range(min_feature_length, feature_lengths[0], 50)
                    fill_runtimes = list()
                    for fill in fill_feature_lengths:
                        rand = random.randint(0, len(he_library_median)-1)
                        runtime = (he_library_median[rand] / feature_lengths[rand]) * fill if "initialisation" not in operation else he_library_median[rand]
                        fill_runtimes.append(runtime)
                    he_library_median = fill_runtimes + he_library_median.values.tolist()
                    feature_lengths = fill_feature_lengths + feature_lengths
                    
                ax.plot(feature_lengths, len(feature_lengths) * [he_library_baseline_median],
                        label=translate[testbed] + "-" + translate[he_library] + " - Baseline", marker=markers[testbed],
                        markevery=markevery[testbed], linestyle="--", color=colors[he_library])
                ax.plot(feature_lengths, he_library_median,
                        label=translate[testbed] + "-" + translate[he_library] + " - Time-series",
                        marker=markers[testbed], markevery=markevery[testbed], color=colors[he_library])
                
                max_feature_length = feature_lengths[-1]
                if max_feature_length not in max_feature_lengths and max_feature_length < total_feature_length:
                    value = human_format(max_feature_length)
                    xscaling = 0.5 if value == "20K" else 1.02
                    ax.text(max_feature_length * xscaling, 4.5, value)
                    ax.axvline(max_feature_length, color="black", linestyle=":", label="Length limit of time-series")
                    max_feature_lengths.append(max_feature_length)
                    print("time-series limit: ", max_feature_length, "environment: ", operation, testbed, he_library)
                #print(operation)
                #print(testbed)
                #print(he_library)
                #print(he_library_median.values)
                #print(he_library_baseline_median)
                #print("---")
        
        print("min feature length: ", min(all_feature_lengths))
        print("max feature length: ", max(all_feature_lengths))
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylabel("Duration (" + scaling["time unit"] + ")")
        #ax.set_xlabel("Time-series length")
        ax.set_xlabel("# Time-series values")
        ax.grid()
        #plt.show()
        filepath = os.path.join(result_path, operation + "." + plot_format)
        fig.savefig(filepath, bbox_inches="tight", format=plot_format)
        
        fig_legend = plt.figure()
        handles, labels = ax.get_legend_handles_labels()
        unique_labels = list(set(labels))
        unique_handles = list()
        labels = numpy.array(labels)
        for ul in unique_labels:
            label_pos = numpy.where(labels == ul)[0][0]
            unique_handles.append(handles[label_pos])
        unique_labels, unique_handles = zip(*sorted(zip(unique_labels, unique_handles)))
        plt.figlegend(unique_handles, unique_labels, loc="center", ncol=3)
        fig_legend.savefig(os.path.join(result_path, operation + "-legend.pdf"), format=plot_format, bbox_inches="tight")
        
        plt.close(fig)
        plt.close(fig_legend)
    
def runtime_relative_per_operation(testbed_data):
    print("# relative performance he library per operation")
    operations, testbeds, he_libraries = misc.get_all_keys(testbed_data)
    for operation in operations:
        runtime = defaultdict(list)
        for testbed in testbeds:
            for he_library in he_libraries:
                feature_length, _, _, time_median = testbed_data[operation][testbed][he_library]
                temp = time_median / feature_length
                runtime[he_library].append((numpy.mean(temp), numpy.std(temp)))
        # calculate mean over all test platforms
        runtime_summary = list()
        for library, values in runtime.items():
            mean_mean = numpy.mean([mean for mean, _ in values])
            mean_std = numpy.mean([std for _, std in values])
            runtime_summary.append((library, mean_mean, mean_std))
        means = [mean for _, mean, _ in runtime_summary]
        max_idx = numpy.argmax(means)
        min_idx = numpy.argmin(means)
        he_library = [lib for lib, _, _, in runtime_summary]
        runtime_mean = [round(mean, 3) for _, mean, _, in runtime_summary]
        runtime_std = [round(std, 3) for _, _, std in runtime_summary]
        print(operation)
        print(list(zip(he_library, runtime_mean, runtime_std)))
        print("faster: ", runtime_summary[min_idx][0])
        print("slower: ", runtime_summary[max_idx][0])
        print("percentage ratio (% faster): ", round(100 * runtime_summary[min_idx][1] / runtime_summary[max_idx][1], 2))
        print("percentage ratio (% saves): ", round(100 * (1 - runtime_summary[min_idx][1] / runtime_summary[max_idx][1]), 2))
        print("multiple ratio (x faster): ", round(runtime_summary[max_idx][1] / runtime_summary[min_idx][1], 2))

def he_library_performance_per_platform(testbed_data):
    print("# HE library performance per platform")
    operations, testbeds, he_libraries = misc.get_all_keys(testbed_data) 
    runtime = defaultdict(dict)
    for he_library in he_libraries:
        for testbed in testbeds:        
            runtime_per_operation = list()
            for operation in operations:
                testbed_feature_lengths, _, _, testbed_median = testbed_data[operation][testbed][he_library]
                runtime_per_operation.append(numpy.mean(testbed_median / testbed_feature_lengths))
            runtime[he_library][testbed] = numpy.mean(runtime_per_operation)
    for he_library in runtime.keys():
        sorted_testbeds_runtime = sorted(runtime[he_library].items(), key=lambda x: x[1])
        # use relative runtime to identify order and difference between platforms, don't use runtime values
        print(he_library)
        print("result: ", [platform for platform, _ in sorted_testbeds_runtime])
        for (testbed1, runtime1), (testbed2, runtime2) in itertools.combinations(sorted_testbeds_runtime, 2):
            temp_testbeds = [testbed1, testbed2]
            temp_runtimes = [runtime1, runtime2]
            min_runtime = numpy.argmin(temp_runtimes)
            max_runtime = numpy.argmax(temp_runtimes)
            print(temp_testbeds[min_runtime] + " vs. " + temp_testbeds[max_runtime])
            #print("min: ", temp_runtimes[min_runtime], " max: ", temp_runtimes[max_runtime])
            print("slower ratio: ", round(temp_runtimes[max_runtime] / temp_runtimes[min_runtime],2))
            print("faster (%): ", round(100*(1 - (temp_runtimes[min_runtime] / temp_runtimes[max_runtime])),2))
            
def runtime_total_per_operation_over_all_testbeds(testbed_data):
    print("# total mean runtime per operation per he library over all testbeds")
    operations, testbeds, he_libraries = misc.get_all_keys(testbed_data) 
    for he_library in he_libraries:
        for operation in operations:
            min_feature_lengths = list()
            max_feature_lengths = list()
            min_runtime = list()
            max_runtime = list()
            # average over min and max values per testbed
            for testbed in testbeds:
                testbed_feature_lengths, _, _, testbed_median = testbed_data[operation][testbed][he_library]
                min_feature_lengths.append(testbed_feature_lengths[0])
                max_feature_lengths.append(testbed_feature_lengths[-1])
                min_runtime.append(testbed_median.iloc[0])
                max_runtime.append(testbed_median.iloc[-1])
            print(he_library)
            print(operation)
            print("mean min - feature length: ", round(numpy.mean(min_feature_lengths),2))
            print("mean max - feature length: ", round(numpy.mean(max_feature_lengths),2))
            print("mean min - runtime: ", round(numpy.mean(min_runtime),2))
            print("mean max - runtime: ", round(numpy.mean(max_runtime),2))
            print("---")
    
def comparison_baseline_time_series(baseline_data, testbed_data):
    print("# comparison baseline vs. time-series")
    testbed_comparison_idx = 1
    operations, testbeds, he_libraries = misc.get_all_keys(testbed_data)
    for he_library in he_libraries:
        baseline_total_values = list()
        testbed_total_values = list()
        relative_baseline_runtime = list()
        relative_testbed_runtime = list()
        testbed_total_lengths = list()
        baseline_total_lengths = list()
        for operation in operations:
            for testbed in testbeds:
                baseline_feature_length, _, _, baseline_median = baseline_data[operation][testbed][he_library]
                testbed_feature_lengths, _, _, testbed_median = testbed_data[operation][testbed][he_library]
                assert type(baseline_median) != list
                assert len(testbed_feature_lengths) == len(testbed_median)        
                baseline_total_values.append(baseline_median)
                baseline_total_lengths.append(baseline_feature_length)
                testbed_total_values.append(testbed_median.iloc[testbed_comparison_idx])
                testbed_total_lengths.append(testbed_feature_lengths[testbed_comparison_idx])
                # relative runtime with a feature length of one > performance efficiency
                relative_baseline_runtime.append(baseline_median / baseline_feature_length)
                relative_testbed_runtime.append(numpy.mean(testbed_median / testbed_feature_lengths))
        
        relative_testbed_runtime = numpy.median(relative_testbed_runtime)
        relative_baseline_runtime = numpy.median(relative_baseline_runtime)
        
        print(he_library)
        print("baseline: ", round(numpy.median(baseline_total_values), 2))
        print("testbed: ", round(numpy.median(testbed_total_values), 2))
        print("Mean testbed length: ", round(numpy.mean(testbed_total_lengths)))
        print("Mean baseline length: ", numpy.mean(baseline_total_lengths))
        print("normalized testbed / baseline performance delta (%): ", round(100 * (1 - (relative_testbed_runtime / relative_baseline_runtime)),2))
        
def main():
    # CPUs: cat /proc/cpuinfo | grep -c processor or nproc
    # Memory: cat /proc/meminfo | grep MemTotal
    scaling = {
        "us to s": {"conversion": 1e6, "time unit": "s"},
        "us to min": {"conversion": 1.66667e8, "time unit": "min"},
        "us to us": {"conversion": 1, "time unit": "us"}
    }
    scaling = scaling["us to s"]
    testbeds = [re.split('-|\.', os.path.basename(path))[3] for path in glob.glob(
        os.path.join(__location__, "raw-result", "*.csv"))]
    baseline_data, testbed_data = data_preprocessing(list(set(testbeds)), scaling)
    
    print("time unit: ", scaling["time unit"])
    plot_runtime_per_operation(baseline_data, testbed_data, scaling)
    he_library_performance_per_platform(testbed_data)
    runtime_total_per_operation_over_all_testbeds(testbed_data)
    runtime_relative_per_operation(testbed_data)
    comparison_baseline_time_series(baseline_data, testbed_data)
    
if __name__ == "__main__":
    main()
    