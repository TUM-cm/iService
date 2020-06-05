from __future__ import division
import os
import time
import numpy
import random
import itertools
import matplotlib.pyplot as plt
import coupling.utils.misc as misc
from collections import defaultdict
from utils.nested_dict import nested_dict
from utils.serializer import DillSerializer
import coupling.utils.vector_similarity as vector_similarity
import coupling.light_grouping_pattern.light_analysis as light_analysis
from coupling.device_grouping.offline.sampling_time import sampling_times_ms
from coupling.device_grouping.online.static.coupling_data_provider import CouplingDataProvider
from sklearn.metrics.classification import accuracy_score, precision_score, recall_score, f1_score

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

'''
conda install rpy2
install R for Windows (newest version)
run method: install_dtw_package() in Dtw class
'''

class Client:
    
    def __init__(self, signal, signal_time, signal_noise):
        self.signal = signal
        self.signal_time = signal_time
        self.signal_noise = signal_noise
        
    def get_distorted_light_signal(self, distortion_rate):
        return (1-distortion_rate) * self.signal + distortion_rate * self.signal_noise

def get_light_signals(clients, unequal_signal_length=False):
    client_data = list()
    for client in clients:
        run = True
        while run:
            light_signal, light_signal_time = light_analysis.load_light_pattern(client)
            coupling_data_provider = CouplingDataProvider(light_signal, light_signal_time, None, None)
            sampling_range = sampling_times_ms[client]
            sampling_period = random.uniform(sampling_range[0], sampling_range[1])
            sampling_period = round(sampling_period / 1e3, 3)
            light_signal, light_signal_time = coupling_data_provider.get_light_data(sampling_period)
            mean = light_signal.mean()
            std = light_signal.std()
            datalen = len(light_signal)
            noise = numpy.random.normal(mean, std, datalen)
            if unequal_signal_length:
                len_light_signal = len(light_signal)
                signal_lengths = [len_light_signal != len(data.signal) for data in client_data]
                if signal_lengths.count(True) == len(signal_lengths):
                    client_data.append(Client(light_signal, light_signal_time, noise))
                    run = False
            else:
                client_data.append(Client(light_signal, light_signal_time, noise))
                run = False
    return client_data

def evaluate_impact_signal_distortion(
        len_light_patterns, distortion_rates, path_distorted_light_signals, path_distortion_similarity, rounds):
    
    distorted_light_signals = defaultdict(list)
    results_distortion_similarity = nested_dict(3, list)
    for run in range(rounds):
        print("round: ", run)
        for len_light_pattern in len_light_patterns:
            print("len light pattern:", len_light_pattern)            
            equalize_method = "dummy"
            client = get_light_signals([len_light_pattern])[0]
            distorted_light_signals[len_light_pattern].append(client)
            for distortion_rate in distortion_rates:
                print("distortion rate: ", distortion_rate)
                for similarity_method in vector_similarity.similarity_methods:
                    distorted_light_signal = client.get_distorted_light_signal(distortion_rate)
                    similarity = similarity_method(client.signal, distorted_light_signal, equalize_method)                    
                    if distortion_rate == 0:
                        assert numpy.array_equal(client.signal, distorted_light_signal)
                        assert similarity >= 0.98
                    results_distortion_similarity[len_light_pattern][distortion_rate][similarity_method.__name__].append(similarity)
    DillSerializer(path_distortion_similarity).serialize(results_distortion_similarity)
    DillSerializer(path_distorted_light_signals).serialize(distorted_light_signals)

def evaluate_similarity_runtime(len_light_patterns, path_similarity, path_runtime, rounds):
    results_runtime = nested_dict(4, list)
    results_similarity = nested_dict(4, list)
    same = list(zip(len_light_patterns, len_light_patterns))
    combined = list(itertools.combinations(len_light_patterns, 2))
    pattern_conbination = same + combined
    for len_light_pattern1, len_light_pattern2 in pattern_conbination:
        print("from-to:", len_light_pattern1, len_light_pattern2)
        for run in range(rounds):
            print("round: ", run)
            client1, client2 = get_light_signals([len_light_pattern1, len_light_pattern2])
            for equalize_method in [vector_similarity.equalize_methods.fill,
                                    vector_similarity.equalize_methods.cut,
                                    vector_similarity.equalize_methods.dtw]:
                print("equalize:", equalize_method)
                for similarity_method in vector_similarity.similarity_methods:
                    print("similarity:", similarity_method.__name__)
                    start_time = time.time()
                    similarity = similarity_method(client1.signal, client2.signal, equalize_method)
                    elapsed_time = time.time() - start_time
                    assert elapsed_time > 0
                    results_similarity[len_light_pattern1][len_light_pattern2][equalize_method][similarity_method.__name__].append(
                        similarity)
                    results_runtime[len_light_pattern1][len_light_pattern2][equalize_method][similarity_method.__name__].append(
                        elapsed_time)
    DillSerializer(path_similarity).serialize(results_similarity)
    DillSerializer(path_runtime).serialize(results_runtime)
    
def plot_distorted_light_signals(
        distortion_rates, path_light_signals, conversion_us_to_ms, result_path, plot_format, plot_round=0):
    
    print("plot distorted light signals")
    light_signals = DillSerializer(path_light_signals).deserialize()
    len_light_patterns = misc.get_all_keys(light_signals)[0]
    for len_light_pattern in len_light_patterns:
        print("len light pattern:", len_light_pattern)
        fig, axarr = plt.subplots(len(distortion_rates))
        for i, distortion_rate in enumerate(distortion_rates):
            client = light_signals[len_light_pattern][plot_round]
            light_signal = client.get_distorted_light_signal(distortion_rate)
            light_signal_time = client.signal_time
            relative_time_ms = (light_signal_time-light_signal_time[0]) / conversion_us_to_ms
            axarr[i].plot(relative_time_ms, light_signal)
            xticks = [] if i+1 < len(distortion_rates) else numpy.arange(relative_time_ms[-1], step=10)
            axarr[i].set_xticks(xticks)
            #axarr[i].set_yticks([round(numpy.mean(light_signal))])
            axarr[i].yaxis.tick_right()
            axis = "both" if i+1 < len(distortion_rates) else "y"
            axarr[i].tick_params(axis=axis, which='both', length=0)
            axarr[i].set_yticks([numpy.mean(light_signal)])
            axarr[i].set_yticklabels([distortion_rate])
        
        axarr[-1].set_xlabel("Signal time (ms)")
        ax = fig.add_subplot(111, frameon=False)
        ax.set_yticks([])
        ax.set_xticks([])
        ax.set_ylabel("Voltage signal (mV)")
        ax2 = ax.twinx()
        ax2.set_yticks([])
        ax2.set_xticks([])
        ax2.set_ylabel("Distortion rate", labelpad=50)
        
        filename = "distortion-rate-signal-len-" + str(len_light_pattern) + "." + plot_format
        filepath = os.path.join(result_path, filename)
        fig.savefig(filepath, format=plot_format, bbox_inches="tight")
        #plt.show()
        plt.close(fig)
    
def get_runtime(result_path):
    
    def median(runtimes):
        return {key: numpy.median(values) for key, values in runtimes.items()}
    
    def sort(runtimes):
        return sorted(runtimes.items(), key=lambda kv: kv[1])
    
    runtimes = DillSerializer(result_path).deserialize()
    runtime_equalize_methods = defaultdict(list)
    runtime_similarity_method = defaultdict(list)
    runtime_equalize_similarity_methods = defaultdict(list)
    len_light_patterns1, len_light_patterns2, equalize_methods, similarity_methods = misc.get_all_keys(runtimes)
    for len_light_pattern1 in len_light_patterns1:
        for len_light_pattern2 in len_light_patterns2:
            for equalize_method in equalize_methods:
                for similarity_method in similarity_methods:
                    runtime = runtimes[len_light_pattern1][len_light_pattern2][equalize_method][similarity_method]
                    if len(runtime) > 0:
                        median_runtime = numpy.median(runtime)
                        runtime_equalize_methods[equalize_method].append(median_runtime)
                        runtime_similarity_method[similarity_method].append(median_runtime)
                        key = equalize_method + ":" + similarity_method
                        runtime_equalize_similarity_methods[key].append(median_runtime)
    
    runtime_equalize_methods = sort(median(runtime_equalize_methods))
    runtime_similarity_method = sort(median(runtime_similarity_method))
    runtime_equalize_similarity_methods = sort(median(runtime_equalize_similarity_methods))
    return runtime_equalize_methods, runtime_similarity_method, runtime_equalize_similarity_methods

def client_runtime_analysis(result_path, nth_best, round_factor=6):
    runtime_equalize_methods, runtime_similarity_method, runtime_equalize_similarity_methods = get_runtime(result_path)
    print("Runtime (s) equalize methods")
    print(list(map(lambda x: x[0] + ":" + str(round(x[1], round_factor)), runtime_equalize_methods)))
    print("Runtime (s) similarity methods")
    print(list(map(lambda x: x[0] + ":" + str(round(x[1], round_factor)), runtime_similarity_method[:nth_best])))
    print("Runtime (s) combination equalize similarity method")
    print(list(map(lambda x: x[0] + ":" + str(round(x[1], round_factor)), runtime_equalize_similarity_methods[:nth_best])))
    
def distortion_similarity_analysis(path_distortion_similarity, result_path, plot_format):
    results = DillSerializer(path_distortion_similarity).deserialize()
    len_light_patterns, distortion_rates, similarity_methods = misc.get_all_keys(results)
    
    print("Similarity threshold by signal distortion")
    distortion_rate = 0.5
    for similarity_method in ["spearman", "pearson", "distance_correlation"]:
        similarity = list()
        for len_light_pattern in len_light_patterns:
            result = results[len_light_pattern][distortion_rate][similarity_method]
            similarity.extend(result)
        print(similarity_method, round(numpy.median(similarity), 2))
    
    fig, ax = plt.subplots()
    markers = itertools.cycle(misc.markers)
    colors = [plt.cm.tab20(i) for i in numpy.linspace(0, 1, len(vector_similarity.similarity_methods))]
    for i, similarity_method in enumerate(similarity_methods):
        distortion = list()
        similarity_mean = list()
        for distortion_rate in distortion_rates:
            mean = list()
            for len_light_pattern in len_light_patterns:
                result = results[len_light_pattern][distortion_rate][similarity_method]
                mean.append(numpy.mean(result))
            distortion.append(distortion_rate)
            similarity_mean.append(numpy.median(mean))
        label = similarity_method.replace("_", " ").capitalize().replace("Dtw", "DTW")
        ax.plot(distortion, similarity_mean, label=label, marker=next(markers), color=colors[i])
    ax.plot([0, 1], [1, 0], color="black", linestyle="--")
    ax.axvline(0.4, color="red", linestyle="--")
    ax.grid()
    ax.set_xticks(distortion_rates)
    ax.set_ylabel("Signal similarity")
    ax.set_xlabel("Distortion rate")
    ax.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=4, mode="expand", borderaxespad=0.)
    fig.set_figwidth(fig.get_figwidth()*2.5)
    filename = "distortion-signal-similarity." + plot_format
    fig.savefig(os.path.join(result_path, filename), plot_format=plot_format, bbox_inches="tight")
    #plt.show()
    plt.close(fig)

def client_similarity_analysis(path_client_similarity, path_runtimes, nth_best, result_path, plot_format):
    
    def adapt_ticklabels(labels):
        return [label.replace("_", " ").capitalize() for label in labels]
    
    def plot_raw_similarities(plot_data, similarity_methods, equalize_methods):
        similarities = [list(similarites.values()) for similarites in plot_data.values()]
        fig, ax = plt.subplots()
        im = ax.imshow(similarities, cmap="jet", vmin=0, vmax=1)
        ax.set_xticks(numpy.arange(len(equalize_methods)))
        ax.set_yticks(numpy.arange(len(similarity_methods)))
        ax.set_xticklabels(adapt_ticklabels(equalize_methods))
        ax.set_yticklabels(adapt_ticklabels(similarity_methods))
        for i in range(len(similarity_methods)):
            for j in range(len(equalize_methods)):
                ax.text(j, i, round(similarities[i][j], 2), ha="center", va="center")
        ax.set_ylabel("Similarity")
        ax.set_xlabel("Equalize")
        ax.figure.colorbar(im)
        filename = "raw-similarities." + plot_format
        fig.savefig(os.path.join(result_path, filename), format=plot_format, bbox_inches="tight")
        #plt.show()
        plt.close(fig)
    
    def find_best_similarity_equalize_threshold(total_similarity, path_runtimes, round_factor=2):
        print("Best similarity equalize threshold")
        total_similarity = sorted(total_similarity.items(), key=lambda kv: numpy.mean(kv[1]), reverse=True)
        _, _, runtime_equalize_similarity_methods = get_runtime(path_runtimes)
        runtime_equalize_similarity_methods = dict(runtime_equalize_similarity_methods)
        best_similarity = dict()
        for similarity, metrics in total_similarity[:nth_best]:
            similarity_method, equalize_method, _ = similarity.split(":")
            runtime = runtime_equalize_similarity_methods[equalize_method + ":" + similarity_method]
            weight = 0.8 * numpy.mean(metrics) + 0.2 * (1-runtime)
            best_similarity[similarity] = round(weight, round_factor)
            print("Similarity / metrics / runtime (s):", similarity, numpy.round(metrics, round_factor), round(runtime, 4))
        best_similarity = sorted(best_similarity.items(), key=lambda kv: kv[1], reverse=True)
        print("Weighted best results:", best_similarity)
    
    results = DillSerializer(path_client_similarity).deserialize()
    len_light_patterns1, len_light_patterns2, equalize_methods, similarity_methods = misc.get_all_keys(results)
    total_similarity = dict()
    plot_data = nested_dict(1, dict)
    for similarity_method in similarity_methods:
        for equalize_method in equalize_methods:
            y_true = list()
            similarities = list()
            for len_light_pattern1 in len_light_patterns1:
                for len_light_pattern2 in len_light_patterns2:
                    if len_light_pattern1 in results and len_light_pattern2 in results[len_light_pattern1]:
                        result = results[len_light_pattern1][len_light_pattern2][equalize_method][similarity_method]
                        similarities.extend(result)
                        y_true.extend(len(result) * [1 if len_light_pattern1 == len_light_pattern2 else 0])
            plot_data[similarity_method][equalize_method] = numpy.median(similarities)
            assert len(similarities) == len(y_true)
            y_true = numpy.asarray(y_true)
            similarities = numpy.asarray(similarities)
            similarity_thresholds = numpy.arange(1, step=0.1)
            for similarity_threshold in similarity_thresholds:
                similarity_threshold = round(similarity_threshold, 1)
                y_pred = numpy.zeros(len(y_true))
                y_pred[similarities >= similarity_threshold] = 1
                acc = accuracy_score(y_true, y_pred)
                prec = precision_score(y_true, y_pred)
                rec = recall_score(y_true, y_pred)
                f1 = f1_score(y_true, y_pred)
                key = similarity_method + ":" + equalize_method + ":" + str(similarity_threshold)
                total_similarity[key] = [acc, prec, rec, f1]
    
    find_best_similarity_equalize_threshold(total_similarity, path_runtimes)
    plot_raw_similarities(plot_data, similarity_methods, equalize_methods)
    
def create_dir(dirpath):
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)
    
def main():
    nth_best = 5
    test_rounds = 10
    plot_format = "pdf"
    conversion_to_ms = 1e3
    test_platform = "vm" # vm, server    
    len_light_patterns = range(2, 11, 2)
    distortion_rates = numpy.round(numpy.arange(0, 1.05, 0.1), 1)
    
    result_path = os.path.join(__location__, "results", "similarity-metrics")
    raw_result_path = os.path.join(__location__, "raw-results", "similarity-metrics", test_platform)
    create_dir(raw_result_path)
    create_dir(result_path)
    
    path_distorted_light_signals = os.path.join(raw_result_path, "distorted-light-signals")
    path_distortion_similarity = os.path.join(raw_result_path, "distortion-similarity")
    path_client_similarity = os.path.join(raw_result_path, "client-similarity")
    path_client_similarity_runtime = os.path.join(raw_result_path, "client-similarity-runtime")
    
    if not os.path.exists(path_client_similarity_runtime):
        evaluate_similarity_runtime(
            len_light_patterns, path_client_similarity, path_client_similarity_runtime, test_rounds)
    if not os.path.exists(path_distortion_similarity):
        evaluate_impact_signal_distortion(
            len_light_patterns, distortion_rates, path_distorted_light_signals, path_distortion_similarity, test_rounds)
    
    client_runtime_analysis(
        path_client_similarity_runtime, nth_best)
    client_similarity_analysis(
        path_client_similarity, path_client_similarity_runtime, nth_best, result_path, plot_format)
    distortion_similarity_analysis(
        path_distortion_similarity, result_path, plot_format)
    plot_distorted_light_signals(
        distortion_rates, path_distorted_light_signals, conversion_to_ms, result_path, plot_format)
    
if __name__ == '__main__':
    main()
    