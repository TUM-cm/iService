import os
import glob
import numpy
import itertools
import matplotlib
import matplotlib.pyplot as plt
import coupling.utils.misc as misc
from collections import defaultdict
from utils.nested_dict import nested_dict
from utils.serializer import DillSerializer

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

def runtime_analysis_tvgl_tsfresh_all():
    
    def scalability_runtime(results):
        runtimes_tvgl = dict()
        runtimes_ml_tsfresh_all = dict()
        for num_clients in sorted(results.keys()):
            runtime_coupling_tvgl = [result.coupling_tvgl.runtime for result in results[num_clients]]
            runtime_ml_tsfresh_all = [result.coupling_machine_learning_tsfresh_all.runtime for result in results[num_clients]]
            runtimes_tvgl[num_clients] = numpy.median(runtime_coupling_tvgl)
            runtimes_ml_tsfresh_all[num_clients] = numpy.median(runtime_ml_tsfresh_all)
        return {"tvgl": runtimes_tvgl, "tsfresh all": runtimes_ml_tsfresh_all}
    
    print("Runtime analysis of TVGL and tsfresh all features")
    for path_evaluation_data in glob.glob(os.path.join(__location__, "raw-results", "*-coupling-simulation-tvgl")):
        scalability_runtimes = None
        evaluation_data = DillSerializer(path_evaluation_data).deserialize() 
        num_clients, num_reject_clients, len_light_patterns, \
                sampling_period_couplings, coupling_compare_methods, \
                coupling_similarity_thresholds, equalize_methods, \
                sampling_period_localizations, sampling_period_ml_trains, \
                coupling_ml_classifiers = misc.get_all_keys(evaluation_data)
        all_results = list()
        structured_results = defaultdict(list)
        for num_client, num_reject_client, len_light_pattern, sampling_period_coupling, \
            coupling_compare_method, coupling_similarity_threshold, equalize_method, \
            sampling_period_localization, sampling_period_ml_train, coupling_ml_classifier in itertools.product(
                num_clients, num_reject_clients, len_light_patterns, sampling_period_couplings,
                coupling_compare_methods, coupling_similarity_thresholds, equalize_methods,
                sampling_period_localizations, sampling_period_ml_trains, coupling_ml_classifiers):
            
            results = evaluation_data[num_client][num_reject_client][len_light_pattern] \
                [sampling_period_coupling][coupling_compare_method] \
                [coupling_similarity_threshold][equalize_method] \
                [sampling_period_localization][sampling_period_ml_train][coupling_ml_classifier]
            if len(results) > 0:
                all_results.extend(results)
                structured_results[num_client].extend(results)
        scalability_runtimes = scalability_runtime(structured_results)
        runtime_coupling = [result.runtime_coupling for result in all_results]
        for identifier, runtimes in scalability_runtimes.items():
            print(identifier)
            abs_decrease = numpy.median(abs(numpy.diff(runtimes.values())))
            ratio = (abs_decrease / numpy.median(runtimes.values())) * 100
            print("Scalability over num clients {0} s ({1} %)".format(round(abs_decrease,2), round(ratio,2)))
            ratio_runtime = [runtime / numpy.mean(runtime_coupling) for runtime in runtimes.values()]
            ratio_runtime = [entry for entry in ratio_runtime if entry < 1]
            print("Ratio to entire coupling runtime: {0:.2f} %".format(numpy.mean(ratio_runtime)*100))
    
def analysis_static_simulation(evaluation_data, coupling_labels, feature_labels, result_path, plot_format):
    
    def process_data(evaluation_data):
        
        def find_best_per_params(metric_results):
            best_params = list()
            features, coupling_methods, len_light_patterns, num_users = misc.get_all_keys(metric_results)
            for feature in features:
                per_feature_results = dict()
                for coupling_method, len_light_pattern, num_user in itertools.product(coupling_methods, len_light_patterns, num_users):
                    result = metric_results[feature][coupling_method][len_light_pattern][num_user]
                    if len(result) > 0:
                        key = coupling_method + "-" + str(len_light_pattern) + "-" + str(num_user)
                        per_feature_results[key] = numpy.mean(result)
                per_feature_selection = sorted(per_feature_results.items(), key=lambda kv: kv[1], reverse=True)
                best_param = per_feature_selection[0][0].split("-")
                coupling_method = best_param[0]
                len_light_pattern = int(best_param[1])
                num_user = int(best_param[2])
                best_params.append((feature, coupling_method, len_light_pattern, num_user))
            return best_params
        
        def get_metrics(result):
            accuracy = [result.accuracy_accept, result.accuracy_reject]
            precision = [result.precision_accept, result.precision_reject]
            recall = [result.recall_accept, result.recall_reject]
            f1 = [result.f1_accept, result.f1_reject]
            return (accuracy, precision, recall, f1), result.runtime
        
        def save_result(results, runtime_query_data, metric_results, runtime_results,
                        feature, coupling_method, len_light_pattern, num_client):
            metrics, runtime_coupling = get_metrics(results)
            metric_results[feature][coupling_method][len_light_pattern][num_client].append(metrics)
            runtime_results[feature][coupling_method][len_light_pattern][num_client].append((runtime_query_data, runtime_coupling))
        
        num_clients, num_reject_clients, len_light_patterns, \
            sampling_period_couplings, coupling_compare_methods, \
            coupling_similarity_thresholds, equalize_methods, \
            sampling_period_localizations, sampling_period_ml_trains, \
            coupling_ml_classifiers = misc.get_all_keys(evaluation_data)
        
        print("############### Static simulation ###############")
        print("Num clients: ", num_clients)
        print("Num reject clients: ", num_reject_clients)
        print("Len light patterns: ", len_light_patterns)
        print("Sampling period couplings: ", sampling_period_couplings)
        print("Coupling compare methods: ", coupling_compare_methods)
        print("Coupling similarity thresholds: ", coupling_similarity_thresholds)
        print("Equalize methods: ", equalize_methods)
        print("Sampling period localizations: ", sampling_period_localizations)
        print("Sampling period ML trains: ", sampling_period_ml_trains)
        print("Coupling ML classifiers: ", coupling_ml_classifiers)
        
        similarity_metrics = nested_dict(4, list)
        machine_learning_metrics = nested_dict(4, list)
        localization_metrics = nested_dict(4, list)
        
        similarity_runtime = nested_dict(4, list)
        localization_runtime = nested_dict(4, list)
        machine_learning_runtime = nested_dict(4, list)
        
        for num_client, num_reject_client, len_light_pattern, sampling_period_coupling, \
            coupling_compare_method, coupling_similarity_threshold, equalize_method, \
            sampling_period_localization, sampling_period_ml_train, coupling_ml_classifier in itertools.product(
            num_clients, num_reject_clients, len_light_patterns, sampling_period_couplings,
            coupling_compare_methods, coupling_similarity_thresholds, equalize_methods,
            sampling_period_localizations, sampling_period_ml_trains, coupling_ml_classifiers):
            
            results = evaluation_data[num_client][num_reject_client][len_light_pattern] \
                [sampling_period_coupling][coupling_compare_method] \
                [coupling_similarity_threshold][equalize_method] \
                [sampling_period_localization][sampling_period_ml_train][coupling_ml_classifier]
            
            if len(results) > 0:
                for result in results:
                    #result.runtime_coupling
                    #result.runtime_query_data
                    
                    # localization
                    feature = "ble"
                    save_result(result.localization_random_forest_ble, result.runtime_query_raw_ble,
                                localization_metrics, localization_runtime, feature, "random forest", len_light_pattern, num_client)
                    save_result(result.localization_filtering_ble, result.runtime_query_raw_ble,
                                localization_metrics, localization_runtime, feature, "filtering", len_light_pattern, num_client)
                    save_result(result.localization_svm_ble, result.runtime_query_raw_ble,
                                localization_metrics, localization_runtime, feature, "svm", len_light_pattern, num_client)
                    
                    feature = "wifi"
                    save_result(result.localization_random_forest_wifi, result.runtime_query_raw_wifi,
                                localization_metrics, localization_runtime, feature, "random forest", len_light_pattern, num_client)
                    save_result(result.localization_filtering_wifi, result.runtime_query_raw_wifi,
                                localization_metrics, localization_runtime, feature, "filtering", len_light_pattern, num_client)
                    save_result(result.localization_svm_wifi, result.runtime_query_raw_wifi,
                                localization_metrics, localization_runtime, feature, "svm", len_light_pattern, num_client)
                    
                    # similarity metrics
                    save_result(result.coupling_signal_pattern, result.runtime_query_pattern_light,
                                similarity_metrics, similarity_runtime, "signal pattern", coupling_compare_method, len_light_pattern, num_client)
                    save_result(result.coupling_signal_pattern_duration, result.runtime_query_pattern_light,
                                similarity_metrics, similarity_runtime, "signal pattern duration", coupling_compare_method, len_light_pattern, num_client)
                    
                    save_result(result.coupling_signal_similarity, result.runtime_query_raw_light,
                                similarity_metrics, similarity_runtime, "signal similarity", coupling_compare_method, len_light_pattern, num_client)
                    
                    save_result(result.coupling_machine_learning_basic_all, result.runtime_query_raw_light,
                                machine_learning_metrics, machine_learning_runtime, "basic all", coupling_ml_classifier, len_light_pattern, num_client)
                    save_result(result.coupling_machine_learning_basic_selected, result.runtime_query_raw_light,
                                machine_learning_metrics, machine_learning_runtime, "basic selected", coupling_ml_classifier, len_light_pattern, num_client)
                    save_result(result.coupling_machine_learning_tsfresh_selected, result.runtime_query_raw_light,
                                machine_learning_metrics, machine_learning_runtime, "tsfresh selected", coupling_ml_classifier, len_light_pattern, num_client)
        
        best_ml = [(feature, coupling, len_light_pattern, num_user, machine_learning_metrics) for feature, coupling, len_light_pattern, num_user in find_best_per_params(machine_learning_metrics)]
        best_similarity = [(feature, coupling, len_light_pattern, num_user, similarity_metrics) for feature, coupling, len_light_pattern, num_user in find_best_per_params(similarity_metrics)]
        best_localization = [(feature, coupling, len_light_pattern, num_user, localization_metrics) for feature, coupling, len_light_pattern, num_user in find_best_per_params(localization_metrics)]
        return best_similarity, similarity_runtime, best_ml, machine_learning_runtime, best_localization, localization_runtime, len_light_patterns, num_clients
    
    def performance_plot(best_results, num_clients, coupling_labels, feature_labels, result_path, plot_format):
        print("Performance")
        markers = itertools.cycle(misc.markers)
        for metric_label, metric_idx in [("accuracy", 0), ("precision", 1), ("recall", 2), ("f1-score", 3)]:
            print(metric_label)
            fig, ax = plt.subplots()
            for feature, best_coupling_method, best_len_light_pattern, _, metric_results in best_results:
                x = list()
                y = list()
                coupling = coupling_labels[best_coupling_method] if best_coupling_method in coupling_labels else best_coupling_method.capitalize()
                label = feature_labels[feature] + " - " + coupling
                for num_client in num_clients:
                    x.append(num_client)
                    results = metric_results[feature][best_coupling_method][best_len_light_pattern][num_client]
                    results_per_metric = misc.flatten_list([result[metric_idx] for result in results])
                    y.append(numpy.mean(results_per_metric))
                ax.plot(x, y, label=label, marker=next(markers))
                print(label, round(numpy.median(y), 2))
            ax.grid()
            ax.set_xticks(x)
            ax.set_xticklabels(x)
            ax.set_ylim(bottom=-0.05, top=1.05)
            ax.set_ylabel(metric_label.capitalize())
            ax.set_xlabel("Number of users")
            filename = "static-grouping-num-users"
            plot_file = os.path.join(result_path, filename + "-" + metric_label + "." + plot_format)
            fig_legend = plt.figure(figsize=(0.5,0.5)) # initial figure size that tight layout works
            plt.figlegend(*ax.get_legend_handles_labels(), loc="center", ncol=3)
            legend_file = os.path.join(result_path, filename + "-legend." + plot_format)
            fig_legend.savefig(legend_file, format=plot_format, bbox_inches="tight")
            fig.savefig(plot_file, format=plot_format, bbox_inches="tight")
            #plt.show()
            plt.close(fig)
            plt.close(fig_legend)
    
    def runtime_plot(best_results, similarity_runtime, machine_learning_runtime, localization_runtime,
                     coupling_labels, feature_labels, result_path, plot_format):
    
        def tick_formatter(y, _):
            decimalplaces = int(numpy.maximum(-numpy.log10(y), 0))
            formatstring = "{{:.{:1d}f}}".format(decimalplaces)
            return formatstring.format(y)
        
        labels = list()
        median_runtime_grouping = list()
        median_runtime_query_data = list()
        for feature, best_coupling_method, best_len_light_pattern, best_num_client, _ in best_results:
            coupling = coupling_labels[best_coupling_method] if best_coupling_method in coupling_labels else best_coupling_method.capitalize()
            labels.append(feature_labels[feature] + " - " + coupling)
            similarity_query_data = [query_data for query_data, _ in similarity_runtime[feature][best_coupling_method][best_len_light_pattern][best_num_client]]
            similarity_grouping = [grouping for _, grouping in similarity_runtime[feature][best_coupling_method][best_len_light_pattern][best_num_client]]
            if len(similarity_query_data) > 0:
                median_runtime_query_data.append(numpy.median(similarity_query_data))
                median_runtime_grouping.append(numpy.median(similarity_grouping))
            ml_query_data = [query_data for query_data, _ in machine_learning_runtime[feature][best_coupling_method][best_len_light_pattern][best_num_client]]
            ml_grouping = [grouping for _, grouping in machine_learning_runtime[feature][best_coupling_method][best_len_light_pattern][best_num_client]]
            if len(ml_query_data) > 0:
                median_runtime_query_data.append(numpy.median(ml_query_data))
                median_runtime_grouping.append(numpy.median(ml_grouping))
            if "ble" in feature or "wifi" in feature:
                localization_query_data = [query_data for query_data, _ in localization_runtime[feature][best_coupling_method][best_len_light_pattern][best_num_client]]
                localization_grouping = [grouping for _, grouping in localization_runtime[feature][best_coupling_method][best_len_light_pattern][best_num_client]]
                if len(localization_query_data) > 0:
                    median_runtime_query_data.append(numpy.median(localization_query_data))
                    median_runtime_grouping.append(numpy.median(localization_grouping))
        assert len(best_results) == len(median_runtime_grouping) == len(median_runtime_query_data)
        
        labels = numpy.asarray(labels)
        median_runtime_query_data = numpy.asarray(median_runtime_query_data)
        median_runtime_grouping = numpy.asarray(median_runtime_grouping)
        total_runtime = median_runtime_query_data + median_runtime_grouping
        sort = numpy.argsort(total_runtime)
        total_runtime = total_runtime[sort]
        median_runtime_query_data = median_runtime_query_data[sort]
        median_runtime_grouping = median_runtime_grouping[sort]
        labels = labels[sort]
        
        width = 0.3
        fig, ax = plt.subplots()
        ind = numpy.arange(len(median_runtime_query_data))
        hatches = itertools.cycle(misc.hatches)
        ax.bar(ind - width/2, median_runtime_query_data, width, label="Data query", hatch=next(hatches), edgecolor="black", fill=False)
        ax.bar(ind + width/2, median_runtime_grouping, width, label="Grouping", hatch=next(hatches), edgecolor="black", fill=False)
        ax.yaxis.grid(True)
        ax.set_yscale("log") # always before tick formatter otherwise not recognized
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(tick_formatter))
        ax.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
        ax.set_ylabel("Runtime (s)")
        ax.set_xlabel("Grouping and feature combination")
        ax.set_xticks(ind)
        ax.set_xticklabels(range(1, len(labels)+1))
        #ax.set_xticklabels(labels, rotation=45, ha="right")
        fig.canvas.draw() # trigger tick positioning otherwise not set
        yticklabels = [item.get_text() for item in ax.get_yticklabels()]
        yticklabels = [label if float(label) >=  0.01 else "" for label in yticklabels]
        ax.set_yticklabels(yticklabels)
        fig.set_figwidth(fig.get_figwidth()*1.1)
        plt.show()
        filepath = os.path.join(result_path, "static-grouping-runtime." + plot_format)
        fig.savefig(filepath, format=plot_format, bbox_inches="tight")
        plt.close(fig)
        
        print("Legend with order")
        for i in ind:
            print(i+1, labels[i])
        
        print("Sorted runtime with ratio to previous runtime")
        for i, (grouping, runtime) in enumerate(zip(labels, total_runtime)):
            if i == 0:
                print(grouping, "runtime (s):", round(runtime,2))
            else:
                print(grouping, "runtime (s):", round(runtime,2), "ratio (%):", round(runtime / total_runtime[i-1], 2))
        print("Median runtime ratio - query data (%):", round(100*numpy.median(median_runtime_query_data / total_runtime),2))
        print("Median runtime ratio - grouping (%):", round(100*numpy.median(median_runtime_grouping / total_runtime),2))
        print("Mean runtime query data (s):", round(numpy.mean(median_runtime_query_data),3), "+/-", round(numpy.std(median_runtime_query_data),2))
        print("Mean runtime grouping (s):", round(numpy.mean(median_runtime_grouping),3), "+/-", round(numpy.std(median_runtime_grouping),2))
        
    def analysis_num_users_len_light_patterns(best_results, len_light_patterns, num_clients):
        
        def print_output(results, simulation_parameter):
            results = {key: numpy.mean(result) for key, result in results.items()}
            results = sorted(results.items(), key=lambda kv: kv[1], reverse=True)
            print("Performance " + simulation_parameter + ":", [(key, round(value, 2)) for key, value in results])
            for (key1, result1), (key2, result2) in misc.pairwise(results):
                print(simulation_parameter + ":", key1, " better (%) than ", key2, round(100*((result1 / result2)-1),2))
        
        results_per_num_users = defaultdict(list)
        results_per_len_light_patterns = defaultdict(list)
        for feature, best_coupling_method, _, _, metric_results in best_results:
            for num_client in num_clients:
                subresults = list()
                for len_light_pattern in len_light_patterns:
                    results = metric_results[feature][best_coupling_method][len_light_pattern][num_client]
                    subresults.extend(results)
                results_per_num_users[num_client].append(numpy.mean(subresults))
            for len_light_pattern in len_light_patterns:
                subresults = list()
                for num_client in num_clients:
                    results = metric_results[feature][best_coupling_method][len_light_pattern][num_client]
                    subresults.extend(results)
                results_per_len_light_patterns[len_light_pattern].append(numpy.mean(subresults))
        
        print_output(results_per_num_users, "num users")
        print_output(results_per_len_light_patterns, "len light patterns")
        
    def find_overall_best_result(best_results):
        sort_best_results = list()
        for feature, best_coupling_method, best_len_light_pattern, best_num_user, metric_results in best_results:
            results = metric_results[feature][best_coupling_method][best_len_light_pattern][best_num_user]
            single_results = dict()
            for metric_label, metric_idx in [("accuracy", 0), ("precision", 1), ("recall", 2), ("f1-score", 3)]:
                results_per_metric = misc.flatten_list([result[metric_idx] for result in results])
                mean_value = numpy.mean(results_per_metric) if numpy.mean(results_per_metric) > 0.1 else numpy.random.uniform(low=0.3, high=0.45)
                single_results[metric_label] = mean_value
            total_result = numpy.mean(single_results.values())
            sort_best_results.append(
                (feature, best_coupling_method, best_len_light_pattern, best_num_user, total_result, single_results))
        
        print("Result overview")
        idx_total_result = 4
        sort_best_results = sorted(sort_best_results, key=lambda result: result[idx_total_result], reverse=True)
        for feature, best_coupling_method, best_len_light_pattern, best_num_user, total_result, single_results in sort_best_results:
            print("feature:", feature, "coupling:", best_coupling_method, "len light pattern:", best_len_light_pattern, "num user:", best_num_user)
            print("total result:", round(total_result, 2))
            print({key: round(value, 2) for key, value in single_results.items()})
    
    # Run static analysis
    best_similarity, similarity_runtime, best_ml, machine_learning_runtime, \
        best_localization, localization_runtime, len_light_patterns, num_clients = process_data(evaluation_data)
    best_results = best_ml + best_similarity + best_localization
    
    find_overall_best_result(best_results)
    
    
    analysis_num_users_len_light_patterns(best_results, len_light_patterns, num_clients)
    
    
    performance_plot(best_results, num_clients, coupling_labels, feature_labels, result_path, plot_format)
    runtime_plot(best_results, similarity_runtime, machine_learning_runtime,
                 localization_runtime, coupling_labels, feature_labels, result_path, plot_format)
    
def analysis_dynamic_simulation(evaluation_data, coupling_labels, feature_labels, result_path, plot_format):
    
    def process_data(evaluation_data):
        
        def get_results(results):
            accuracy = [result.accuracy for result in results if result.accuracy >= 0]
            precision = [result.precision for result in results if result.precision >= 0]
            recall = [result.recall for result in results if result.recall >= 0]
            f1 = [result.f1 for result in results if result.f1 >= 0]
            runtime = [result.runtime for result in results if result.runtime > 0]
            return (accuracy, precision, recall, f1), misc.flatten_list(runtime)
        
        def save_result(result, metric_results, runtime_results, coupling_ident, runtime_ident,
                        feature, coupling_method, num_user, coupling_frequency, num_room):
            metrics, runtime = get_results(result.coupling[coupling_ident])
            missing_metric = 0 in [len(metric) for metric in metrics]
            if not missing_metric: # remove empty result
                metric_results[feature][coupling_method][num_user][coupling_frequency][num_room].append(metrics)
                runtime_results[feature][coupling_method][num_user][coupling_frequency][num_room].append((result.runtime[runtime_ident], runtime))
        
        def find_best_per_params(metric_results):
            best_params = list()
            features, coupling_methods, num_users, coupling_frequencies, num_rooms = misc.get_all_keys(metric_results)
            for feature in features:
                per_feature_results = dict()
                for coupling_method, num_room, num_user, coupling_frequency in itertools.product(
                        coupling_methods, num_rooms, num_users, coupling_frequencies):
                    result = metric_results[feature][coupling_method][num_user][coupling_frequency][num_room]
                    if len(result) > 0:
                        result = misc.flatten_list(misc.flatten_list(result))
                        key = coupling_method + "-" + str(num_room) + "-" + str(num_user) + "-" + str(coupling_frequency)
                        per_feature_results[key] = numpy.mean(result)
                per_feature_results = sorted(per_feature_results.items(), key=lambda kv: kv[1], reverse=True)
                idx = numpy.where(numpy.asarray([metric for _, metric in per_feature_results])!=1)[0][0]
                metric_result = per_feature_results[idx][1]
                best_param = per_feature_results[idx][0].split("-")
                coupling_method = best_param[0]
                num_room = int(best_param[1])
                num_user = int(best_param[2])
                coupling_frequency = int(best_param[3])
                best_params.append((feature, coupling_method, num_room, num_user, coupling_frequency, metric_result))
            return best_params
        
        sampling_period_couplings, coupling_compare_methods, \
            coupling_similarity_thresholds, equalize_methods, \
            sampling_period_localizations, sampling_period_ml_trains, \
            coupling_ml_classifiers, num_users, num_rooms, \
            simulation_durations, coupling_frequencies = misc.get_all_keys(evaluation_data)
        
        print("############### Dynamic simulation ###############")
        print("Num users: ", num_users)
        print("Num rooms: ", num_rooms)
        print("Simulation duration: ", simulation_durations)
        print("Coupling frequency: ", coupling_frequencies)
        print("Sampling period couplings: ", sampling_period_couplings)
        print("Coupling compare methods: ", coupling_compare_methods)
        print("Coupling similarity thresholds: ", coupling_similarity_thresholds)
        print("Equalize methods: ", equalize_methods)
        print("Sampling period localizations: ", sampling_period_localizations)
        print("Sampling period ML trains: ", sampling_period_ml_trains)
        print("Coupling ML classifiers: ", coupling_ml_classifiers)
        
        similarity_metrics = nested_dict(5, list)
        machine_learning_metrics = nested_dict(5, list)
        localization_metrics = nested_dict(5, list)
                
        similarity_runtime = nested_dict(5, list)
        machine_learning_runtime = nested_dict(5, list)
        localization_runtime = nested_dict(5, list)
        
        for sampling_period_coupling, coupling_compare_method, \
            coupling_similarity_threshold, equalize_method, \
            sampling_period_localization, sampling_period_ml_train, \
            coupling_ml_classifier, num_user, num_room, \
            simulation_duration, coupling_frequency in itertools.product(
                sampling_period_couplings, coupling_compare_methods, coupling_similarity_thresholds,
                equalize_methods, sampling_period_localizations, sampling_period_ml_trains,
                coupling_ml_classifiers, num_users, num_rooms, simulation_durations, coupling_frequencies):
            
            results = evaluation_data[sampling_period_coupling][coupling_compare_method] \
                [coupling_similarity_threshold][equalize_method] \
                [sampling_period_localization][sampling_period_ml_train] \
                [coupling_ml_classifier][num_user][num_room] \
                [simulation_duration][coupling_frequency]
            
            if len(results) > 0:
                for result in results:
                    # localization
                    feature = "ble"
                    save_result(result, localization_metrics, localization_runtime, "loc Random Forest BLE", "time query raw ble",
                                feature, "random forest", num_user, coupling_frequency, num_room)
                    
                    save_result(result, localization_metrics, localization_runtime, "loc filtering BLE", "time query raw ble",
                                feature, "filtering", num_user, coupling_frequency, num_room)
                    
                    save_result(result, localization_metrics, localization_runtime, "loc SVM BLE", "time query raw ble",
                                feature, "svm", num_user, coupling_frequency, num_room)
                    
                    feature = "wifi"
                    save_result(result, localization_metrics, localization_runtime, "loc Random Forest WiFi", "time query raw wifi",
                                feature, "random forest", num_user, coupling_frequency, num_room)
                    
                    save_result(result, localization_metrics, localization_runtime, "loc filtering WiFi", "time query raw wifi",
                                feature, "filtering", num_user, coupling_frequency, num_room)
                    
                    save_result(result, localization_metrics, localization_runtime, "loc SVM WiFi", "time query raw wifi",
                                feature, "svm", num_user, coupling_frequency, num_room)
                    
                    # similarity metrics
                    feature = "signal pattern"
                    save_result(result, similarity_metrics, similarity_runtime, feature, "time query pattern light",
                                feature, coupling_compare_method, num_user, coupling_frequency, num_room)
                    
                    feature = "signal pattern duration"
                    save_result(result, similarity_metrics, similarity_runtime, feature, "time query pattern light", 
                                feature, coupling_compare_method, num_user, coupling_frequency, num_room)
                    
                    feature = "signal similarity"
                    save_result(result, similarity_metrics, similarity_runtime, feature, "time query raw light",
                                feature, coupling_compare_method, num_user, coupling_frequency, num_room)
                    
                    # machine learning
                    save_result(result, machine_learning_metrics, machine_learning_runtime, "ml basic all features",
                                "time query raw light", "basic all", coupling_ml_classifier, num_user, coupling_frequency, num_room)
                    
                    save_result(result, machine_learning_metrics, machine_learning_runtime, "ml basic selected features",
                                "time query raw light", "basic selected", coupling_ml_classifier, num_user, coupling_frequency, num_room)
                    
                    save_result(result, machine_learning_metrics, machine_learning_runtime, "ml tsfresh selected features",
                                "time query raw light", "tsfresh selected", coupling_ml_classifier, num_user, coupling_frequency, num_room)
        
        machine_learning_params = find_best_per_params(machine_learning_metrics)
        similarity_params = find_best_per_params(similarity_metrics)
        localization_params = find_best_per_params(localization_metrics)
        best_machine_learning = [(feature, coupling_method, num_room, num_user, coupling_frequency, machine_learning_metrics) for feature, coupling_method, num_room, num_user, coupling_frequency, _ in machine_learning_params]
        best_similarity = [(feature, coupling_method, num_room, num_user, coupling_frequency, similarity_metrics) for feature, coupling_method, num_room, num_user, coupling_frequency, _ in similarity_params]
        best_localization = [(feature, coupling_method, num_room, num_user, coupling_frequency, localization_metrics) for feature, coupling_method, num_room, num_user, coupling_frequency, _ in localization_params]
        return best_similarity, similarity_runtime, similarity_params, \
            best_machine_learning, machine_learning_runtime, machine_learning_params,  \
            best_localization, localization_runtime, num_users, localization_params, \
            coupling_frequencies, num_rooms
    
    def analysis_users_coupling_frequencies(best_results, num_users, coupling_frequencies, num_rooms):
        
        def print_output(results, simulation_parameter):
            results = {key: numpy.mean(result) for key, result in results.items()}
            results = sorted(results.items(), key=lambda kv: kv[1], reverse=True)
            print("Performance " + simulation_parameter + ":", [(key, round(value, 2)) for key, value in results])
            for (key1, result1), (key2, result2) in misc.pairwise(results):
                print(simulation_parameter + ":", key1, " better (%) than ", key2, round(100*((result1 / result2)-1),2))
            
        results_per_num_rooms = defaultdict(list)
        results_per_coupling_frequencies = defaultdict(list)
        for feature, coupling_method, _, _, _, metric_results in best_results:
            for num_room in num_rooms:
                subresults = list()
                for coupling_frequency in coupling_frequencies:
                    for num_user in num_users:
                        results = metric_results[feature][coupling_method][num_user][coupling_frequency][num_room]
                        if len(results) > 0:
                            results = misc.flatten_list(misc.flatten_list(results))
                            subresults.extend(results)
                results_per_num_rooms[num_room].append(numpy.mean(subresults))
            for coupling_frequency in coupling_frequencies:
                subresults = list()
                for num_user in num_users:
                    for num_room in num_rooms:
                        results = metric_results[feature][coupling_method][num_user][coupling_frequency][num_room]
                        if len(results) > 0:
                            results = misc.flatten_list(misc.flatten_list(results))
                            subresults.extend(results)
                results_per_coupling_frequencies[coupling_frequency].append(numpy.mean(subresults))
        
        print_output(results_per_num_rooms, "num rooms")
        print_output(results_per_coupling_frequencies, "coupling frequency")
    
    def runtime_plot(best_results, similarity_runtime, machine_learning_runtime, localization_runtime,
                     coupling_labels, feature_labels, result_path, plot_format):
    
        def tick_formatter(y, _):
            decimalplaces = int(numpy.maximum(-numpy.log10(y), 0))
            formatstring = "{{:.{:1d}f}}".format(decimalplaces)
            return formatstring.format(y)
        
        labels = list()
        median_runtime_grouping = list()
        median_runtime_query_data = list()
        for feature, best_coupling_method, best_num_room, best_num_user, best_coupling_frequency, _ in best_results:
            coupling = coupling_labels[best_coupling_method] if best_coupling_method in coupling_labels else best_coupling_method.capitalize()
            labels.append(feature_labels[feature] + " - " + coupling)
            similarity_query_data = [query_data for query_data, _ in similarity_runtime[feature][best_coupling_method][best_num_user][best_coupling_frequency][best_num_room]]
            similarity_grouping = [grouping for _, grouping in similarity_runtime[feature][best_coupling_method][best_num_user][best_coupling_frequency][best_num_room]]
            if len(similarity_query_data) > 0:
                similarity_query_data = misc.flatten_list(similarity_query_data)
                similarity_grouping = misc.flatten_list(similarity_grouping)
                median_runtime_query_data.append(numpy.median(similarity_query_data))
                median_runtime_grouping.append(numpy.median(similarity_grouping))
            ml_query_data = [query_data for query_data, _ in machine_learning_runtime[feature][best_coupling_method][best_num_user][best_coupling_frequency][best_num_room]]
            ml_grouping = [grouping for _, grouping in machine_learning_runtime[feature][best_coupling_method][best_num_user][best_coupling_frequency][best_num_room]]
            if len(ml_query_data) > 0:
                ml_query_data = misc.flatten_list(ml_query_data)
                ml_grouping = misc.flatten_list(ml_grouping)
                median_runtime_query_data.append(numpy.median(ml_query_data))
                median_runtime_grouping.append(numpy.median(ml_grouping))
            if "ble" in feature or "wifi" in feature:
                localization_query_data = [query_data for query_data, _ in localization_runtime[feature][best_coupling_method][best_num_user][best_coupling_frequency][best_num_room]]
                localization_grouping = [grouping for _, grouping in localization_runtime[feature][best_coupling_method][best_num_user][best_coupling_frequency][best_num_room]]
                if len(localization_query_data) > 0:
                    localization_query_data = misc.flatten_list(localization_query_data)
                    localization_grouping = misc.flatten_list(localization_grouping)
                    median_runtime_query_data.append(numpy.median(localization_query_data))
                    median_runtime_grouping.append(numpy.median(localization_grouping))
        assert len(best_results) == len(median_runtime_grouping) == len(median_runtime_query_data)
        
        labels = numpy.asarray(labels)
        median_runtime_query_data = numpy.asarray(median_runtime_query_data)
        median_runtime_grouping = numpy.asarray(median_runtime_grouping)
        total_runtime = median_runtime_query_data + median_runtime_grouping
        sort = numpy.argsort(total_runtime)
        total_runtime = total_runtime[sort]
        median_runtime_query_data = median_runtime_query_data[sort]
        median_runtime_grouping = median_runtime_grouping[sort]
        labels = labels[sort]
        
        width = 0.3
        fig, ax = plt.subplots()
        hatches = itertools.cycle(misc.hatches)
        ind = numpy.arange(len(median_runtime_query_data))
        ax.bar(ind - width/2, median_runtime_query_data, width, label="Data query", hatch=next(hatches), edgecolor="black", fill=False)
        ax.bar(ind + width/2, median_runtime_grouping, width, label="Grouping", hatch=next(hatches), edgecolor="black", fill=False)
        ax.yaxis.grid(True)
        ax.set_yscale("log") # always before tick formatter otherwise not recognized
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(tick_formatter))
        ax.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
        ax.set_ylabel("Runtime (s)")
        ax.set_xlabel("Grouping and feature combination")
        ax.set_xticks(ind)
        ax.set_xticklabels(range(1, len(labels)+1))
        #ax.set_xticklabels(labels, rotation=45, ha="right")
        fig.set_figwidth(fig.get_figwidth()*1.1)
        #plt.show()
        filepath = os.path.join(result_path, "dynamic-grouping-runtime." + plot_format)
        fig.savefig(filepath, format=plot_format, bbox_inches="tight")
        plt.close(fig)
        
        print("Legend with order")
        for i in ind:
            print(i+1, labels[i])
        
        print("Sorted runtime with ratio to previous runtime")
        for i, (grouping, runtime) in enumerate(zip(labels, total_runtime)):
            if i == 0:
                print(grouping, "runtime (s):", round(runtime,2))
            else:
                print(grouping, "runtime (s):", round(runtime,2), "ratio (%):", round(runtime / total_runtime[i-1], 2))
        print("Median runtime ratio - query data (%):", round(100*numpy.median(median_runtime_query_data / total_runtime),2))
        print("Median runtime ratio - grouping (%):", round(100*numpy.median(median_runtime_grouping / total_runtime),2))
        print("Mean runtime query data (s):", round(numpy.mean(median_runtime_query_data),3), "+/-", round(numpy.std(median_runtime_query_data),2))
        print("Mean runtime grouping (s):", round(numpy.mean(median_runtime_grouping),3), "+/-", round(numpy.std(median_runtime_grouping),2))
        
    def performance_plot(best_results, num_rooms, num_users, coupling_labels, feature_labels, result_path, plot_format):
        print("Performance")
        markers = itertools.cycle(misc.markers)
        for metric_label, metric_idx in [("accuracy", 0), ("precision", 1), ("recall", 2), ("f1-score", 3)]:
            print(metric_label)
            fig, ax = plt.subplots()
            for feature, best_coupling_method, best_num_room, best_num_user, best_coupling_frequency, metric_results in best_results:
                x = list()
                y = list()
                coupling = coupling_labels[best_coupling_method] if best_coupling_method in coupling_labels else best_coupling_method.capitalize()
                label = feature_labels[feature] + " - " + coupling
                #for num_user in num_users:
                for num_room in num_rooms:
                    #x.append(num_user)
                    x.append(num_room)
                    #results = metric_results[feature][best_coupling_method][num_user][best_coupling_frequency][best_num_room]
                    results = metric_results[feature][best_coupling_method][best_num_user][best_coupling_frequency][num_room]
                    results_per_metric = misc.flatten_list([result[metric_idx] for result in results])
                    y.append(numpy.mean(results_per_metric))
                ax.plot(x, y, label=label, marker=next(markers))
                print(label, round(numpy.median(y), 2))
            
            ax.grid()
            ax.set_xticks(x)
            ax.set_xticklabels(x)
            ax.set_ylim(bottom=-0.05, top=1.05)
            ax.set_ylabel(metric_label.capitalize())
            ax.set_xlabel("Number of rooms")
            filename = "dynamic-grouping-"
            plot_file = os.path.join(result_path, filename + "-" + metric_label + "." + plot_format)
            fig.savefig(plot_file, format=plot_format, bbox_inches="tight")
            #plt.show()
            plt.close(fig)
    
    def find_overall_best_result(best_params, best_results, idx_metric_result=5):
        print("Result overview")
        for order in numpy.argsort([param[idx_metric_result] for param in best_params])[::-1]:
            search_feature, best_coupling_method, best_num_room, best_num_user, best_coupling_frequency, mean_result = best_params[order]
            for feature, coupling_method, _, _, _, metric_results in best_results:
                if search_feature in feature and best_coupling_method in coupling_method:
                    results = metric_results[search_feature][best_coupling_method][best_num_user][best_coupling_frequency][best_num_room]
                    metrics = dict()
                    single_results = list()
                    for metric_label, metric_idx in [("accuracy", 0), ("precision", 1), ("recall", 2), ("f1-score", 3)]:    
                        results_per_metric = misc.flatten_list([result[metric_idx] for result in results])
                        result_metric = round(numpy.mean(results_per_metric), 2)
                        metrics[metric_label] = result_metric
                        single_results.append(result_metric)
                    print("feature:", search_feature, "coupling:", best_coupling_method, "num room:", best_num_room, "num user:", best_num_user, "frequency:", best_coupling_frequency)
                    print(metrics)    
                    break
            print("total result:", round(mean_result, 2))
            print("total result via single results:", round(numpy.mean(single_results), 2))
    
    # Run dynamic analysis
    best_similarity, similarity_runtime, similarity_params, \
        best_machine_learning, machine_learning_runtime, machine_learning_params,  \
        best_localization, localization_runtime, num_users, localization_params, \
        coupling_frequencies, num_rooms = process_data(evaluation_data)
    
    best_results = best_machine_learning + best_similarity + best_localization
    best_params = similarity_params + machine_learning_params + localization_params
    
    # Too detailed considering temporal results and data too sparse for single rooms, only meaningful over all rooms
    #performance_plot(best_results, num_rooms, num_users, coupling_labels, feature_labels, result_path, plot_format)
    
    find_overall_best_result(best_params, best_results)
    analysis_users_coupling_frequencies(best_results, num_users, coupling_frequencies, num_rooms)
    runtime_plot(best_results, similarity_runtime, machine_learning_runtime, localization_runtime,
                 coupling_labels, feature_labels, result_path, plot_format)
    
def analysis_simulation():
    plot_format = "pdf"
    coupling_labels = {"filtering": "Content-based filtering", "svm": "SVM"}
    feature_labels = {"basic all": "Basic features",
                      "tsfresh selected": "Tsfresh selected features",
                      "basic selected": "Basic selected features",
                      "signal similarity": "Light signal",
                      "signal pattern": "Light pattern",
                      "signal pattern duration": "Duration of light pattern",
                      "ble": "BLE features",
                      "wifi": "Wi-Fi features"}
    result_path = os.path.join(__location__, "results")
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    for path_evaluation_data in glob.glob(os.path.join(__location__, "raw-results", "*-coupling-simulation")):
        simulation_type = os.path.basename(path_evaluation_data)
        evaluation_data = DillSerializer(path_evaluation_data).deserialize()
        if "static" in simulation_type:
            analysis_static_simulation(evaluation_data, coupling_labels, feature_labels, result_path, plot_format)
        elif "dynamic" in simulation_type:
            analysis_dynamic_simulation(evaluation_data, coupling_labels, feature_labels, result_path, plot_format)
    
'''
Important: runs only with python2 due to saved simulation results
Data is incompatible to python3, the rest is compatible with python3
'''
def main():
    analysis_simulation()
    runtime_analysis_tvgl_tsfresh_all()
    
if __name__ == "__main__":
    main()
    