from __future__ import division
import os
import time
import glob
import numpy
import pandas
import itertools
import matplotlib.pyplot as plt
import tsfresh.feature_extraction
import coupling.utils.misc as misc
from collections import defaultdict
from collections import OrderedDict
from utils.nested_dict import nested_dict
from utils.serializer import DillSerializer
from coupling.utils.misc import get_all_keys
from coupling.utils.coupling_data import Client
from sklearn.model_selection import ParameterGrid
from coupling.utils.misc import create_random_mac
from sklearn.model_selection import cross_val_score
from coupling.utils.coupling_data import StaticCouplingResult
import coupling.light_grouping_pattern.light_analysis as light_analysis
from coupling.device_grouping.online.machine_learning_features import LightData
from coupling.device_grouping.online.machine_learning_features import Classifier
from coupling.device_grouping.online.machine_learning_features import BasicFeatures
from coupling.device_grouping.online.machine_learning_features import TsFreshFeatures
from coupling.device_grouping.offline.sampling_time import get_pattern_max_sampling_period
from coupling.device_grouping.online.static.coupling_data_provider import CouplingDataProvider

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

def add_data(feature_importances, feature_importance, row):
    for entry in feature_importance:
        feature_importances.loc[row] = entry.relative_importance.values
        row += 1
    return row, feature_importances

def get_features(light_data_type, basic_features_selection,
                 sampling_periods, classifiers, len_light_patterns=None):
    if "single" in light_data_type:
        header_features = basic_features_selection[len_light_patterns[0]][sampling_periods[0]][classifiers[0]][0]    
    else: # combined
        header_features = basic_features_selection[sampling_periods[0]][classifiers[0]][0]
    return header_features.feature.values
    
def analysis_tsfresh_features(testbed, save_features_to_extract, testbed_features_to_extract):
    
    def plot_feature_importance(importances, testbed, filename):
        label = {
            'X__mean_abs_change': 'Mean absolute change',
            'X__standard_deviation': 'Standard deviation',
            'X__variance': 'Variance',
            'X__absolute_sum_of_changes': 'Absolute sum of changes',

            'X__change_quantiles__f_agg_"mean"__isabs_True__qh_1.0__ql_0.2': 'Quantiles absolute change mean (QH=1, QL=.2)',
            'X__change_quantiles__f_agg_"mean"__isabs_True__qh_0.8__ql_0.0': 'Quantiles absolute change mean (QH=.8, QL=0)',
            'X__change_quantiles__f_agg_"mean"__isabs_True__qh_1.0__ql_0.0': 'Quantiles absolute change mean (QH=1, QL=0)',
            
            'X__change_quantiles__f_agg_"var"__isabs_False__qh_1.0__ql_0.0': 'Quantiles change var (QH=1, QL=0)',
            'X__change_quantiles__f_agg_"var"__isabs_False__qh_0.8__ql_0.0': 'Quantiles change var (QH=.8, QL=0)',
            'X__change_quantiles__f_agg_"var"__isabs_False__qh_1.0__ql_0.2': 'Quantiles change var (QH=1, QL=.2)',
            
            'X__change_quantiles__f_agg_"var"__isabs_True__qh_1.0__ql_0.0': 'Quantiles absolute change var (QH=1, QL=0)',
            'X__change_quantiles__f_agg_"var"__isabs_True__qh_0.8__ql_0.0': 'Quantiles absolute change var (QH=.8, QL=0)',
            
            'X__autocorrelation__lag_1': 'Autocorrelation Lag 1',
            'X__autocorrelation__lag_2': 'Autocorrelation Lag 2',
            'X__autocorrelation__lag_3': 'Autocorrelation Lag 3',
            'X__autocorrelation__lag_4': 'Autocorrelation Lag 4',
            'X__autocorrelation__lag_5': 'Autocorrelation Lag 5',
            
            'X__fft_aggregated__aggtype_"centroid"': 'FFT centroid',
            'X__symmetry_looking__r_0.05': 'Symmetry',
            'X__cid_ce__normalize_False': 'CID',
        }
        
        selected_features = importances.median().sort_values()[:10].index.values
        selected_features = numpy.flip(selected_features) # best feature plotted at first place
        importances = importances[selected_features]
        importances.columns = [label[old_column] for old_column in importances.columns]        
        fig, ax = plt.subplots()
        importances.boxplot(ax=ax, vert=False, showfliers=False)
        ax.set_xlabel("p-value")
        ax.set_xscale("log")
        filename = os.path.join(__location__, "results", "machine-learning", testbed, filename + ".pdf")
        directory = os.path.dirname(filename)
        if not os.path.exists(directory):
            os.makedirs(directory)
        plt.savefig(filename, format="pdf", bbox_inches="tight")
        #plt.show()
        plt.close(fig)
    
    def filter_tsfresh_features(testbed, feature_importances, light_data_type):
        filtered_feature_importances = dict()
        for feature_name, p_values in feature_importances.iteritems():
            if len(p_values) > 1:
                filtered_feature_importances[feature_name] = p_values
        # Only for output
        sorted_feature_importances = OrderedDict(sorted(filtered_feature_importances.items(), key=lambda kv: min(kv[1]), reverse=True))
        feature_importances = pandas.DataFrame(dict([ (feature_name, pandas.Series(p_values))
                                                     for feature_name, p_values in sorted_feature_importances.items() ]),
                                                     columns=sorted_feature_importances.keys())
        filename = "tsfresh-features-importance-" + light_data_type
        plot_feature_importance(feature_importances, testbed, filename)
        #return filtered_feature_importances.keys()
        return OrderedDict(sorted(filtered_feature_importances.items(), key=lambda kv: sum(kv[1]))).keys()
        
    for path_tsfresh_data in glob.glob(os.path.join(__location__, "raw-results", "feature-selection", testbed, "*-patterns-tsfresh")):
        light_data_type = os.path.basename(path_tsfresh_data).split("-")[0]
        tsfresh_features_selection = DillSerializer(path_tsfresh_data).deserialize()
        if "single" in light_data_type:
            len_light_patterns = tsfresh_features_selection.keys()
            sampling_periods = tsfresh_features_selection[len_light_patterns[0]].keys()
        elif "combined" in light_data_type:
            sampling_periods = tsfresh_features_selection.keys()
        
        if "single" in light_data_type:
            feature_importances = defaultdict(list)
            for len_light_pattern, sampling_period in itertools.product(len_light_patterns, sampling_periods):
                feature_importance = tsfresh_features_selection[len_light_pattern][sampling_period]
                for entry in feature_importance:
                    for feature_name, p_value in zip(entry.feature.values, entry.p_value.values):
                        feature_importances[feature_name].append(p_value)
            filtered_feature_importances = filter_tsfresh_features(testbed, feature_importances, light_data_type)
        elif "combined" in light_data_type:
            feature_importances = defaultdict(list)
            for sampling_period in sampling_periods:
                feature_importance = tsfresh_features_selection[sampling_period]
                for entry in feature_importance:
                    for feature_name, p_value in zip(entry.feature.values, entry.p_value.values):
                        feature_importances[feature_name].append(p_value)
            filtered_feature_importances = filter_tsfresh_features(testbed, feature_importances, light_data_type)
            
            if save_features_to_extract:
                src_features_extracted = glob.glob(
                    os.path.join(__location__, "..", "online", "ml-train-data", testbed_features_to_extract, "combined-*-tsfresh-features-extracted"))[0]
                target_features_to_be_extracted = os.path.join(
                    __location__, "..", "online", "tsfresh-features-to-be-extracted")
                tsfresh_features_extracted = DillSerializer(src_features_extracted).deserialize()
                tsfresh_features_extracted = tsfresh_features_extracted[tsfresh_features_extracted.keys()[0]][0]
                column_names = filtered_feature_importances
                features_to_extract = tsfresh.feature_extraction.settings.from_columns(
                    tsfresh_features_extracted[column_names])
                #print("features extracted: ", tsfresh_features_extracted.shape)
                DillSerializer(target_features_to_be_extracted).serialize(features_to_extract)
                # For runtime evaluation
                for i in range(len(filtered_feature_importances)):
                    column_names = filtered_feature_importances[:i+1]
                    features_to_extract = tsfresh.feature_extraction.settings.from_columns(
                        tsfresh_features_extracted[column_names])
                    filepath = os.path.join(
                        __location__, "raw-results", "feature-selection", "tsfresh-features-to-be-extracted-" + str(i+1))
                    DillSerializer(filepath).serialize(features_to_extract)
    
def analysis_runtime_tsfresh(testbeds):
    print("runtime tsfresh")
    for runtime_type in ["patterns-runtime", "only-runtime"]:
        for feature_type in ["combined", "single"]:
            runtimes = dict()
            plot_data = list()
            fileregex = "*" + feature_type + "*" + runtime_type + "*"
            labels = {"bbb": "IoT Board", "server": "Virtual Machine", "vm": "Server"}
            for testbed in testbeds:
                filepath = glob.glob(os.path.join(__location__, "raw-results", "feature-selection", testbed, fileregex))
                if len(filepath) == 0: # raw data not available
                    continue
                assert len(filepath) == 1
                path_runtime_tsfresh = filepath[0]
                
                # Get runtimes
                filename = os.path.basename(path_runtime_tsfresh)
                light_data_type = filename.split("-")[0]
                runtime_tsfresh = DillSerializer(path_runtime_tsfresh).deserialize()
                if "only" in filename:
                    if "single" in filename:
                        len_light_patterns = runtime_tsfresh.keys()
                        tmp_runtime_tsfresh = pandas.DataFrame(
                            columns=runtime_tsfresh[len_light_patterns[0]].columns)
                        # Merge runtime only per feature length
                        for len_light_pattern in len_light_patterns:
                            runtime_data = runtime_tsfresh[len_light_pattern]
                            tmp_runtime_tsfresh = tmp_runtime_tsfresh.append(runtime_data, ignore_index=True)
                        runtime_tsfresh = tmp_runtime_tsfresh
                else:
                    if "combined" in filename:
                        sampling_periods = runtime_tsfresh.keys()
                        tmp_runtime_tsfresh = pandas.DataFrame(
                            columns=runtime_tsfresh[sampling_periods[0]][0].columns)
                        row = 0
                        for sampling_period in sampling_periods:
                            runtime_data = runtime_tsfresh[sampling_period]
                            for entry in runtime_data:
                                tmp_runtime_tsfresh.loc[row] = entry.loc[0]
                                row += 1
                        runtime_tsfresh = tmp_runtime_tsfresh      
                    elif "single" in filename:
                        len_light_patterns = runtime_tsfresh.keys()
                        sampling_periods = runtime_tsfresh[len_light_patterns[0]].keys()
                        tmp_runtime_tsfresh = pandas.DataFrame(
                            columns=runtime_tsfresh[len_light_patterns[0]][sampling_periods[0]][0].columns)
                        row = 0
                        for len_light_pattern in len_light_patterns:
                            for sampling_period in sampling_periods:
                                runtime_data = runtime_tsfresh[len_light_pattern][sampling_period]
                                for entry in runtime_data:
                                    assert len(entry) == 1
                                    tmp_runtime_tsfresh.loc[row] = entry.loc[0]
                                    row += 1
                        runtime_tsfresh = tmp_runtime_tsfresh
                
                # remove outlier, more than 3 three times std
                #runtime_tsfresh[numpy.abs(runtime_tsfresh - runtime_tsfresh.mean()) > 3 * runtime_tsfresh.std()] = numpy.nan
                median = runtime_tsfresh.median()
                feature_len = median.index.values
                relative_runtime = median.values / feature_len
                runtimes[labels[testbed]] = numpy.mean(relative_runtime)
                plot_data.append((labels[testbed], feature_len, median))
            
            nth_label = 5
            fig, ax = plt.subplots()
            markers = itertools.cycle(misc.markers)
            datalen = [numpy.where(numpy.diff(median) < -0.8)[0] for _, _, median in plot_data]
            datalen = [array[0]+1 for array in datalen if len(array) > 0]
            datalen = min(datalen) if len(datalen) > 0 else min([len(feature_len) for _, feature_len, _ in plot_data])
            #datalen = min([len(feature_len) for _, feature_len, _ in plot_data])
            
            for label, feature_len, median in plot_data:   
                ax.plot(feature_len[:datalen], median[:datalen], label=label, marker=markers.next(), markevery=nth_label)
            ax.grid()
            ax.set_ylabel("Runtime (s)")
            ax.set_xlabel("Number of features")
            feature_len = feature_len[:datalen]            
            xticks = feature_len[::nth_label]
            xticks = numpy.concatenate([xticks, [feature_len[-1]]])
            ax.set_xticks(xticks)
            ax.set_ylim(bottom=0)
            ax.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=3, mode="expand", borderaxespad=0.)
            fig.set_figwidth(fig.get_figwidth()*1.6)
            #plt.show()
            
            filepath = os.path.join(__location__,  "results", "feature-selection")
            filename = "tsfresh-features-only-runtime-" if "only" in filename else "tsfresh-features-runtime-"
            filename = filename + light_data_type + ".pdf"
            save_path = os.path.join(filepath,  filename)
            directory = os.path.dirname(save_path)
            if not os.path.exists(directory):
                os.makedirs(directory)
            plt.savefig(save_path, format="pdf", bbox_inches="tight")
            plt.close(fig)
            
            runtimes_ms = {key: value*1e3 for key, value in runtimes.items()}
            runtimes_ms = sorted(runtimes_ms.items(), key=lambda kv: kv[1])
            print("runtime type:", runtime_type, "feature type:", feature_type)
            for (testbed1_name, testbed1_runtime), (testbed2_name, testbed2_runtime) in misc.pairwise(runtimes_ms):
                print(testbed1_name, "relative runtime (ms):", round(testbed1_runtime, 2))
                print(testbed2_name, "relative runtime (ms):", round(testbed2_runtime, 2))
                print("ratio faster:", round(testbed2_runtime / testbed1_runtime, 2))
            print("---")
    
def analysis_basic_features(testbeds):
    
    def plot_feature_importance(importances, filename):
        feature_order = importances.median().sort_values().index.values
        importances = importances[feature_order]
        fig, ax = plt.subplots()
        importances.boxplot(ax=ax, vert=False, showfliers=False)
        ax.set_xlabel("Relative importance")
        filename = os.path.join(__location__, "results", "machine-learning", filename + ".pdf")
        directory = os.path.dirname(filename)
        if not os.path.exists(directory):
            os.makedirs(directory)
        plt.savefig(filename, format="pdf", bbox_inches="tight")
        #plt.show()
        plt.close(fig)
    
    print("importance basic features")
    for feature_type in ["combined", "single"]:
        feature_importances = list()
        for testbed in testbeds:
            file_regex = "*" + feature_type + "*basic*"
            path_basic_data = glob.glob(os.path.join(__location__, "raw-results", "feature-selection", testbed, file_regex))
            if len(path_basic_data) == 0:
                continue
            assert len(path_basic_data) == 1
            
            path_basic_data = path_basic_data[0]
            light_data_type = os.path.basename(path_basic_data).split("-")[0]
            basic_features_selection = DillSerializer(path_basic_data).deserialize()
            if "single" in light_data_type:
                len_light_patterns = basic_features_selection.keys()
                sampling_periods = basic_features_selection[len_light_patterns[0]].keys()
                classifiers = basic_features_selection[len_light_patterns[0]][sampling_periods[0]].keys()
                features = get_features(light_data_type, basic_features_selection, sampling_periods, classifiers, len_light_patterns)
            elif "combined" in light_data_type:
                sampling_periods = basic_features_selection.keys()
                classifiers = basic_features_selection[sampling_periods[0]].keys()
                features = get_features(light_data_type, basic_features_selection, sampling_periods, classifiers)
                 
            row = 0
            importances = pandas.DataFrame(columns=features)
            if "single" in light_data_type:
                for len_light_pattern, sampling_period, classifier in itertools.product(len_light_patterns, sampling_periods, classifiers):
                    feature_importance = basic_features_selection[len_light_pattern][sampling_period][classifier]
                    row, importances = add_data(importances, feature_importance, row)
            elif "combined" in light_data_type:
                for sampling_period, classifier in itertools.product(sampling_periods, classifiers):
                    feature_importance = basic_features_selection[sampling_period][classifier]
                    row, importances = add_data(importances, feature_importance, row)
                
            feature_importances.append(importances)
        
        df = pandas.concat(feature_importances)
        importance_median = df.median().sort_values(ascending=False)
        importance_median = zip(importance_median.index, importance_median.values)
        filename = "basic-features-importance-" + light_data_type
        plot_feature_importance(df, filename)
        
        print(light_data_type)
        print(classifiers)
        for (feature1_name, feature1_importance), (feature2_name, feature2_importance) in misc.pairwise(importance_median):
            print(feature1_name, "importance:", round(feature1_importance, 2))
            print(feature2_name, "importance:", round(feature2_importance, 2))
            print("ratio importance:", round(feature2_importance / feature1_importance, 2))
        print("---")
    
def analysis_feature_selection_runtime():
    testbeds = ["server", "vm", "bbb"]
    analysis_basic_features(testbeds)
    analysis_runtime_tsfresh(testbeds)
    save_features_to_extract = True
    testbed_features_to_extract = "vm"
    for testbed in testbeds:
        analysis_tsfresh_features(testbed, save_features_to_extract, testbed_features_to_extract)
    
def feature_selection(signal_pattern_combination,
                      range_len_light_pattern=range(2, 11, 2),
                      range_sampling_period=numpy.arange(0.03, 0.13, 0.01),
                      rounds=10):
    
    if "single" in signal_pattern_combination:
        print("single type")
        basic_features_selection = nested_dict(3, list)
        tsfresh_features_selection = nested_dict(2, list)
        runtime_tsfresh_features = nested_dict(2, list)
        raw_feature_data = nested_dict(2, list)
        tsfresh_extracted_features = nested_dict(2, list)
        for len_light_pattern in range_len_light_pattern:
            for sampling_period in range_sampling_period:
                for i in range(rounds):
                    print("round: ", i)
                    sampling_period = round(sampling_period, 2)
                    print("sampling period: ", sampling_period)
                    data = LightData(sampling_period, [len_light_pattern])
                    basic_features = BasicFeatures()
                    basic_features_extracted = basic_features.extract(data.X_basic)
                    for clf in Classifier:
                        if clf != Classifier.SVM:
                            features_relevance = basic_features.relevance(clf, basic_features_extracted, data.y_basic)
                            basic_features_selection[len_light_pattern][sampling_period][clf.name].append(features_relevance)
                    
                    tsfresh_features = TsFreshFeatures()
                    tsfresh_features_extracted, relevance_features = tsfresh_features.relevance(data.X_tsfresh, data.y_tsfresh)
                    selected_features = tsfresh_features.select_n_most_useful_features(relevance_features)
                    elapsed_times = tsfresh_features.performance_evaluation(tsfresh_features_extracted, relevance_features, data.X_tsfresh, rounds=1)
                    runtime_tsfresh_features[len_light_pattern][sampling_period].append(elapsed_times)
                    tsfresh_features_selection[len_light_pattern][sampling_period].append(selected_features)
                    
                    raw_feature_data[len_light_pattern][sampling_period].append(data)
                    tsfresh_extracted_features[len_light_pattern][sampling_period].append(tsfresh_features_extracted)
                    print("---")
            print("###")
    else:
        print("combined type")
        basic_features_selection = nested_dict(2, list)
        tsfresh_features_selection = nested_dict(1, list)
        runtime_tsfresh_features = nested_dict(1, list)
        raw_feature_data = nested_dict(1, list)
        tsfresh_extracted_features = nested_dict(1, list)
        for sampling_period in range_sampling_period:
            for i in range(rounds):
                print("round: ", i)
                sampling_period = round(sampling_period, 2)
                print("sampling period: ", sampling_period)
                data = LightData(sampling_period)
                basic_features = BasicFeatures()
                basic_features_extracted = basic_features.extract(data.X_basic)
                for clf in Classifier:
                    if clf != Classifier.SVM:
                        features_relevance = basic_features.relevance(clf, basic_features_extracted, data.y_basic)
                        basic_features_selection[sampling_period][clf.name].append(features_relevance)
                
                tsfresh_features = TsFreshFeatures()
                tsfresh_features_extracted, relevance_features = tsfresh_features.relevance(data.X_tsfresh, data.y_tsfresh)
                selected_features = tsfresh_features.select_n_most_useful_features(relevance_features)
                elapsed_times = tsfresh_features.performance_evaluation(tsfresh_features_extracted, relevance_features, data.X_tsfresh, rounds=1)
                runtime_tsfresh_features[sampling_period].append(elapsed_times)
                tsfresh_features_selection[sampling_period].append(selected_features)
                
                raw_feature_data[sampling_period].append(data)
                tsfresh_extracted_features[sampling_period].append(tsfresh_features_extracted)
                print("---")
    
    path_feature_selection = os.path.join(__location__, "raw-results", "feature-selection")
    DillSerializer(os.path.join(path_feature_selection, signal_pattern_combination + "-runtime-tsfresh")).serialize(runtime_tsfresh_features)
    DillSerializer(os.path.join(path_feature_selection, signal_pattern_combination + "-basic")).serialize(basic_features_selection)
    DillSerializer(os.path.join(path_feature_selection, signal_pattern_combination + "-tsfresh")).serialize(tsfresh_features_selection)
    path_ml_train_data = os.path.join(__location__, "..", "online", "ml-train-data")
    DillSerializer(os.path.join(path_ml_train_data, signal_pattern_combination + "-raw-feature-data")).serialize(raw_feature_data)
    DillSerializer(os.path.join(path_ml_train_data, signal_pattern_combination + "-tsfresh-features-extracted")).serialize(tsfresh_extracted_features)
    
def tsfresh_performance_evaluation(single_light_pattern=False, range_len_light_pattern=range(2, 11, 2)):
    sampling_period = get_pattern_max_sampling_period()
    if single_light_pattern: # single light patterns
        elapsed_times = dict()
        for len_light_pattern in range_len_light_pattern:
            data = LightData(sampling_period, [len_light_pattern])
            tsfresh_features = TsFreshFeatures()
            features_extracted, relevance_features = tsfresh_features.relevance(data.X_tsfresh, data.y_tsfresh)
            elapsed_time = tsfresh_features.performance_evaluation(features_extracted, relevance_features, data.X_tsfresh)
            elapsed_times[len_light_pattern] = elapsed_time
        filename = os.path.join(__location__, "raw-results", "feature-selection", "single-light-patterns-only-runtime-tsfresh")
        DillSerializer(filename).serialize(elapsed_times)      
    else: # combined light patterns
        data = LightData(sampling_period)
        tsfresh_features = TsFreshFeatures()
        features_extracted, relevance_features = tsfresh_features.relevance(data.X_tsfresh, data.y_tsfresh)
        elapsed_time = tsfresh_features.performance_evaluation(features_extracted, relevance_features, data.X_tsfresh)
        filename = os.path.join(__location__, "raw-results", "feature-selection", "combined-light-patterns-only-runtime-tsfresh")
        DillSerializer(filename).serialize(elapsed_time)

def analysis_runtime_tsfresh_selected_features(evaluate):
    data_path = os.path.join(__location__, "raw-results", "feature-selection", "tsfresh-selected-features-runtime")
    if evaluate:
        features_path = glob.glob(os.path.join(
            __location__, "raw-results", "feature-selection", "tsfresh-*-to-be-extracted-*"))
        features_path = sorted(features_path, key=lambda entry: int(os.path.basename(entry).split("-")[-1]))
        tsfresh_features = TsFreshFeatures()
        runtime = nested_dict(2, dict)
        for len_light_pattern in [2, 4, 6, 8, 10]:
            light_signal, light_signal_time = light_analysis.load_light_pattern(len_light_pattern)
            coupling_data_provider = CouplingDataProvider(light_signal, light_signal_time, None, None)
            sampling_period_coupling = get_pattern_max_sampling_period()
            light_signal, _ = coupling_data_provider.get_light_data(sampling_period_coupling)
            print("len light pattern: ", len_light_pattern)
            print("sampling period: ", sampling_period_coupling)
            print("len sample: ", len(light_signal))
            for feature_path in features_path:
                num_features = int(os.path.basename(feature_path).split("-")[-1])
                print("num features: ", num_features)
                features_to_extract = DillSerializer(feature_path).deserialize()
                start = time.time()
                X = tsfresh_features.extract_selected_features(light_signal, features_to_extract, True)
                end = time.time()
                print("feature shape: ", X.shape)
                assert num_features == X.shape[1]
                runtime[len_light_pattern][num_features] = end-start
                print("duration: ", end-start)
            DillSerializer(data_path).serialize(runtime)
    else:
        runtime = DillSerializer(data_path).deserialize()
        runtime_per_num_feature = defaultdict(list)
        len_light_patterns, num_features = get_all_keys(runtime)
        for len_light_pattern, num_feature in itertools.product(len_light_patterns, num_features):
            runtime_per_num_feature[num_feature].append(runtime[len_light_pattern][num_feature])
        fig, ax = plt.subplots()
        num_features = sorted(runtime_per_num_feature.keys())
        median_runtime = [numpy.median(runtime_per_num_feature[num_feature]) for num_feature in num_features]
        nth_feature = 10
        ax.text(nth_feature+0.3, median_runtime[nth_feature]+0.015, round(median_runtime[nth_feature],3))
        ax.axvline(nth_feature, linestyle="--", color="black")
        ax.plot(num_features, median_runtime, label="Virtual Machine", marker="o", color="#1f77b4")
        ax.set_ylabel("Runtime (s)")
        ax.set_xlabel("Number of features")
        ax.set_xticks(num_features[::4] + [num_features[-1]])
        ax.grid()
        ax.set_ylim(bottom=0, top=0.3)
        ax.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=1, mode="expand", borderaxespad=0.)
        filepath = os.path.join(
            __location__, "results", "feature-selection", "vm", "tsfresh-features-selected-runtime.pdf")
        result_path = os.path.dirname(filepath)
        if not os.path.exists(result_path):
            os.makedirs(result_path)
        fig.savefig(filepath, format="pdf", bbox_inches="tight")
        #plt.show()
        plt.close(fig)

def offline_test_ml_model(path_ml_offline_evaluation):
        
    def filter_params(param_grid):
        filtered_params = list()
        for param in param_grid:
            if param["num clients"] - param["num reject clients"] >= 2:
                filtered_params.append(param)
        return filtered_params
    
    testbed = "vm"
    path_ml_train_data = os.path.join(__location__, "..", "online", "ml-train-data", testbed)
    combined_raw_feature_data = glob.glob(os.path.join(path_ml_train_data, "combined-*-raw-feature-data"))[0]
    combined_raw_feature_data = DillSerializer(combined_raw_feature_data).deserialize()
    tsfresh_features_to_extract_selected = os.path.join(__location__, "..", "online", "tsfresh-features-to-be-extracted")
    tsfresh_features_to_extract_selected = DillSerializer(tsfresh_features_to_extract_selected).deserialize()
    sampling_periods = sorted(combined_raw_feature_data.keys())
    
    num_clients = 10
    num_reject_clients = range(num_clients-1)
    num_clients = range(2, num_clients+1)
    len_light_patterns = range(2, 11, 2)
    param_grid = ParameterGrid({"num clients": num_clients,
                                "num reject clients": num_reject_clients,
                                "len light pattern": len_light_patterns})
    sampling_period_coupling = get_pattern_max_sampling_period()
    filtered_params = filter_params(param_grid)
    results = nested_dict(5, list)
    for i, param in enumerate(filtered_params):
        print("Param: {0}/{1}".format(i+1, len(filtered_params)))
        clients = dict()
        groundtruth_accept_clients = list()
        groundtruth_reject_clients = list()
        light_signal, light_signal_time = light_analysis.load_light_pattern(param["len light pattern"])
        coupling_data_provider = CouplingDataProvider(light_signal, light_signal_time, None, None)
        for _ in range(param["num clients"]-param["num reject clients"]): # accept client    
            mac = create_random_mac()
            client = Client()
            client.light_signal, _ = coupling_data_provider.get_light_data(sampling_period_coupling)
            clients[mac] = client
            groundtruth_accept_clients.append(mac)
        
        #light_signal_random, light_signal_random_time = light_analysis.load_random_light_signal()
        #coupling_data_provider = CouplingDataProvider(light_signal_random, light_signal_random_time, None, None)
        
        datalen = len(light_signal)
        mean = light_signal.mean()
        std = light_signal.std()
        noise = numpy.random.normal(mean, std, datalen)
        coupling_data_provider = CouplingDataProvider(noise, light_signal_time, None, None)
        for _ in range(param["num reject clients"]): # reject client
            mac = create_random_mac()
            client = Client()
            client.light_signal, _ = coupling_data_provider.get_light_data(sampling_period_coupling)
            clients[mac] = client
            groundtruth_reject_clients.append(mac)
        
        for clf in Classifier:
            for sampling_period in sampling_periods:
                print("Classifier: ", clf)
                print("Sampling period: ", sampling_period)
                tsfresh_features = TsFreshFeatures()
                X_tsfresh = combined_raw_feature_data[sampling_period][0].X_tsfresh
                y_tsfresh = combined_raw_feature_data[sampling_period][0].y_tsfresh
                print("X: ", X_tsfresh.shape)
                print("X samples: ", len(X_tsfresh.id.unique()))
                print("y: ", y_tsfresh.shape)
                print("Extract features ...")
                X_selected_features = tsfresh_features.extract_selected_features(
                    X_tsfresh, tsfresh_features_to_extract_selected)
                print("X selected: ", X_selected_features.shape)    
                print("y: ", y_tsfresh.shape)
                
                print("Coupling simulation ...")
                ml_model = Classifier.get_clf(clf)
                print("Class 1: ", len(y_tsfresh[y_tsfresh == 1]))
                print("Class 0: ", len(y_tsfresh[y_tsfresh == 0]))
                ml_model = ml_model.fit(X_selected_features, y_tsfresh)
                accept_clients = set()
                reject_clients = set()
                for client_mac in clients.keys():
                    client_light_data = clients[client_mac].light_signal
                    feature = tsfresh_features.extract_selected_features(
                        client_light_data, tsfresh_features_to_extract_selected, True)
                    print("Feature shape: ", feature.shape)
                    result = ml_model.predict(feature)
                    if result == 1.0:
                        accept_clients.add(client_mac)
                    else:
                        reject_clients.add(client_mac)
                accept_clients = list(accept_clients)
                reject_clients = list(reject_clients)
                mac_mapping = {key:value for key, value in zip(range(len(clients)), clients.keys())}
                result = StaticCouplingResult(accept_clients, reject_clients,
                                              groundtruth_accept_clients, groundtruth_reject_clients,
                                              None, mac_mapping)
                results[param["num clients"]][param["num reject clients"]] \
                    [param["len light pattern"]][clf.name][sampling_period].append(result)
                print("accept:")
                print("result:", accept_clients)
                print("ground truth: ", groundtruth_accept_clients)
                print(result.accuracy_accept)
                print("reject:")
                print("result: ", reject_clients)
                print("ground truth: ", groundtruth_reject_clients)
                print(result.accuracy_reject)
                print("ML cross validation ...")
                ml_model = Classifier.get_clf(clf)
                scores = cross_val_score(ml_model, X_selected_features, y_tsfresh, cv=10, n_jobs=-1)
                print("Scores: ", scores)
                print("------------------------------------------------------")
        DillSerializer(path_ml_offline_evaluation).serialize(results)

def offline_analysis_ml_model(path_ml_offline_evaluation):
    evaluation_data = DillSerializer(path_ml_offline_evaluation).deserialize()
    num_clients, num_reject_clients, len_light_patterns, \
        classifiers, sampling_periods = misc.get_all_keys(evaluation_data)
    analysis_result = nested_dict(2, list)
    for num_client, num_reject_client, len_light_pattern, classifier, sampling_period in itertools.product(
                            num_clients, num_reject_clients, len_light_patterns, classifiers, sampling_periods):
        results = evaluation_data[num_client][num_reject_client][len_light_pattern][classifier][sampling_period]
        if len(results) > 0:
            analysis_result[classifier][sampling_period].extend(results)
    
    print("Num clients: ", num_clients)
    print("Num reject clients: ", num_reject_clients)
    print("Len light patterns: ", len_light_patterns)
    print("Classifiers: ", classifiers)
    print("Sampling periods: ", sampling_periods)
    
    for classifier in classifiers:
        results = analysis_result[classifier]
        sub_results = list()
        for sampling_period in sampling_periods:
            accuracy = [entry.accuracy_accept for entry in results[sampling_period]] + \
                [entry.accuracy_reject for entry in results[sampling_period]]
            precision = [entry.precision_accept for entry in results[sampling_period]] + \
                [entry.precision_reject for entry in results[sampling_period]]
            recall = [entry.recall_accept for entry in results[sampling_period]] + \
                [entry.recall_reject for entry in results[sampling_period]]
            f1 = [entry.f1_accept for entry in results[sampling_period]] + \
                [entry.f1_reject for entry in results[sampling_period]]
            
            entry = [numpy.mean(accuracy), numpy.mean(precision), numpy.mean(recall), numpy.mean(f1)]
            entry = [round(value, 2) for value in entry]
            sub_results.append(entry)
        
        fig, ax = plt.subplots()
        ax.imshow(sub_results, cmap="Greens", aspect="auto", interpolation="nearest", vmin=0, vmax=1.4)
        ax.set_ylabel("Sampling period (ms)")
        ytickpos = numpy.arange(len(sampling_periods))
        ax.set_yticks(ytickpos)
        ax.set_yticklabels([int(sampling_period*1e3) for sampling_period in sampling_periods])
        xticks = ["Accuracy", "Precision", "Recall", "F1-score"]
        xtickpos = range(len(xticks))
        ax.set_xticks(xtickpos)
        ax.set_xticklabels(xticks, rotation=20, ha="right")
        for i in range(len(sub_results)):
            for j in range(len(sub_results[0])):
                ax.text(j, i, sub_results[i][j], ha="center", va="center")
        ticks = [start+((end-start)/2) for start, end in misc.pairwise(xtickpos)]
        ax.set_xticks(ticks, minor=True)
        ticks = [start+((end-start)/2) for start, end in misc.pairwise(ytickpos)]
        ax.set_yticks(ticks, minor=True)
        ax.grid(which='minor', color="black")
        filepath = os.path.join(
            __location__, "results", "machine-learning", "vm", "ml-param-" + classifier.lower() + ".pdf")
        result_path = os.path.dirname(filepath)
        if not os.path.exists(result_path):
            os.makedirs(result_path)
        fig.savefig(filepath, format="pdf", bbox_inches="tight")
        #plt.show()
        plt.close(fig)
    
def test_feature_selection_runtime(evaluate):
    if evaluate:
        path_ml_train_data = os.path.join(__location__, "..", "online", "ml-train-data")    
        if not os.path.exists(path_ml_train_data):
            os.makedirs(path_ml_train_data)
        path_feature_selection = os.path.join(__location__, "raw-results", "feature-selection")
        if not os.path.exists(path_feature_selection):
            os.makedirs(path_feature_selection)
        tsfresh_performance_evaluation(False)
        tsfresh_performance_evaluation(True)
        for signal_pattern_combination in ["single-light-patterns", "combined-light-patterns"]: 
            feature_selection(signal_pattern_combination)
    else:
        analysis_feature_selection_runtime()
    
def test_offline_ml_parameters(evaluate):
    path_ml_offline_evaluation = os.path.join(
        __location__, "raw-results", "machine-learning", "ml-offline-evaluation")
    if evaluate:
        offline_test_ml_model(path_ml_offline_evaluation)
    else:
        offline_analysis_ml_model(path_ml_offline_evaluation)

def test_runtime_tsfresh_selected_features(evaluate):
    if evaluate:
        testbed = "vm"
        save_features_to_extract = True
        analysis_tsfresh_features(testbed, save_features_to_extract, testbed)
    analysis_runtime_tsfresh_selected_features(evaluate)

'''
Important: runs only with python2 due to saved simulation results
Data is incompatible to python3, the rest is compatible with python3
'''
def main():
    evaluate = False
    test_feature_selection_runtime(evaluate)
    test_runtime_tsfresh_selected_features(evaluate)
    test_offline_ml_parameters(evaluate)
    
if __name__ == "__main__":
    main()
    