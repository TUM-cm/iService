from __future__ import division
import re
import os
import json
import glob
import numpy
import random
import itertools
from collections import Counter
import matplotlib.pyplot as plt
from coupling.log import analysis
import coupling.utils.misc as misc
from collections import defaultdict
from sklearn import model_selection
from utils.nested_dict import nested_dict
from utils.serializer import DillSerializer
from coupling.relay_attack import clustering
from coupling.device_grouping.online.machine_learning_features import Classifier
from coupling.device_grouping.online.machine_learning_features import BasicFeatures
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

testbeds = ["non-line-of-sight", "line-of-sight"]
measurement_directory = os.path.join(__location__, "measurements")
result_base_path = os.path.join(__location__, "results")

'''
emu03: ssh haus@emu03.cm.in.tum.de
nohup python -u -m coupling.relay_attack.analysis &

sudo apt-get install build-essential cython python-numpy
pip install -e "git+https://github.com/perrygeo/jenks.git#egg=jenks"

pyclustering error 
Error:
from . import _imaging as core
ImportError: DLL load failed: The specified procedure could not be found.

Solution:
pip uninstall pillow
pip install pillow==4.0.0
'''

def filter_merge_dicts(dicts):
    super_dict = dict()
    for single_dict in dicts:
        for key, values in single_dict.items():
            super_dict[int(key)] = [value for value in values if value != 0]
    return super_dict

def join_metric_results(metric, y, metrics):
    metric_per_class = dict()
    for pos_id in numpy.unique(y):
        result = 0
        if pos_id in metrics.index:
            result = metrics[metric][pos_id]
        metric_per_class[pos_id] = result    
    return metric_per_class
                
def mean_dict(dicts):    
    sums = Counter()
    counters = Counter()
    for itemset in dicts:
        sums.update(itemset)
        counters.update(itemset.keys())
    return {x: float(sums[x])/counters[x] for x in sums.keys()}

def join_measurements_per_room(measurements):
    pos_per_room = {1:[1, 2], 2:[4, 5, 9, 10, 14, 15, 19, 20],
                    3:[24, 25, 29, 30], 4:[34, 35, 39, 40],
                    5:[44, 45, 49, 50], 6:[54, 55, 59, 60],
                    7:[64, 65, 69, 70], 8:[74, 75, 79, 80],
                    9:[84, 85, 89, 90], 10:[94, 95, 99, 100],
                    11:[104, 105, 109, 110], 12:[114, 115, 117, 118],
                    13:[107, 108, 112, 113], 14:[92, 93, 97, 98, 102, 103],
                    15:[77, 78, 82, 83, 87, 88], 16:[67, 68, 72, 73],
                    17:[57, 58, 62, 63], 18:[47, 48, 52, 53],
                    19:[37, 38, 42, 43], 20:[27, 28, 32, 33],
                    21:[17, 18, 22, 23], 22:[12, 13], 23:[7, 8]}
    
    measurements_per_room = dict()
    for room, positions in pos_per_room.items():
        joined_measurements = list(itertools.chain(*[measurements[pos] for pos in positions if pos in measurements]))
        if len(joined_measurements) > 0:
            measurements_per_room[room] = joined_measurements
        else:
            print("empty room measurements:", room)
    return measurements_per_room

# latency: 0 - max
def load_latency_data(measurement_directory, testbed):
    latency_path_dataset = glob.glob(os.path.join(measurement_directory, testbed + "*latency*"))
    latency_measurements = [DillSerializer(fpath).deserialize() for fpath in latency_path_dataset]
    return filter_merge_dicts(latency_measurements)

# rssi: -100 (min) - -55 (max)
def load_rssi_data(measurement_directory, testbed):
    signal_strength_path_dataset = glob.glob(os.path.join(measurement_directory, testbed + "*signal-strength*"))
    signal_strength_measurements = [json.load(open(fpath)) for fpath in signal_strength_path_dataset]
    return filter_merge_dicts(signal_strength_measurements)

def run_spatial_granularity(clustering_filepath, prediction_filepath, measurement_groups):
    
    def create_features(data, data_type, measurement_type):
            
        def create_Xy(description, data, create_basic_features, create_selected_features):
            X = list()
            y = list()
            basic_features = BasicFeatures()
            for measurement_id, values in data.items():
                if create_basic_features:
                    X.append(basic_features.compute(values))
                    y.append(measurement_id)
                elif create_selected_features:
                    X.append(basic_features.compute_selected_features(values))
                    y.append(measurement_id)
                else:
                    X.extend(values)
                    y.extend([measurement_id] * len(values))
            assert len(X) == len(y)
            return (description, numpy.asarray(X), numpy.asarray(y))
        
        raw = create_Xy(data_type + "-raw-" +  measurement_type, data, False, False)
        basic = create_Xy(data_type + "-basic-" + measurement_type, data, True, False)
        selected = create_Xy(data_type + "-selected-" + measurement_type, data, False, True)
        return [raw, basic, selected]
    
    def run_clustering(results, testbed, data_description, data, measurement_groups, rounds=10):
        print("run clustering")
        min_clusters = 2
        true_clusters = measurement_groups[testbed]
        max_upper_bound = int(true_clusters*1.2)
        max_clusters = max_upper_bound if len(data) > max_upper_bound else len(data)-1
        print("data:", data.shape)
        for i in range(rounds):
            print("round:", i)
            results[testbed][data_description]["dbscan"].append(clustering.dbscan(data))
            results[testbed][data_description]["birch"].append(clustering.birch(data))
            # Not usable due to memory error
            #results[testbed][data_description]["affinity"].append(clustering.affinity_propagation(data))
            results[testbed][data_description]["jenks"].append(clustering.jenks_natural_breaks(data, min_clusters, max_clusters))
            results[testbed][data_description]["kmeans"].append(clustering.kmeans_silhouette(data, min_clusters, max_clusters))
            results[testbed][data_description]["hierarchical"].append(clustering.agglomerative_hierarchical(data, min_clusters, max_clusters))
            results[testbed][data_description]["kernel density"].append(clustering.kernel_density_estimation(data))
            results[testbed][data_description]["xmeans"].append(clustering.xmeans_clustering(data, min_clusters, max_clusters))
            results[testbed][data_description]["gauss bic"].append(clustering.gaussian_mixture_bic(data, min_clusters, max_clusters))
            results[testbed][data_description]["gauss aic"].append(clustering.gaussian_mixture_aic(data, min_clusters, max_clusters))
            results[testbed][data_description]["mean shift"].append(clustering.mean_shift(data))
    
    def run_prediction(results, testbed, data_description, X, y):
        print("run prediction")
        clfs = [Classifier.ExtraTreesClassifier, 
                #Classifier.GradientBoostingClassifier, # too slow
                Classifier.SVM,
                Classifier.RandomForest,
                Classifier.NaiveBayes,
                Classifier.AdaBoost]
        for clf_type in clfs:
            print(clf_type.name)
            per_class = defaultdict(list)
            over_class = defaultdict(list)
            kfold = model_selection.KFold(n_splits=10)
            for train_index, test_index in kfold.split(X):
                print("train:", len(train_index))
                print("test:", len(test_index))
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]
                clf = Classifier.get_clf(clf_type)
                clf = clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)
                
                metrics = analysis.calculate_metrics(y_test, y_pred)
                #from sklearn.metrics import classification_report
                #print(classification_report(y_test, y_pred))
                per_class["accuracy"].append(join_metric_results("accuracy", y, metrics))
                per_class["precision"].append(join_metric_results("precision", y, metrics))
                per_class["recall"].append(join_metric_results("recall", y, metrics))
                per_class["f1"].append(join_metric_results("f1-score", y, metrics))
                
                over_class["accuracy"].append(accuracy_score(y_test, y_pred))
                over_class["precision"].append(precision_score(y_test, y_pred, average="micro"))
                over_class["recall"].append(recall_score(y_test, y_pred, average="micro"))
                over_class["f1"].append(f1_score(y_test, y_pred, average="micro"))
            
            print("mean metrics")
            for metric, metric_results in per_class.items():
                # per round and per class and hence summarize of rounds
                results[testbed][data_description][clf_type.name]["per class"][metric] = mean_dict(metric_results)
            for metric, metric_results in over_class.items():
                # plot over rounds and mean summarizes entire information
                results[testbed][data_description][clf_type.name]["over class"][metric] = metric_results #numpy.mean(metric_results)
    
    def detect_spatial_granularity(clustering_results, prediction_results, testbed, features, measurement_groups):
        for data_description, X, y in features:
            print(data_description)
            if X.ndim == 1:
                X = X.reshape(-1, 1)
            run_prediction(prediction_results, testbed, data_description, X, y)
            # Too time consuming, limited result only showing with what accuracy you can differentiate spatial areas
            #run_clustering(clustering_results, testbed, data_description, X, measurement_groups)
    
    def random_select(data, datalen=60):
        return {key: numpy.random.choice(values, size=datalen) for key, values in data.items()}
    
    from sklearn import warnings
    #from sklearn.exceptions import ConvergenceWarning
    from sklearn.exceptions import UndefinedMetricWarning
    warnings.filterwarnings(action="ignore", category=UndefinedMetricWarning)
    #warnings.filterwarnings(action="ignore", category=ConvergenceWarning) # only for clustering
    
    clustering_results = nested_dict(3, list)
    prediction_results = nested_dict(4, dict)
    for testbed in testbeds:
        print(testbed)
        result_directory = os.path.join(result_base_path, testbed)
        if not os.path.exists(result_directory):
            os.makedirs(result_directory)
        
        features = list()
        latency_point_measurements = load_latency_data(measurement_directory, testbed)
        rssi_point_measurements = load_rssi_data(measurement_directory, testbed)
        latency_point_measurements = random_select(latency_point_measurements)
        rssi_point_measurements = random_select(rssi_point_measurements)
        
        features.extend(create_features(latency_point_measurements, "latency", "position"))
        features.extend(create_features(rssi_point_measurements, "rssi", "position"))
        if "non-line-of-sight" in testbed:
            latency_room_measurements = join_measurements_per_room(latency_point_measurements)
            rssi_room_measurements = join_measurements_per_room(rssi_point_measurements)
            latency_room_measurements = random_select(latency_room_measurements)
            rssi_room_measurements = random_select(rssi_room_measurements)
            features.extend(create_features(latency_room_measurements, "latency", "room"))
            features.extend(create_features(rssi_room_measurements, "rssi", "room"))
        
        detect_spatial_granularity(
            clustering_results, prediction_results, testbed, features, measurement_groups)
    
    print("save ...")
    #DillSerializer(clustering_filepath).serialize(clustering_results)
    DillSerializer(prediction_filepath).serialize(prediction_results)
    
def analysis_spatial_granularity(clustering_result_filepath, prediction_result_filepath, measurement_groups, plot_format):
    
    def find_best_clustering(clustering_result_filepath, measurement_groups, best_nth=10):
        if not os.path.exists(clustering_result_filepath):
            return
        print("find best clustering")
        results = DillSerializer(clustering_result_filepath).deserialize()
        testbeds, data_descriptions, clusterings = misc.get_all_keys(results)
        for testbed in testbeds:
            print(testbed)
            clustering_results = dict()
            true_clusters = measurement_groups[testbed]
            for data_description in data_descriptions:
                for clustering in clusterings:
                    clusters = results[testbed][data_description][clustering]    
                    y_pred = clusters
                    y_true = len(clusters) * [true_clusters]
                    key = data_description + "-" + clustering
                    clustering_results[key] = accuracy_score(y_true, y_pred)
            clustering_results = sorted(clustering_results.items(), key=lambda entry: entry[1], reverse=True)
            for combination, accuracy in clustering_results[:best_nth]:
                print(combination, ":", accuracy)
    
    def find_best_prediction(prediction_result_filepath, plot_format, nth_best=1):
        
        def plot_best_result(data_type, clf_feature_combination, metrics, threshold):
            feature = clf_feature_combination.split(":")[0]
            clf = clf_feature_combination.split(":")[1]
            fig, ax = plt.subplots()
            marker_cycle = itertools.cycle(misc.markers)
            for metric in metrics:
                result = results[testbed][feature][clf][analysis_type][metric]
                if isinstance(result, dict):
                    y = list(result.values())
                    x = list(result.keys())
                else:
                    y = result
                    x = list(range(1, len(result)+1))
                y = numpy.asarray(y)
                x = numpy.asarray(x)
                over_threshold = len(x[numpy.where(y >= threshold)])
                print("over threshold:", round(100 * (over_threshold / len(x)),2))
                ax.plot(x, y, label=metric.capitalize(), marker=next(marker_cycle))  
            ax.grid()
            ax.set_ylim(bottom=-0.05, top=1.05)        
            ax.set_ylabel(metrics[0].capitalize())
            
            if "per class" in analysis_type:
                if "room" in feature:
                    ax.set_xlabel("Room")
                else:
                    ax.set_xlabel("Measurement point")
            else:
                ax.set_xlabel("Round")
            
            if len(x) <= 10:
                scaling_factor = 1
            elif len(x) > 10 and len(x) <= 30:
                scaling_factor = 10
            else:
                scaling_factor = 20
            
            ax.set_xticks(x[::scaling_factor])
            if "non-line-of-sight" not in testbed:
                ax.set_xticklabels(range(1, len(x)+1)[::scaling_factor])
            ax.axhline(threshold, label="Threshold", color="red", linestyle="--")
            
            #ax.legend()
            #ax.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=4, mode="expand", borderaxespad=0.)
            result_directory = os.path.join(result_base_path, testbed)
            if not os.path.exists(result_directory):
                os.makedirs(result_directory)
            
            #figLegend = plt.figure()
            #plt.figlegend(*ax.get_legend_handles_labels(), loc="center", ncol=4)
            #filename = "spatial-granularity-legend" + plot_format
            #filepath = os.path.join(result_directory, filename)
            #figLegend.savefig(filepath, format=plot_format, bbox_inches="tight")
            
            filename = "spatial-granularity-" + data_type + "-" + analysis_type.replace(" ", "-") # + "-" + clf.lower() + "-" + feature
            filepath = os.path.join(result_directory, filename + "." + plot_format)
            fig.savefig(filepath, format=plot_format, bbox_inches="tight")
            #plt.show()
            plt.close(fig)
            #plt.close(figLegend)
        
        if not os.path.exists(prediction_result_filepath):
            return
        print("find best prediction")
        results = DillSerializer(prediction_result_filepath).deserialize()
        testbeds, features, clfs, analysis_types, _ = misc.get_all_keys(results, 5)
        latency_features = [feature for feature in features if "latency-" in feature]
        rssi_features = [feature for feature in features if "rssi-" in feature]
        metrics = ["accuracy"]
        for testbed in testbeds:
            for analysis_type in analysis_types:
                for subfeatures in [rssi_features, latency_features]:
                    analysis_results = dict()
                    for feature in subfeatures:
                        # room features not available for line-of-sight
                        if "room" in feature and "line-of-sight" in testbed:
                            continue
                        for clf in clfs:
                            metric_results = list()
                            for metric in metrics:
                                result = results[testbed][feature][clf][analysis_type][metric]
                                if "per class" in analysis_type:
                                    metric_results.append(sum(result.values()) / len(result)) # best over all rounds / points
                                else:
                                    metric_results.append(numpy.mean(result)) # best over all rounds
                            key = feature + ":" + clf
                            analysis_results[key] = numpy.mean(metric_results)
                    
                    threshold = 0.75
                    analysis_results = sorted(analysis_results.items(), key=lambda kv: kv[1], reverse=True)
                    data_type = feature.split("-")[0]
                    print(testbed)
                    print(analysis_type)
                    for clf_feature_combination, result in analysis_results[:nth_best]:
                        print(clf_feature_combination, ":", round(result, 3))
                        plot_best_result(data_type, clf_feature_combination, metrics, threshold)        
                print("---")
    
    find_best_prediction(prediction_result_filepath, plot_format)    
    find_best_clustering(clustering_result_filepath, measurement_groups)
    
def relay_attack_simulation(
        simulation_results, baseline_results, testbed, measurements, data_type, measurement_type, non_attacker_position):
    
    def calculate_metrics(y_true, y_pred, results, testbed, attacker_position, attacker_ratio, clf, data_type, feature_description):    
        results[testbed][attacker_position][attacker_ratio][clf][data_type][feature_description]["accuracy"].append(accuracy_score(y_true, y_pred))
        #results[testbed][attacker_position][attacker_ratio][clf][data_type][feature_description]["precision"].append(precision_score(y_true, y_pred))
        #results[testbed][attacker_position][attacker_ratio][clf][data_type][feature_description]["recall"].append(recall_score(y_true, y_pred))
        #results[testbed][attacker_position][attacker_ratio][clf][data_type][feature_description]["f1"].append(f1_score(y_true, y_pred))
    
    def create_features(measurements, labels, data_type, measurement_type):
        
        def create_X(description, measurements, labels, create_basic_features, create_selected_features):
            X = list()
            y = list()
            basic_features = BasicFeatures()
            for measurement, label in zip(measurements, labels):
                if create_basic_features:
                    X.append(basic_features.compute(measurement))
                    y.append(label)
                elif create_selected_features:
                    #X.append(basic_features.compute_selected_features(measurement))
                    #y.append(label)
                    X.append(numpy.median(measurement))
                    y.append(label)
                else:
                    X.append(measurement) # n-dim
                    y.append(label)
                    #X.extend(measurement) # 1-dim
                    #y.extend([label] * len(measurement))
            return (description, numpy.asarray(X), numpy.asarray(y))
        
        raw = create_X(data_type + "-raw-" +  measurement_type, measurements, labels, False, False)
        basic = create_X(data_type + "-basic-" + measurement_type, measurements, labels, True, False)
        selected = create_X(data_type + "-selected-" + measurement_type, measurements, labels, False, True)
        return [raw, basic, selected]
    
    datalen = 40
    num_users = 20
    measurement_positions = measurements.keys()
    non_attacker_measurements = measurements[non_attacker_position]
    attacker_positions = set(measurement_positions).difference([non_attacker_position])
    for attacker_position in attacker_positions:
        print("attacker position:", attacker_position)
        attacker_measurements = measurements[attacker_position]
        for attacker_ratio in numpy.round(numpy.arange(0, 1.1, 0.2), 1):
            num_attacker = int(num_users * attacker_ratio)
            num_non_attacker = int(num_users - num_attacker)
            print("attacker ratio:", attacker_ratio)
            print("attacker:", num_attacker)
            print("non attacker:", num_non_attacker)
            
            #datalen = min(len(non_attacker_measurements), len(attacker_measurements))
            non_attacker_data = numpy.asarray([numpy.random.choice(non_attacker_measurements, size=datalen) for _ in range(num_non_attacker)])
            attacker_data = numpy.asarray([numpy.random.choice(attacker_measurements, size=datalen) for _ in range(num_attacker)])
            #non_attacker_data = numpy.asarray([non_attacker_measurements[:datalen] for _ in range(num_non_attacker)])
            #attacker_data = numpy.asarray([attacker_measurements[:datalen] for _ in range(num_attacker)])
            non_attacker_data = non_attacker_data.reshape(-1, datalen)
            attacker_data = attacker_data.reshape(-1, datalen)
            
            if "latency" in data_type and num_attacker > 0:
                # attacker: assumption for latency, two times the latency compared to non-attacker
                scaling_factor = numpy.round(numpy.random.uniform(1.9, 4, len(attacker_data)), 1).reshape(-1,1)
                attacker_data *= scaling_factor
            relay_attack_data = numpy.vstack([non_attacker_data, attacker_data])
            # 0 = non-attacker, 1 = attacker
            attack_labels = numpy.concatenate([numpy.zeros(num_non_attacker), numpy.ones(num_attacker)])
            clfs = [Classifier.ExtraTreesClassifier,
                    #Classifier.GradientBoostingClassifier, # min two classes
                    #Classifier.SVM, # min to classes
                    Classifier.RandomForest,
                    Classifier.NaiveBayes,
                    Classifier.AdaBoost]
            for feature, X, y in create_features(relay_attack_data, attack_labels, data_type, measurement_type):
                if X.ndim == 1:
                    X = X.reshape(-1, 1)
                print("feature:", feature)
                print("X:", X.shape)
                print("y:", y.shape)
                for clf_type in clfs:
                    print(clf_type)
                    kfold = model_selection.KFold(n_splits=10, shuffle=True)
                    for train_index, test_index in kfold.split(X):
                        #print("train:", len(train_index))
                        #print("test:", len(test_index))
                        X_train, X_test = X[train_index], X[test_index]
                        y_train, y_test = y[train_index], y[test_index]
                        clf = Classifier.get_clf(clf_type)
                        clf = clf.fit(X_train, y_train)
                        y_pred = clf.predict(X_test)
                        calculate_metrics(
                            y_test, y_pred, simulation_results, testbed, attacker_position, attacker_ratio, clf_type.name, data_type, feature)
                        
                        y_pred = numpy.array([random.getrandbits(1) for _ in range(len(y_test))])
                        calculate_metrics(
                            y_test, y_pred, baseline_results, testbed, attacker_position, attacker_ratio, clf_type.name, data_type, feature)


def line(p1, p2):
    A = (p1[1] - p2[1])
    B = (p2[0] - p1[0])
    C = (p1[0]*p2[1] - p2[0]*p1[1])
    return A, B, -C

def intersection(L1, L2):
    D  = L1[0] * L2[1] - L1[1] * L2[0]
    Dx = L1[2] * L2[1] - L1[1] * L2[2]
    Dy = L1[0] * L2[2] - L1[2] * L2[0]
    if D != 0:
        x = Dx / D
        y = Dy / D
        return x,y
    else:
        return False

def analysis_relay_attack_simulation(simulation_resultpath, baseline_resultpath, plot_format):
    
    def adapt_label(clf, feature):
        feature = feature.replace("room", "")
        feature = feature.replace("position", "")
        feature = feature.strip().title()
        feature = feature.replace("Rssi", "RSSI")
        feature = feature.replace("Selected", "Median")
        feature_parts = feature.split("-")
        feature = "-".join([feature_parts[1], feature_parts[0], feature_parts[-1]])
        clf = re.sub(r"(\w)([A-Z])", r"\1 \2", clf)
        clf = clf.replace("Classifier", "")
        return clf, feature        
    
    def get_details(parameter_combination):
        parameters = parameter_combination.split(":")
        testbed = parameters[0]
        data_type = parameters[1]
        clf = parameters[2]
        feature = parameters[3]
        return testbed, data_type, clf, feature
    
    def plot_attacker_ratio(best_clf_feature, simulation_results, baseline_results, threshold=0.75):
        print("attacker ratio")
        fig, ax = plt.subplots()
        marker_cycle = itertools.cycle(misc.markers)
        
        coord_offset = 0.02
        second_intersection = False
        
        for parameter_combination, _ in best_clf_feature:
            testbed, data_type, clf, feature = get_details(parameter_combination)
            for metric in metrics:
                x = list()
                y_simulation = list()
                y_baseline = list()
                for attacker_ratio in attacker_ratios:
                    x.append(attacker_ratio)
                    simulation_result = list()
                    baseline_result = list()
                    for attacker_position in attacker_positions:
                        result = numpy.mean(
                            simulation_results[testbed][attacker_position][attacker_ratio][clf][data_type][feature][metric])
                        simulation_result.append(result)
                        result = numpy.mean(
                            baseline_results[testbed][attacker_position][attacker_ratio][clf][data_type][feature][metric])
                        baseline_result.append(result)
                    y_simulation.append(numpy.mean(simulation_result))
                    y_baseline.append(numpy.mean(baseline_result))
                clf, feature = adapt_label(clf, feature)
                label = clf + " - " + feature.replace("-", " ")
                x = numpy.asarray(x)
                y_simulation = numpy.asarray(y_simulation)
                marker = next(marker_cycle)
                ax.axhline(threshold, color="red", linestyle="--")
                ax.plot(x, y_simulation, label=label, marker=marker)[0]
                ax.plot(x, y_baseline, label=label + " - Baseline", marker=marker, linestyle="--")
                
                print(label)
                print("simulation:", round(numpy.mean(y_simulation),3))
                print("baseline:", round(numpy.mean(y_baseline),3))
                for A, B in misc.pairwise(zip(x, y_simulation)):
                    C = (A[0], threshold)
                    D = (B[0], threshold)
                    from shapely.geometry import LineString
                    from shapely.geometry import Point
                    line1 = LineString([A, B])
                    line2 = LineString([C, D])
                    point = line1.intersection(line2)
                    if isinstance(point, Point):
                        print("ratio intersection:", round(point.x,2))
                        xshift = 0
                        if second_intersection:
                            xshift = -0.18
                        ax.text(point.x + coord_offset + xshift, point.y + coord_offset, "%.2f" % point.x)
                        ax.axvline(point.x, color="black", linestyle="--")
                        ax.plot(point.x, point.y, marker="o", color="red")
                        second_intersection = True
        ax.grid()
        ax.set_ylabel("Accuracy")
        ax.set_ylim(bottom=-0.05, top=1.05)
        ax.set_xlabel("Attacker ratio among clients")
        base_filename = "attacker-ratio"
        fig_legend = plt.figure()
        plt.figlegend(*ax.get_legend_handles_labels(), loc="center", ncol=2)
        filename = "attacker-legend." + plot_format
        filepath = os.path.join(result_directory, filename)
        fig_legend.savefig(filepath, format=plot_format, bbox_inches="tight")
        
        filename = base_filename + "." + plot_format
        filepath = os.path.join(result_directory, filename)
        fig.savefig(filepath, format=plot_format, bbox_inches="tight")
        #plt.show()
        plt.close(fig)
        plt.close(fig_legend)
    
    def plot_attacker_position(best_clf_feature, simulation_results, baseline_results):
        print("attacker position")
        fig, ax = plt.subplots()
        marker_cycle = itertools.cycle(misc.markers)
        for parameter_combination, _ in best_clf_feature:
            testbed, data_type, clf, feature = get_details(parameter_combination)
            for metric in metrics:
                x = list()
                y_simulation = list()
                y_baseline = list()
                for attacker_position in attacker_positions:
                    x.append(attacker_position)
                    simulation_result = list()
                    baseline_result = list()
                    for attacker_ratio in attacker_ratios:
                        result = simulation_results[testbed][attacker_position][attacker_ratio][clf][data_type][feature][metric]
                        simulation_result.append(numpy.mean(result))
                        result = baseline_results[testbed][attacker_position][attacker_ratio][clf][data_type][feature][metric]
                        baseline_result.append(numpy.mean(result))
                    y_simulation.append(numpy.mean(simulation_result))
                    y_baseline.append(numpy.mean(baseline_result))
                
                if "non-line-of-sight" in testbed:
                    room_labeling = {14:11, 15:12, 16:13, 17:14, 18:15, 19:16, 20:17, 21:18, 22:19, 23:20}
                    x_relabeled = list()
                    for room in x:
                        if room in room_labeling:
                            x_relabeled.append(room_labeling[room])
                        else:
                            x_relabeled.append(room)
                    x = x_relabeled
                
                clf, feature = adapt_label(clf, feature)
                label = clf + " - " + feature.replace("-", " ")
                x_sim, y_simulation = zip(*sorted(zip(x, y_simulation)))
                x_base, y_baseline = zip(*sorted(zip(x, y_baseline)))
                marker = next(marker_cycle)
                ax.plot(x_sim, y_simulation, label=label, marker=marker)
                ax.plot(x_base, y_baseline, label=label + " - Baseline", marker=marker, linestyle="--")
                print(label)
                print("simulation:", round(numpy.mean(y_simulation),3))
                print("baseline:", round(numpy.mean(y_baseline),3))
        ax.grid()
        ax.set_ylabel("Accuracy")
        ax.set_ylim(bottom=-0.05, top=1.05)
        xlabel = "Attacker's room" if "non-line-of-sight" in testbed else "Attacker's position"
        ax.set_xlabel(xlabel)
        ax.set_xticks(x_sim[::3])
        if "non-line-of-sight" not in testbed:
            ax.set_xticklabels(range(1, len(x_sim)+1)[::3])
        
        base_filename = "attacker-position"
        fig_legend = plt.figure()
        plt.figlegend(*ax.get_legend_handles_labels(), loc="center", ncol=2)
        filename = "attacker-legend." + plot_format
        filepath = os.path.join(result_directory, filename)
        fig_legend.savefig(filepath, format=plot_format, bbox_inches="tight")
        
        filename = base_filename + "." + plot_format
        filepath = os.path.join(result_directory, filename)
        fig.savefig(filepath, format=plot_format, bbox_inches="tight")
        #plt.show()
        plt.close(fig)
        plt.close(fig_legend)
    
    def find_best_performing_clf_feature(
            simulation_results, testbed, data_type, clfs, features, attacker_ratios, attacker_positions, metrics):
        
        results = dict()
        for clf in clfs:
            for feature in features:
                if ("room" in feature and "line-of-sight" == testbed) or \
                ("position" in feature and "non-line-of-sight" == testbed) or \
                data_type not in feature:
                    continue
                metric_results = list()
                for attacker_ratio in attacker_ratios: # [0, 1]
                    for attacker_position in attacker_positions:
                        for metric in metrics: # take average over all of them
                            result = numpy.asarray(
                                simulation_results[testbed][attacker_position][attacker_ratio][clf][data_type][feature][metric])
                            # remove over fitting due to sparse data
                            if len(numpy.where(result == 1)[0]) != len(result):
                                metric_results.append(numpy.mean(result))
                            #metric_results.append(numpy.mean(result))
                if len(metric_results) > 0:
                    key = testbed + ":" + data_type + ":" + clf + ":" + feature
                    results[key] = numpy.mean(metric_results)
        return sorted(results.items(), key=lambda kv: kv[1], reverse=True)
    
    simulation_results = DillSerializer(simulation_resultpath).deserialize()
    baseline_results = DillSerializer(baseline_resultpath).deserialize()
    testbeds, _, attacker_ratios, clfs, data_types, features, metrics = misc.get_all_keys(simulation_results)
    for testbed in testbeds:
        overall_results = list()
        attacker_positions = list(simulation_results[testbed].keys())
        result_directory = os.path.join(result_base_path, testbed)
        for data_type in data_types:
            best_clf_feature = find_best_performing_clf_feature(
                simulation_results, testbed, data_type, clfs, features, attacker_ratios, attacker_positions, metrics)
            best_clf_feature = [best_clf_feature[-1]]
            overall_results.extend(best_clf_feature)
        print(testbed)
        plot_attacker_ratio(overall_results, simulation_results, baseline_results)
        plot_attacker_position(overall_results, simulation_results, baseline_results)
        print("---")
    
def run_relay_attack_simulation(simulation_resultpath, baseline_resultpath):
    simulation_results = nested_dict(7, list)
    baseline_results = nested_dict(7, list)
    non_attacker_positions = {"line-of-sight": 3, "non-line-of-sight": 20} # receiver nearest to sender
    for testbed in testbeds:
        print(testbed)
        latency_measurements = load_latency_data(measurement_directory, testbed)
        rssi_measurements = load_rssi_data(measurement_directory, testbed)
        measurement_type = "position"
        if "non-line-of-sight" in testbed:
            latency_measurements = join_measurements_per_room(latency_measurements)
            rssi_measurements = join_measurements_per_room(rssi_measurements)
            measurement_type = "room"
        non_attacker_position = non_attacker_positions[testbed]
        relay_attack_simulation(
            simulation_results, baseline_results, testbed, latency_measurements, "latency", measurement_type, non_attacker_position)
        relay_attack_simulation(
            simulation_results, baseline_results, testbed, rssi_measurements, "rssi", measurement_type, non_attacker_position)
    print("save ...")
    DillSerializer(simulation_resultpath).serialize(simulation_results)
    DillSerializer(baseline_resultpath).serialize(baseline_results)
    
def main():
    run_experiments = False
    
    plot_format = "pdf"
    raw_result_directory =   os.path.join(__location__, "raw-results")
    if not os.path.exists(raw_result_directory):
        os.makedirs(raw_result_directory)
    
    measurement_groups = { "non-line-of-sight": 20, "line-of-sight": 25 }
    spatial_granularity_clustering_resultpath = os.path.join(__location__, "raw-results", "spatial-granularity-clustering")
    spatial_granularity_prediction_resultpath = os.path.join(__location__, "raw-results", "spatial-granularity-prediction")
    print("### spatial granularity")
    if run_experiments:
        run_spatial_granularity(
            spatial_granularity_clustering_resultpath, spatial_granularity_prediction_resultpath, measurement_groups)
    else:
        analysis_spatial_granularity(
            spatial_granularity_clustering_resultpath, spatial_granularity_prediction_resultpath, measurement_groups, plot_format)
    
    print("### relay attack")
    relay_attack_baseline_resultpath = os.path.join(raw_result_directory, "relay-attack-baseline")
    relay_attack_simulation_resultpath = os.path.join(raw_result_directory, "relay-attack-simulation")
    if run_experiments:
        run_relay_attack_simulation(
            relay_attack_simulation_resultpath, relay_attack_baseline_resultpath)
    else:
        analysis_relay_attack_simulation(
            relay_attack_simulation_resultpath, relay_attack_baseline_resultpath, plot_format)
 
if __name__ == "__main__":
    main()
    