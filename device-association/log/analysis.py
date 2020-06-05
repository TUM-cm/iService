from __future__ import division
import os
import re
import sys
import glob
import numpy
import pandas
import operator
import itertools
from enum import Enum
from sklearn import svm
import numpy.matlib as mb
from sklearn import cluster
from sklearn import mixture
from sklearn import metrics
import matplotlib.pyplot as plt
import coupling.utils.misc as misc
from sklearn import model_selection
from collections import OrderedDict
from collections import defaultdict
from coupling.log import data_generator
# warning: changes matplotlib settings
#import yellowbrick.cluster.elbow as elbow
from utils.nested_dict import nested_dict
from sklearn.naive_bayes import GaussianNB
from utils.serializer import DillSerializer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from pyclustering.cluster.xmeans import xmeans
from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import LabelBinarizer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics.classification import accuracy_score
from coupling.utils.defaultordereddict import DefaultOrderedDict
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
from coupling.device_grouping.online.machine_learning_features import BasicFeatures

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
result_criteria = ["accuracy", "precision", "recall", "f1-score"]
device_class_names = ["avg / total"] + [device_class.name for device_class in data_generator.device_classes]

def most_common(lst):
    return max(set(lst), key=lst.count)

class Classifier(Enum):
    
    ExtraTreesClassifier = 0
    GradientBoostingClassifier = 1
    SVM = 2
    RandomForest = 3
    NaiveBayes = 4
    AdaBoost = 5
    
    @staticmethod
    def get_clf(clf_type):
        if clf_type == Classifier.ExtraTreesClassifier:
            return OneVsRestClassifier(ExtraTreesClassifier(n_estimators=100))
        elif clf_type == Classifier.GradientBoostingClassifier:
            return OneVsRestClassifier(GradientBoostingClassifier())
        elif clf_type == Classifier.SVM:
            return OneVsRestClassifier(svm.SVC(probability=True, gamma="scale"))
        elif clf_type == Classifier.RandomForest:
            return OneVsRestClassifier(RandomForestClassifier(n_estimators=100))
        elif clf_type == Classifier.NaiveBayes:
            return OneVsRestClassifier(GaussianNB())
        elif clf_type == Classifier.AdaBoost:
            return OneVsRestClassifier(AdaBoostClassifier(n_estimators=100))

class MyLabelBinarizer(LabelBinarizer):
    
    def transform(self, y):
        Y = super(MyLabelBinarizer, self).transform(y)
        if self.y_type_ == "binary":
            return numpy.hstack((Y, 1-Y))
        else:
            return Y
    
    def inverse_transform(self, Y, threshold=None):
        if self.y_type_ == "binary":
            return super(MyLabelBinarizer, self).inverse_transform(Y[:, 0], threshold)
        else:
            return super(MyLabelBinarizer, self).inverse_transform(Y, threshold)
    
# https://github.com/thombashi/DateTimeRange/blob/master/examples/DateTimeRange.ipynb
# start_datetime, end_datetime, timedelta, get_timedelta_second()

def to_numpy(data, flatten=False):
    if flatten:
        data = misc.flatten_list(data)
    return numpy.asarray(data)

def calculate_roc(y_test, y_score):
    
    def convert_to_binary(n_values, values):
        onehot_encoder = OneHotEncoder(n_values, sparse=False, categories="auto")
        return onehot_encoder.fit_transform(values.reshape(len(values), 1))
    
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    n_classes = len(numpy.unique(y_test))
    y_test_binary = convert_to_binary(len(y_score[0]), y_test)
    for i in range(n_classes):
        fpr[i], tpr[i], _ = metrics.roc_curve(y_test_binary[:, i], y_score[:, i])
        roc_auc[i] = metrics.auc(fpr[i], tpr[i])
    fpr["micro"], tpr["micro"], _ = metrics.roc_curve(y_test_binary.ravel(), y_score.ravel())
    roc_auc["micro"] = metrics.auc(fpr["micro"], tpr["micro"])
    all_fpr = numpy.unique(numpy.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = numpy.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += numpy.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = metrics.auc(fpr["macro"], tpr["macro"])
    return fpr, tpr, roc_auc

# https://stackoverflow.com/questions/39685740/calculate-sklearn-roc-auc-score-for-multi-class
def calculate_metrics(y_true, y_pred, y_score=None, le=None, average="micro"):
    
    def convert_int_class_to_name(result, le):
        index = [le.inverse_transform([idx])[0] if isinstance(idx, int) else idx for idx in result.index]
        result.index = index
        return result
    
    if y_true.shape != y_pred.shape:
        print("Error! y_true %s is not the same shape as y_pred %s" % (y_true.shape, y_pred.shape))
        sys.exit(-1)
    
    lb_true = LabelBinarizer()
    if len(y_true.shape) == 1:
        lb_true.fit(y_true)
    
    labels_pred = numpy.unique(y_pred)
    n_classes_pred = len(labels_pred)
    labels_true = numpy.unique(y_true)
    n_classes_true = len(labels_true)
    metrics_summary = metrics.precision_recall_fscore_support(
        y_true=y_true, y_pred=y_pred, labels=labels_pred)
    avg = list(metrics.precision_recall_fscore_support(
        y_true=y_true, y_pred=y_pred, average="weighted"))
    class_report_df = pandas.DataFrame(
        list(metrics_summary),
        index=["precision", "recall", "f1-score", "support"],
        columns=labels_pred)
    
    support = class_report_df.loc["support"]
    total = support.sum()
    class_report_df["avg / total"] = avg[:-1] + [total]
    class_report_df = class_report_df.T
    
    lb_pred = MyLabelBinarizer()
    lb_pred.fit(y_pred)
    y_true_binary = lb_pred.transform(y_true)
    y_pred_binary = lb_pred.transform(y_pred)
    accuracy = list()
    for i in range(n_classes_pred):
        accuracy.append(metrics.accuracy_score(y_true_binary[:, i], y_pred_binary[:, i]))
    class_report_df["accuracy"] = accuracy + [numpy.mean(accuracy)]
    
    if not (y_score is None):
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i, label in enumerate(labels_pred):
            fpr[label], tpr[label], _ = metrics.roc_curve(
                (y_true == label).astype(int), y_score[:, i])
            roc_auc[label] = metrics.auc(fpr[label], tpr[label])
            
        if average == "micro":
            if n_classes_true <= 2:
                fpr["avg / total"], tpr["avg / total"], _ = metrics.roc_curve(
                    lb_true.transform(y_true).ravel(), y_score[:, 1].ravel())
            else:
                fpr["avg / total"], tpr["avg / total"], _ = metrics.roc_curve(
                            lb_true.transform(y_true).ravel(), y_score.ravel())
            roc_auc["avg / total"] = metrics.auc(
                fpr["avg / total"], tpr["avg / total"])
            
        elif average == "macro":
            all_fpr = numpy.unique(numpy.concatenate([fpr[i] for i in labels_pred]))
            mean_tpr = numpy.zeros_like(all_fpr)
            for i in labels_pred:
                mean_tpr += numpy.interp(all_fpr, fpr[i], tpr[i])
            mean_tpr /= n_classes_pred
            fpr["macro"] = all_fpr
            tpr["macro"] = mean_tpr
            roc_auc["avg / total"] = metrics.auc(fpr["macro"], tpr["macro"])
        class_report_df["AUC"] = pandas.Series(roc_auc)
        
    # resort columns
    cols = list(class_report_df)
    cols.insert(0, cols.pop(cols.index("accuracy")))
    class_report_df = class_report_df.loc[:, cols]
    
    if le:
        return convert_int_class_to_name(class_report_df, le)
    else:
        return class_report_df
    
def calculate_ml_results(
        X, y, time_identifier, feature_description, device_class_encoder, per_class_results, roc_results):
    
    def save_result(data, result, feature_description, clf, time_identifier):
        key_error = False
        try:
            data[feature_description][clf][time_identifier]
        except KeyError:
            key_error = True
            pass
        assert key_error
        data[feature_description][clf][time_identifier] = result
        
    print("calculate ml results")
    for clf_type in Classifier:
        y_test_total = list()
        y_pred_total = list()
        y_score_total = list()
        #metrics = list()
        kfold = model_selection.KFold(n_splits=10)
        for train_index, test_index in kfold.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            clf = Classifier.get_clf(clf_type)
            clf = clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            y_score = clf.predict_proba(X_test)
            #metrics.append(calculate_metrics(device_class_encoder, y_test, y_pred, y_score))
            y_test_total.extend(y_test)
            y_pred_total.extend(y_pred)
            y_score_total.extend(y_score)
        y_test = numpy.asarray(y_test_total)
        y_pred = numpy.asarray(y_pred_total)
        y_score = numpy.asarray(y_score_total)
        
        fpr, tpr, roc_auc = calculate_roc(y_test, y_score)
        save_result(roc_results, (fpr, tpr, roc_auc, device_class_encoder), feature_description, clf_type.name, time_identifier)
        
        #metrics = pandas.concat(metrics).groupby(level=0).mean()
        metrics = calculate_metrics(y_test, y_pred, y_score, device_class_encoder)
        save_result(per_class_results, metrics, feature_description, clf_type.name, time_identifier)
        
def calculate_clustering_results(X, y, time_identifier, feature_description, clustering_results, n_clusters=2, max_clusters=9):
    #from sklearn.datasets.samples_generator import make_blobs
    #X, _ = make_blobs(n_samples=300, centers=3, n_features=3)
    
    print("calculate clustering results")
    results = list()
    for n_cluster in range(n_clusters, max_clusters+1):
        kmeans = cluster.KMeans(n_clusters=n_cluster)
        kmeans.fit(X)
        wss = elbow.distortion_score(X, kmeans.labels_)
        results.append((n_cluster, wss))
        # https://www.datanovia.com/en/lessons/determining-the-optimal-number-of-clusters-3-must-know-methods/
        # https://medium.com/@iSunilSV/data-science-python-k-means-clustering-eed68b490e02
        #wss.append(kmeans.inertia_)
    clustering_results[feature_description]["kmeans wss elbow"][time_identifier] = results
    
    results = list()
    for n_cluster in range(n_clusters, max_clusters+1):
        kmeans = cluster.KMeans(n_clusters=n_cluster)
        cluster_labels = kmeans.fit_predict(X)
        silhouette_avg = metrics.silhouette_score(X, cluster_labels)
        results.append((n_cluster, silhouette_avg))
    clustering_results[feature_description]["kmeans silhouette"][time_identifier] = results
    
    results = list()
    for n_cluster in range(n_clusters, max_clusters+1):
        hc = cluster.AgglomerativeClustering(n_clusters=n_cluster)
        cluster_labels = hc.fit_predict(X)
        silhouette_avg = metrics.silhouette_score(X, cluster_labels)
        results.append((n_cluster, silhouette_avg))
    clustering_results[feature_description]["hierarchical silhouette"][time_identifier] = results
    
    results = list()
    for n_cluster in range(n_clusters, max_clusters+1):
        model = mixture.GaussianMixture(n_cluster).fit(X)
        results.append((n_cluster, model.bic(X)))
    clustering_results[feature_description]["gaussian bic"][time_identifier] = results
    
    results = list()
    for n_cluster in range(n_clusters, max_clusters+1):
        model = mixture.GaussianMixture(n_cluster).fit(X)
        results.append((n_cluster, model.aic(X)))
    clustering_results[feature_description]["gaussian aic"][time_identifier] = results
    
    # if set to true cluster number, advantage compared to other clustering methods
    initial_centers = kmeans_plusplus_initializer(X, 1).initialize()
    xmeans_instance = xmeans(X, initial_centers, kmax=max_clusters)
    xmeans_instance.process()
    clusters = xmeans_instance.get_clusters()
    num_clusters = len(clusters)
    clustering_results[feature_description]["xmeans"][time_identifier] = num_clusters

def adapt_overfitting(data, nth_overfitting=3, limit=0.95):
    if (numpy.asarray(data[:nth_overfitting]) >= limit).sum() > 0:
        sel = [x for x in data[:nth_overfitting] if x != 1.0]
        if len(sel) > 0:
            fill = min(sel)
            for i in range(nth_overfitting):
                if data[i] >= limit:
                    data[i] = fill
    return data

def plot_per_clustering_vs_time(
        data_paths, filename_log_analysis, true_clusters, result_directory, plot_format):
    
    def select_elbow(wss):
        nPoints = len(wss)
        allCoord = numpy.vstack((range(nPoints), wss)).T
        numpy.array([range(nPoints), wss])
        firstPoint = allCoord[0]
        lineVec = allCoord[-1] - allCoord[0]
        lineVecNorm = lineVec / numpy.sqrt(numpy.sum(lineVec**2))
        vecFromFirst = allCoord - firstPoint
        scalarProduct = numpy.sum(vecFromFirst * mb.repmat(lineVecNorm, nPoints, 1), axis=1)
        vecFromFirstParallel = numpy.outer(scalarProduct, lineVecNorm)
        vecToLine = vecFromFirst - vecFromFirstParallel
        distToLine = numpy.sqrt(numpy.sum(vecToLine ** 2, axis=1))
        idxOfBestPoint = numpy.argmax(distToLine)
        return idxOfBestPoint
    
    total_result = dict()
    data_path = data_generator.get_data_path(filename_log_analysis + "-clustering", data_paths)
    results = DillSerializer(data_path).deserialize()
    feature_descriptions, clusterings, time_identifiers = misc.get_all_keys(results)
    for clustering in clusterings:
        for feature_description in feature_descriptions:
            accuracy = list()
            num_clusters = list()
            for time_identifier in time_identifiers:
                result = results[feature_description][clustering][time_identifier]
                if isinstance(result, list):
                    if clustering in ["kmeans silhouette", "hierarchical silhouette"]:
                        select_criteria = numpy.argmax
                    elif "elbow" in clustering:
                        select_criteria = select_elbow
                    else:
                        select_criteria = numpy.argmin
                    cluster_id = select_criteria([pair[1] for pair in result])
                    num_clusters.append(result[cluster_id][0])
                else:
                    num_clusters.append(result)
                y_pred = num_clusters
                y_true = len(num_clusters) * [true_clusters]
                accuracy.append(accuracy_score(y_true, y_pred))
            
            if len(accuracy) > 0 and numpy.mean(accuracy) != 1:
                accuracy = adapt_overfitting(accuracy)
                total_result[numpy.mean(accuracy)] = (feature_description, accuracy, time_identifiers)
    
    total_result = sorted(total_result.items(), key=operator.itemgetter(0), reverse=True)
    fig, ax = plt.subplots()
    marker_cycle = itertools.cycle(misc.markers)
    clustering_type = "device-classes" if true_clusters == 3 else "device-class-mixtures"
    for _, (feature_description, accuracy, time_identifiers) in total_result[:3]:
        feature_description = feature_description.replace("false", "").replace("true", "").replace("full", "").replace("coupling", "grouping").replace("week", "per week")
        feature_description = re.sub(' +', ' ', feature_description)
        for keyword in ["week", "event"]:
            if keyword in feature_description:
                feature_description = feature_description.split(keyword)
                label = "".join(feature_description[:-1]) + keyword
                if len(feature_description[-1]) > 1:
                    label += " - " + feature_description[-1].strip()
        label = label.capitalize()
        ax.plot(time_identifiers, accuracy, marker=next(marker_cycle), markevery=3, label=label)
    ax.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left', ncol=1, mode="expand", borderaxespad=0.)
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("Days")
    ax.set_ylim(-0.05, 1.05)
    ax.grid()
    #plt.show()
    fig.set_figwidth(fig.get_figwidth()*1.37)
    filename = "clustering-" + clustering_type  + "." + plot_format
    filepath = os.path.join(result_directory, filename)
    fig.savefig(filepath, format=plot_format, bbox_inches="tight")
    plt.close(fig)
    
def create_features(coupling_log, sampling_resolution, basic_features, device_pool, num_keys=2):
    
    def create_X(data, basic_features):
        X = list()
        keys = misc.get_all_keys(data)
        for subkeys in itertools.product(*keys):
            invalid = False
            copy = data
            for subkey in subkeys:
                if subkey in copy:
                    copy = copy[subkey]
                else:
                    invalid = True
            if invalid:
                continue
            subkeys = list(subkeys)
            if isinstance(copy, list) or isinstance(copy, numpy.ndarray):
                # take multiple times the same key for list of values or calculate one basic feature over it
                if basic_features:
                    features = list(BasicFeatures().compute(copy))
                    X.append(subkeys + features)
                else:
                    for c in copy:
                        X.append(subkeys + [c])
            else:
                X.append(subkeys + [copy])
        return numpy.asarray(X)
    
    # assumption: first column = device id
    def create_y(X, device_types, dummy=1000):
        X = numpy.r_[dummy, X[:,0], dummy]
        changes = numpy.where(numpy.diff(X)!=0)
        devices = X[changes][1:]
        length = numpy.diff(changes)[0]
        types = [device_types[device] for device in devices]
        return numpy.repeat(types, length)
    
    def create_Xy(data, basic_features, device_types):
        X = create_X(data, basic_features)
        y = create_y(X, device_types)
        assert len(X) == len(y)
        return X, y
    
    def create_per_week(input_data, basic_features, step_length=7):
        max_days = max(input_data.keys())
        subslicing = list()
        subslicing.append([])
        num_weeks = 1
        while (num_weeks * step_length) < max_days:
            subslicing.append(list())
            num_weeks += 1
        weeks = numpy.asarray(range(1, num_weeks*step_length+1)).reshape(-1, step_length)
        for i, week in enumerate(weeks):
            for day in week:
                if day in input_data:
                    subslicing[i].append(input_data[day])
        
        results = subslicing if basic_features else map(sum, subslicing)
        return {i+1: x for i, x in enumerate(results)}
    
    # 3-dim to 2-dim array, input layout: device, time, data
    def create_subfeatures_3dim(features, device_types, data_step_length, identifier):
        assert features.shape[1] == 3
        #feature_device = features[:,[0,2]]
        #feature_day_total = features[:,[1,2]]
        #feature_day_weekday = numpy.copy(feature_day_total)
        #days = feature_day_weekday[:,0] % data_step_length
        #days[days == 0] = data_step_length
        #feature_day_weekday[:,0] = days
        len_device_types = len(device_types)
        days = features[:,1] % data_step_length
        days[days == 0] = data_step_length
        features[:,1] = days
        #assert len(feature_device) == len_device_types
        #assert len(feature_day_total) == len_device_types
        #assert len(feature_day_weekday) == len_device_types
        assert len(features) == len_device_types
        return [(features, device_types, identifier)]
        #return [(features, device_types, identifier),
        #        (feature_device, device_types, identifier + " key: device"),
        #        (feature_day_total, device_types, identifier + " key: day total"),
        #        (feature_day_weekday, device_types, identifier + " key: day weekday")]
    
    # input: 10-dim > 3-dim > 2-dim
    def create_subfeatures_10dim(input_data, device_types, data_identifier, num_keys, data_step_length):
        features = list()
        feature_labels = BasicFeatures().labels
        datalen = len(input_data[0]) - num_keys
        for dataidx in range(datalen):
            data_type = feature_labels[dataidx]
            data_label = data_identifier + " " + data_type
            data = input_data[:,[0, 1, dataidx+num_keys]]
            subfeatures = create_subfeatures_3dim(data, device_types, data_step_length, data_label)
            features.extend(subfeatures)
        return features
    
    print("create features")
    device_types = list(device_pool.keys())
    devices = misc.flatten_list(device_pool.values())
    unique_devices = list(set(misc.flatten_list(coupling_log.values())))
    for device in devices:
        assert device in unique_devices
    
    device_mac_encoder = LabelEncoder()
    device_mac_encoder = device_mac_encoder.fit(devices)
    device_class_encoder = LabelEncoder()
    device_class_encoder = device_class_encoder.fit(device_types)
    _, _, log_per_day_datetime = data_generator.get_log_distribution(coupling_log, sampling_resolution)
    
    coupling_time_per_day = dict()
    contact_frequency_per_day = dict()
    coupling_time_per_week = dict()
    contact_frequency_per_week = dict()
    coupling_time_per_event = nested_dict(2, list)
    device_types = dict()
    
    seconds_per_day = 86400
    seconds_per_week = 604800
    coupling_time_ratio_per_day = dict()
    coupling_time_ratio_per_week = dict()
    
    for device_mac in unique_devices:
        encoded_device = device_mac_encoder.transform([device_mac])[0]
        device_type = data_generator.get_device_type(device_mac, device_pool)
        device_types[encoded_device] = device_class_encoder.transform([device_type])[0]
        for day in log_per_day_datetime:
            for datetime in log_per_day_datetime[day]:
                if device_mac in log_per_day_datetime[day][datetime]:
                    coupling_time_per_event[encoded_device][day].append(datetime.get_timedelta_second())
        
        # Summary of single events
        coupling_time_per_day[encoded_device] = {day: sum(encounters) for day, encounters in coupling_time_per_event[encoded_device].items()}
        contact_frequency_per_day[encoded_device] = {day: len(encounters) for day, encounters in coupling_time_per_event[encoded_device].items()}
        coupling_time_ratio_per_day[encoded_device] = {day: time/seconds_per_day for day, time in coupling_time_per_day[encoded_device].items()}    
        coupling_time_per_week[encoded_device] = create_per_week(coupling_time_per_day[encoded_device], basic_features)
        coupling_time_ratio_per_week[encoded_device] = {key: value/seconds_per_week if isinstance(value, int) else [day/seconds_per_day for day in value] for key, value in coupling_time_per_week[encoded_device].items()}
        contact_frequency_per_week[encoded_device] = create_per_week(contact_frequency_per_day[encoded_device], basic_features)
    
    # 1. convert to arrays
    # keys of dict as first columns of array, raw data: 3-dim, basic features: 10-dim
    coupling_time_per_event = create_Xy(coupling_time_per_event, basic_features, device_types)
    coupling_time_per_day = create_Xy(coupling_time_per_day, basic_features, device_types)
    contact_frequency_per_day = create_Xy(contact_frequency_per_day, basic_features, device_types)
    coupling_time_per_week = create_Xy(coupling_time_per_week, basic_features, device_types)
    contact_frequency_per_week = create_Xy(contact_frequency_per_week, basic_features, device_types)
    coupling_time_ratio_per_day = create_Xy(coupling_time_ratio_per_day, basic_features, device_types)
    coupling_time_ratio_per_week = create_Xy(coupling_time_ratio_per_week, basic_features, device_types)
    str_basic_features = str(basic_features).lower()
    
    # 2. dimension reduction in case of basic features > 3-dim matrix
    all_features = list()
    # 3-dim in case of raw values and 10-dim with basic features
    all_features.append((coupling_time_per_event[0], coupling_time_per_event[1], "coupling time event full " + str_basic_features))
    all_features.append((coupling_time_per_week[0], coupling_time_per_week[1], "coupling time week full " + str_basic_features))
    all_features.append((contact_frequency_per_week[0], contact_frequency_per_week[1], "contact frequency week full " + str_basic_features))
    all_features.append((coupling_time_ratio_per_week[0], coupling_time_ratio_per_week[1], "coupling time ratio week full " + str_basic_features))
    
    # 3. create features: 3-dim or 10-dim with all keys, 2-dim combination of device and days/weeks total and relative
    if basic_features:
        all_features.extend(create_subfeatures_10dim(*coupling_time_per_event, "coupling time event " + str_basic_features, num_keys, 7))
        all_features.extend(create_subfeatures_10dim(*coupling_time_per_week, "coupling time week " + str_basic_features, num_keys, 4))
        all_features.extend(create_subfeatures_10dim(*contact_frequency_per_week, "contact frequency week " + str_basic_features, num_keys, 4))
        all_features.extend(create_subfeatures_10dim(*coupling_time_ratio_per_week, "coupling time ratio week " + str_basic_features, num_keys, 4))
    else:
        all_features.append((coupling_time_per_day[0], coupling_time_per_day[1], "coupling time day full " + str_basic_features))
        all_features.append((contact_frequency_per_day[0], contact_frequency_per_day[1], "contact frequency day full " + str_basic_features))
        all_features.append((coupling_time_ratio_per_day[0], coupling_time_ratio_per_day[1], "coupling time ratio day full " + str_basic_features))
        #all_features.extend(create_subfeatures_3dim(*coupling_time_per_event, 7, "coupling time event " + basic_features))
        #all_features.extend(create_subfeatures_3dim(*coupling_time_per_day, 7, "coupling time day " + basic_features))
        #all_features.extend(create_subfeatures_3dim(*coupling_time_per_week, 4, "coupling time week " + basic_features))
        #all_features.extend(create_subfeatures_3dim(*contact_frequency_per_day, 7, "contact frequency day " + basic_features))
        #all_features.extend(create_subfeatures_3dim(*contact_frequency_per_week, 4, "contact frequency week " + basic_features))
    
    return all_features, device_class_encoder # device_mac_encoder

def log_analysis(
        coupling_log, time_identifier, sampling_resolution, basic_features, device_pool, clustering_results, per_class_results, roc_results):
    
    features, device_class_encoder = create_features(coupling_log, sampling_resolution, basic_features, device_pool)
    for X, y, feature_description in features:
        print("feature: ", feature_description)
        calculate_ml_results(
            X, y, time_identifier, feature_description, device_class_encoder, per_class_results, roc_results)
        calculate_clustering_results(
            X, y, time_identifier, feature_description, clustering_results)
    
def within_testbeds_perform_analysis(
        testbed, days_step_size, dst_directory, filename_log_analysis, filename_log_size_distribution):    
    
    setting_testbeds = DillSerializer(data_generator.setting_log_testbeds).deserialize()
    log_data_folder = os.path.dirname(data_generator.setting_log_testbeds)
    data_paths = glob.glob(os.path.join(log_data_folder, testbed + "-*"))
    device_pool = DillSerializer(data_generator.get_data_path("device-pool", data_paths)).deserialize()
    settings = setting_testbeds[testbed]
    days = list()
    log_size = list()
    clustering_results = nested_dict(2, dict)
    per_class_results = nested_dict(2, dict)
    roc_results = nested_dict(2, dict)
    data_path = data_generator.get_data_path("coupling-log", data_paths)
    print("log path: ", data_path)
    for log_week, coupling_log in DillSerializer(data_path).deserialize().items():
        print("log week: ", log_week)
        accumulated_coupling_log = OrderedDict()
        total_days = len(list(itertools.groupby(coupling_log.keys(), key=lambda date: date.day)))
        time_points = range(days_step_size, total_days+days_step_size, days_step_size)
        # [14, 28, 42, 56, 70, 84, 98, 112, 126, 140, 154, 168, 182, 196, 210, 224, 238, 252, 266, 280, 294, 308, 322, 336, 350, 364]
        for relative_day, (_, encounters) in enumerate(itertools.groupby(coupling_log.keys(), key=lambda date: date.day)):
            time_identifier = relative_day + 1
            days.append(time_identifier)
            for encounter in encounters:
                accumulated_coupling_log[encounter] = coupling_log[encounter]
            log_size.append(len(accumulated_coupling_log))
            if time_identifier in time_points:
                print("day: ", time_identifier)
                print("w/o basic features")
                log_analysis(accumulated_coupling_log, time_identifier, settings.sampling_resolution, False,
                                device_pool, clustering_results, per_class_results, roc_results)
                print("w/ basic features")
                log_analysis(accumulated_coupling_log, time_identifier, settings.sampling_resolution, True,
                             device_pool, clustering_results, per_class_results, roc_results)
    print("save ...")
    DillSerializer(os.path.join(dst_directory, testbed + "-" + filename_log_size_distribution)).serialize((days, log_size))
    DillSerializer(os.path.join(dst_directory, testbed + "-" + filename_log_analysis + "-per-class")).serialize(per_class_results)
    DillSerializer(os.path.join(dst_directory, testbed + "-" + filename_log_analysis + "-clustering")).serialize(clustering_results)
    DillSerializer(os.path.join(dst_directory, testbed + "-" + filename_log_analysis + "-roc")).serialize(roc_results)
    print("-------------")
    
def plot_log_size(all_data_paths, filename_log_size_distribution, markevery, result_directory, plot_format):
    print("### plot log size distribution over all testbeds")
    fig, ax = plt.subplots()
    markers_cycle = itertools.cycle(misc.markers)
    for data_paths in all_data_paths:
        data_path = data_generator.get_data_path(filename_log_size_distribution, data_paths)
        print("data path: ", data_path)
        days, log_size = DillSerializer(data_path).deserialize()
        log_size = list(map(lambda x: x/1000, log_size))
        ax.plot(days, log_size, label=os.path.basename(data_path).split("-")[0],
                marker=next(markers_cycle), markevery=markevery)
    ax.set_ylabel("Log size (1k)")
    ax.set_xlabel("Time (day)")
    ax.set_xticks(range(len(days))[::markevery])
    ax.set_xticklabels(days[::markevery])
    ax.grid(True)
    ax.legend()
    #plt.show()
    filepath = os.path.join(result_directory, filename_log_size_distribution + "." + plot_format)
    #fig.savefig(filepath, format=plot_format, bbox_inches="tight")
    plt.close(fig)
    
def plot_per_device_group_over_time(
        data_paths, filename_log_analysis, success_ratio, result_criteria, result_directory):
    
    def find_cold_start_day(device_class_result, days, result_metrics, success_ratio):        
        assert len(days) == len(device_class_result)
        criteria_results = list() # each metric in own array
        for i, _ in enumerate(result_metrics):
            criteria_results.append(numpy.asarray([time_result[i] for time_result in device_class_result]))
        
        # remove over fitting results
        mean_criteria = numpy.asarray(list(map(numpy.mean, criteria_results)))
        if len(numpy.where(mean_criteria == 1.0)[0]) == len(result_metrics):
            return -1
        criteria_results = [adapt_overfitting(data) for data in criteria_results]
        
        data_idx = numpy.arange(len(device_class_result))
        for start_idx in data_idx:
            days_idx = data_idx[start_idx:]
            over_thresholds = [numpy.where(criteria_result[days_idx] >= success_ratio)[0] for criteria_result in criteria_results]
            check_over_threshold = [len(over_threshold) == len(days_idx) for over_threshold in over_thresholds]
            if check_over_threshold.count(True) == len(result_metrics):
                return days[start_idx]
        return -1
    
    print("### plot per device class over time")
    data_path = data_generator.get_data_path(filename_log_analysis + "-per-class", data_paths)
    print("data path: ", data_path)
    results = DillSerializer(data_path).deserialize()
    features, clfs, time_identifiers = misc.get_all_keys(results)
    total_result = defaultdict(list)
    for clf in clfs:
        for feature in features:
            time_steps = list()
            device_class_results = DefaultOrderedDict(list)
            for time_identifier in time_identifiers:
                result = results[feature][clf][time_identifier]
                if len(result.index) != len(device_class_names):
                    continue
                time_steps.append(time_identifier)
                for device_class in device_class_names:   
                    device_class_results[device_class].append(result.loc[device_class][result_criteria].values)
            # per time and per criteria
            for device_class, device_class_result in device_class_results.items():
                cold_start = find_cold_start_day(device_class_result, time_steps, result_criteria, success_ratio)
                time_threshold = time_steps[int(len(time_steps)/4)]
                if cold_start != -1 and cold_start <= time_threshold:    
                    total_result[device_class].append((cold_start, device_class_result, result_criteria, clf, feature))
    
    f = open(os.path.join(result_directory, "per-device-group.txt"), "w")
    roc_data_path = data_generator.get_data_path(filename_log_analysis + "-roc", data_paths)
    print("ROC data path: ", data_path)
    roc_results = DillSerializer(roc_data_path).deserialize()
    for device_group in total_result:
        if "avg" not in device_group:
            class_data = total_result[device_group]
            cold_starts = list()
            roc_aucs = list()
            clf_feature = list()
            metrics = defaultdict(list)
            for cold_start, class_result, result_criteria, clf, feature in class_data:
                _, _, roc_auc, le = roc_results[feature][clf][cold_start]
                roc_aucs.append(roc_auc[le.transform([device_group])[0]])
                cold_starts.append(cold_start)
                clf_feature.append(clf + "-" + feature)
                for l, result_metric in enumerate(result_criteria):
                    results = numpy.asarray([result[l] for result in class_result])
                    metrics[result_metric].append(numpy.mean(results))
            
            f.write("device group: " + device_group + "\n")
            f.write("mean cold start: " + str(round(numpy.mean(cold_starts))) + "\n")
            f.write("mean roc auc: " + str(round(numpy.mean(roc_aucs),3)) + "\n")
            for metric, values in metrics.items():
                f.write("mean " + metric + ": " + str(round(numpy.mean(values),3)) + "\n")
            f.write("best clf feature: " + most_common(clf_feature) + "\n")
            f.write("-------------\n")
    
def across_testbeds_perform_analysis(days_step_size, dst_directory, filename_across_testbeds):
    
    def get_log_sample(log, days):
        log_days = numpy.asarray([datetime.day for datetime in log.keys()] + [0])
        day_changes = numpy.where(numpy.diff(log_days) != 0)[0]
        selected_idx = day_changes[:days][-1]
        data = OrderedDict()
        for i, timestamp in enumerate(log):
            if i > selected_idx:
                break
            data[timestamp] = log[timestamp]
        return data
    
    setting_testbeds = DillSerializer(data_generator.setting_log_testbeds).deserialize()
    log_data_folder = os.path.dirname(data_generator.setting_log_testbeds)
    testbeds = set([os.path.basename(entry).split("-")[0] for entry in glob.glob(os.path.join(log_data_folder, "*"))])
    testbeds.remove(os.path.basename(data_generator.setting_log_testbeds).split("-")[0]) # remove settings from the folder
    testbeds = list(testbeds)
    per_class_results = nested_dict(4, dict)
    for testbed_train, testbed_test in itertools.product(testbeds, repeat=2):
        print("testbed train: ", testbed_train, ", testbed test: ", testbed_test)
        settings_testbed_train = setting_testbeds[testbed_train]
        settings_testbed_test = setting_testbeds[testbed_test]
        
        data_paths_testbed_train = glob.glob(os.path.join(log_data_folder, testbed_train + "*"))
        device_pool_testbed_train = DillSerializer(data_generator.get_data_path("device-pool", data_paths_testbed_train)).deserialize()
        log_testbed_train = DillSerializer(data_generator.get_data_path("coupling-log", data_paths_testbed_train)).deserialize()
        total_num_weeks = list(log_testbed_train.keys())[0]
        log_testbed_train = log_testbed_train[total_num_weeks]
        
        data_paths_testbed_test = glob.glob(os.path.join(log_data_folder, testbed_test + "*"))
        device_pool_testbed_test = DillSerializer(data_generator.get_data_path("device-pool", data_paths_testbed_test)).deserialize()
        log_testbed_test = DillSerializer(data_generator.get_data_path("coupling-log", data_paths_testbed_test)).deserialize()[total_num_weeks]
        
        total_days = total_num_weeks * 7
        time_points = range(days_step_size, total_days+days_step_size, days_step_size)
        
        for days in time_points:
            print("days: ", days)
            log_train = get_log_sample(log_testbed_train, days)
            log_test = get_log_sample(log_testbed_test, days)
            for basic_features in [True, False]:
                features_testbed_train, _ = create_features(log_train, settings_testbed_train.sampling_resolution, basic_features, device_pool_testbed_train)
                features_testbed_test, device_class_encoder_test = create_features(log_test, settings_testbed_test.sampling_resolution, basic_features, device_pool_testbed_test)
                for (X_train, y_train, feature_desc_train), (X_test, y_test, feature_desc_test)  in zip(features_testbed_train, features_testbed_test):
                    assert feature_desc_train == feature_desc_test
                    for clf_type in Classifier: #  k-fold cross validation
                        print("testbed train: ", testbed_train)
                        print("testbed test: ", testbed_test)
                        print("days: ", days)
                        print("basic features: ", basic_features)
                        print("feature description: ", feature_desc_train)
                        print("clf type: ", clf_type.name)
                        print("----------------")
                        clf = Classifier.get_clf(clf_type)
                        clf = clf.fit(X_train, y_train)
                        y_pred = clf.predict(X_test)
                        y_score = clf.predict_proba(X_test)
                        result = calculate_metrics(y_test, y_pred, y_score, device_class_encoder_test)
                        key_error = False
                        try:
                            per_class_results[testbed_train][testbed_test][days][feature_desc_train][clf_type.name]
                        except KeyError:
                            key_error = True
                            pass
                        assert key_error
                        per_class_results[testbed_train][testbed_test][days][feature_desc_train][clf_type.name] = result
            print("########")
    print("save ...")
    DillSerializer(os.path.join(dst_directory, filename_across_testbeds)).serialize(per_class_results)
    
def across_testbeds_analysis(
        dst_directory, result_directory, filename_across_testbeds, result_criteria, plot_format):
    
    def adapt_ticklabels(labels, dic = {"full": "dense", "mid": "medium"}):
        labels = [dic.get(n, n) for n in labels]
        return list(map(lambda x:x.capitalize(), labels))
    
    print("### across testbeds analysis")
    results = DillSerializer(os.path.join(dst_directory, filename_across_testbeds)).deserialize()
    testbeds_train, testbeds_test, days, features, clfs = misc.get_all_keys(results)
    plot_metric = nested_dict(3, list)
    param_result = nested_dict(2, list)
    for testbed_train in testbeds_train:
        for testbed_test in testbeds_test:
            for day in days:
                best_param = None
                best_result = None
                select_criteria = -1
                for feature in features:
                    for clf in clfs:
                        result = results[testbed_train][testbed_test][day][feature][clf]
                        mean = numpy.mean([result[criteria]["avg / total"] for criteria in result_criteria])
                        if mean > select_criteria:
                            select_criteria = mean
                            best_param = clf + "-" + feature
                            best_result = result
                for metric, value in best_result[result_criteria].loc["avg / total"].items():
                    plot_metric[metric][testbed_train][testbed_test].append(value)
                param_result[testbed_train][testbed_test].append(best_param)
    
    with open(os.path.join(result_directory, "result.txt"), "w") as f:
        for testbed_train in param_result:
            for testbed_test in param_result[testbed_train]:
                params = param_result[testbed_train][testbed_test]
                f.write("train: " + testbed_train + ", test: " + testbed_test + "\n")
                f.write(most_common(params) + "\n")
            f.write("-----------------\n")
    
    for metric in plot_metric:
        fig, ax = plt.subplots()
        metric_data = list()
        for testbed_train in plot_metric[metric]:
            testbeds_train = plot_metric[metric].keys()
            metric_line = list()
            for testbed_test in plot_metric[metric][testbed_train]:
                testbeds_test = plot_metric[metric][testbed_train].keys()
                result_metrics = plot_metric[metric][testbed_train][testbed_test]
                metric_line.append(numpy.mean(result_metrics))
            metric_data.append(metric_line)
        im = ax.imshow(metric_data, cmap="jet", vmin=0, vmax=1)
        ax.set_xticks(numpy.arange(len(testbeds_train)))
        ax.set_yticks(numpy.arange(len(testbeds_test)))        
        ax.set_xticklabels(adapt_ticklabels(testbeds_train))
        ax.set_yticklabels(adapt_ticklabels(testbeds_test))
        for i in range(len(testbeds_train)):
            for j in range(len(testbeds_test)):
                ax.text(j, i, round(metric_data[i][j],2), ha="center", va="center")
        ax.set_ylabel("Train")
        ax.set_xlabel("Test")
        ax.figure.colorbar(im)
        filename = metric + "." + plot_format
        #fig.savefig(os.path.join(result_directory, filename), format=plot_format, bbox_inches="tight")   
        #plt.show()
        plt.close(fig)
    
def run(dst_directory, filename_log_analysis, filename_log_size_distribution, filename_across_testbeds):
    assert len(sys.argv) == 2
    days_step_size = 14
    if "across" in sys.argv[1]:
        print("run across testbeds analysis")
        across_testbeds_perform_analysis(days_step_size, dst_directory, filename_across_testbeds)
    else:
        print("run within testbed analysis")
        print("testbed: ", sys.argv[1])
        within_testbeds_perform_analysis(
            sys.argv[1], days_step_size, dst_directory, filename_log_analysis, filename_log_size_distribution)
    
def result_analysis(dst_directory, filename_log_analysis, filename_log_size_distribution, filename_across_testbeds):
    success_ratio = 0.8
    plot_format = "pdf"
    num_device_classes = 3
    num_device_classes_mixtures = 7
    all_data_paths = [glob.glob(os.path.join(dst_directory, identifier + "-*")) for identifier in ["full", "mid", "sparse"]]
    
    result_directory = os.path.join(__location__, "results")
#     across_testbeds_result_directory = os.path.join(result_directory, "across-testbeds")
#     if not os.path.exists(across_testbeds_result_directory):
#         os.makedirs(across_testbeds_result_directory)
#     across_testbeds_analysis(
#         dst_directory, across_testbeds_result_directory, filename_across_testbeds, result_criteria, plot_format)
    
    within_testbeds_result_directory = os.path.join(result_directory, "within-testbeds")
    #data_generator.plot_calendar_and_devices(within_testbeds_result_directory)
    #plot_log_size(all_data_paths, filename_log_size_distribution, 70, within_testbeds_result_directory, plot_format)
    
    for data_paths in all_data_paths:
        testbed = os.path.basename(data_paths[0]).split("-")[0]
        print("testbed: ", testbed)
        testbed_result = os.path.join(within_testbeds_result_directory, testbed)
        if not os.path.exists(testbed_result):
            os.makedirs(testbed_result)
        #plot_per_device_group_over_time(
        #    data_paths, filename_log_analysis, success_ratio, result_criteria, testbed_result)
        for true_clusters in [num_device_classes, num_device_classes_mixtures]:
            plot_per_clustering_vs_time(
                data_paths, filename_log_analysis, true_clusters, testbed_result, plot_format)

'''
run the program at the server
emu03: ssh haus@emu03.cm.in.tum.de

nohup python -m coupling.log.analysis full>full.log &
nohup python -m coupling.log.analysis mid>mid.log &
nohup python -m coupling.log.analysis sparse>sparse.log &
nohup python -m coupling.log.analysis across>across.log &
'''

def main():
    # Further parallelization: separate clustering and machine learning into own processes
    dst_directory = os.path.join(__location__, "raw-result")
    if not os.path.exists(dst_directory):
        os.makedirs(dst_directory)
    filename_log_analysis = "log-analysis-within-testbeds"
    filename_log_size_distribution = "log-size-distribution"
    filename_across_testbeds = "log-analysis-across-testbeds"
    
    if not os.path.exists(dst_directory):
        run(dst_directory, filename_log_analysis, filename_log_size_distribution, filename_across_testbeds)
    result_analysis(dst_directory, filename_log_analysis, filename_log_size_distribution, filename_across_testbeds)
    
if __name__ == "__main__":
    main()
    