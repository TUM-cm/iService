import os
import time
import numpy
import pandas
import random
from enum import Enum
from sklearn import svm
import tsfresh.utilities
import tsfresh.feature_selection
import tsfresh.feature_extraction
import coupling.utils.misc as misc
from sklearn.model_selection import KFold
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
import coupling.light_grouping_pattern.light_analysis as light_analysis
from coupling.device_grouping.online.static.coupling_data_provider import CouplingDataProvider

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

class LightData:
    
    def __init__(self, sampling_period, len_light_patterns=range(2, 11, 2),
                 distortion_threshold=0.6, rounds=100, time_delta=0.0001):
        light_pattern_sequences = list()
        reset_sampling_period = sampling_period
        print("len light patterns: ", len_light_patterns)
        for len_light_pattern in len_light_patterns:
            light_pattern, light_pattern_time = light_analysis.load_light_pattern(len_light_pattern)
            coupling_data_provider = CouplingDataProvider(light_pattern, light_pattern_time, None, None)
            counter = 0
            found = False
            print("len light pattern: ", len_light_pattern)
            print("sampling period: ", sampling_period)
            while not found:
                try:
                    if counter == rounds:
                        counter = 0
                        sampling_period += time_delta
                        print("sampling period: ", sampling_period)
                    light_pattern, light_pattern_time = coupling_data_provider.get_light_data(sampling_period)
                    light_pattern_duration, light_pattern_sequence = light_analysis.detect_cycle_by_sequence(
                                                                        light_pattern, light_pattern_time)
                    if misc.valid_light_pattern(light_pattern_duration, len_light_pattern):
                        light_pattern_sequences.extend(light_pattern_sequence)
                        found = True
                    counter += 1
                except:
                    pass
            sampling_period = reset_sampling_period
        self.X_basic, self.y_basic = self.basic(light_pattern_sequences, distortion_threshold)
        self.X_tsfresh, self.y_tsfresh = self.tsfresh(light_pattern_sequences, distortion_threshold)
        assert len(self.X_basic) == len(self.y_basic) == len(self.X_tsfresh.id.unique()) == len(self.y_tsfresh)
    
    def basic(self, voltage_sequences, distortion_threshold):
        X_raw, y_raw = self.get_raw_data(voltage_sequences)
        X_distorted, y_distorted = self.get_distorted_data(voltage_sequences, distortion_threshold)
        X_random, y_random = self.get_random_data(voltage_sequences)
        X = X_raw + X_distorted + X_random
        y = numpy.concatenate([y_raw, y_distorted, y_random])
        assert len(X) == len(y)
        return X, y
    
    def get_raw_data(self, voltage_sequences):
        X_raw = list(voltage_sequences)
        y_raw = numpy.ones(len(voltage_sequences))
        return X_raw, y_raw
    
    def get_distorted_data(self, voltage_sequences, distortion_threshold):
        data_amount = len(voltage_sequences)
        mean = [numpy.mean(entry) for entry in voltage_sequences] # mean = 0
        std = [numpy.std(entry) for entry in voltage_sequences] # std = 1
        X = list()
        y = list()
        for distortion_rate in numpy.arange(0.1, 1.05, 0.1):
            for voltage_sequence in voltage_sequences:
                noise = numpy.random.normal(random.choice(mean), random.choice(std), len(voltage_sequence))
                noisy_data = voltage_sequence + noise * distortion_rate
                X.append(noisy_data)    
            if distortion_rate < distortion_threshold:
                result = numpy.ones(shape=(data_amount,))                
            else:
                result = numpy.zeros(shape=(data_amount,))
            y.extend(result)
        return X, numpy.asarray(y)
    
    def get_random_data(self, voltage_sequences):
        data_amount = len(voltage_sequences)
        datalen = [len(entry) for entry in voltage_sequences]
        start_range = min([min(entry) for entry in voltage_sequences])
        end_range = max([max(entry) for entry in voltage_sequences])
        X = list()
        for _ in range(len(voltage_sequences)):
            len_voltage_sequence = random.choice(datalen)
            sequence = numpy.random.randint(start_range, end_range, size=len_voltage_sequence)
            X.append(sequence)
        y = numpy.zeros(shape=(data_amount,))
        return X, y
    
    def tsfresh(self, voltage_sequences, distortion_threshold):
        X_raw, X_raw_id, y_raw = self.get_raw_data_tsfresh(voltage_sequences)
        start_signal_id = numpy.max(X_raw_id) + 1
        X_distortion, X_distortion_id, y_distortion = self.get_distorted_data_tsfresh(
            voltage_sequences, start_signal_id, distortion_threshold)
        start_signal_id = numpy.max(X_distortion_id) + 1
        X_random, X_random_id, y_random = self.get_random_data_tsfresh(voltage_sequences, start_signal_id)
        signal_id_range = numpy.max(X_random_id)
        X = pandas.DataFrame({"id": numpy.concatenate([X_raw_id, X_distortion_id, X_random_id]),
                              "X": numpy.concatenate([X_raw, X_distortion, X_random])})
        y = pandas.Series(numpy.concatenate([y_raw, y_distortion, y_random]),
                          index=range(1, signal_id_range + 1))
        assert len(X.id.unique()) == len(y)
        return X, y
    
    def get_raw_data_tsfresh(self, voltage_sequences):
        signal_len = [len(voltage_signal) for voltage_signal in voltage_sequences]
        X_raw_id = numpy.array(range(1, len(voltage_sequences) + 1))
        X_raw_id = numpy.repeat(X_raw_id, signal_len)
        X_raw = numpy.concatenate(voltage_sequences)
        y_raw = numpy.ones(len(voltage_sequences))
        return X_raw, X_raw_id, y_raw
    
    def get_distorted_data_tsfresh(self, voltage_sequences, start_signal_id, distortion_threshold):
        voltage_signal = numpy.concatenate(voltage_sequences)
        mean = voltage_signal.mean()
        std = voltage_signal.std()
        X_distortion = numpy.array([])
        X_distortion_id = numpy.array([], numpy.int64)
        y_distortion = numpy.array([])
        signal_id = start_signal_id
        for series in voltage_sequences:
            series_len = len(series)
            noise = numpy.random.normal(mean, std, series_len)
            for distortion_rate in numpy.arange(0.1, 1.05, 0.1):  
                noisy_data = series + (noise * distortion_rate)
                if distortion_rate < distortion_threshold:
                    y_distortion = numpy.concatenate([y_distortion, [1]])       
                else:
                    y_distortion = numpy.concatenate([y_distortion, [0]])
                X_distortion = numpy.concatenate([X_distortion, noisy_data])
                X_distortion_id = numpy.concatenate([X_distortion_id, [signal_id] * series_len])
                signal_id += 1
        return X_distortion, X_distortion_id, y_distortion
    
    def get_random_data_tsfresh(self, voltage_sequences, start_signal_id):
        signal_len = [len(sequence) for sequence in voltage_sequences]
        signal_len_min = min(signal_len)
        signal_len_max = max(signal_len)
        signal_value_min = min([min(sequence) for sequence in voltage_sequences])
        signal_value_max = max([max(sequence) for sequence in voltage_sequences])
        num_samples = len(signal_len)
        X_random = numpy.array([])
        X_random_id = numpy.array([], numpy.int64)
        y_random = numpy.array([])
        signal_id = start_signal_id
        for _ in range(num_samples):
            signal_len = random.randint(signal_len_min, signal_len_max)
            X = numpy.random.randint(signal_value_min, signal_value_max, signal_len)
            X_random = numpy.concatenate([X_random, X])
            y_random = numpy.concatenate([y_random, [0]])
            X_random_id = numpy.concatenate([X_random_id, [signal_id] * signal_len])
            signal_id += 1
        return X_random, X_random_id, y_random

class TsFreshFeatures:
    
    # nohup python -m coupling.device_grouping.simulator.machine_learning_features &
    def extract(self, X, y):
        features_extracted = tsfresh.extract_features(X, column_id="id", disable_progressbar=True)
        tsfresh.utilities.dataframe_functions.impute(features_extracted)
        features_filtered = tsfresh.select_features(features_extracted, y)
        return features_filtered
    
    def extract_selected_features(self, X, features_to_extract, create_vector=False):
        if create_vector:
            X = pandas.DataFrame({"id": [1] * len(X), "X": X})
        X = tsfresh.extract_features(
            X, column_id="id", kind_to_fc_parameters=features_to_extract,
            disable_progressbar=True, n_jobs=0)
        tsfresh.utilities.dataframe_functions.impute(X)
        return X
    
    # Num features
    # 794: default
    # 794: comprehensive
    # 788: efficient
    # 8: minimal
    def relevance(self, X, y):
        from tsfresh.feature_extraction import EfficientFCParameters
        features_extracted = tsfresh.extract_features(
            X, column_id="id", default_fc_parameters=EfficientFCParameters(), disable_progressbar=True)
        tsfresh.utilities.dataframe_functions.impute(features_extracted)
        relevance_features = tsfresh.feature_selection.relevance.calculate_relevance_table(features_extracted, y)
        return features_extracted, relevance_features
    
    def select_n_most_useful_features(self, relevance_features, num_features=10):
        relevance_features = relevance_features[relevance_features.relevant == True]
        relevance_features = relevance_features[~relevance_features.p_value.isna()]
        relevance_features = relevance_features.sort_values(by=["p_value"]) # ascending=False
        return relevance_features[:num_features][["feature", "p_value"]]
    
    def performance_evaluation(self, features_extracted, relevance_features, X_tsfresh, rounds=10, step_size=10):
        print("features extracted: ", features_extracted.shape)
        print("relevance features: ", relevance_features.shape)
        relevance_features = relevance_features[relevance_features.relevant == True]
        relevance_features = relevance_features[~relevance_features.p_value.isna()]
        relevance_features = relevance_features.sort_values(by=["p_value"])
        print("relevance features: ", relevance_features.shape)
        num_test_features = range(step_size, len(relevance_features), step_size)
        print("test sizes: ", num_test_features)
        elapsed_times = pandas.DataFrame(columns=num_test_features)
        test_data = X_tsfresh[X_tsfresh.id == 1]
        for num_features in num_test_features:
            print("num features: ", num_features)
            column_names = relevance_features[:num_features].feature    
            features_to_extract = tsfresh.feature_extraction.settings.from_columns(features_extracted[column_names])
            print("shape test data: ", test_data.shape)
            print("unique test data: ", test_data.id.unique())
            print("amount of feature names: ", len(column_names))
            elapsed_time = list()
            for _ in range(rounds):
                start_time = time.time()
                test_data_extracted = tsfresh.extract_features(
                    test_data, column_id="id", kind_to_fc_parameters=features_to_extract, disable_progressbar=True)
                end = time.time() - start_time
                elapsed_time.append(end)
                print("shape test data extracted: ", test_data_extracted.shape)
            print("elapsed time: ", elapsed_time)
            print("---")
            elapsed_times[num_features] = elapsed_time
        print(elapsed_times.to_string())
        return elapsed_times
    
class BasicFeatures:
    
    def __init__(self):
        self.labels_selected = ["var", "std"]
        self.labels = ["length", "max", "mean", "median", "min", "std", "sum", "var"]
        self.num_features_all = len(self.labels)
        self.num_features_selected = len(self.labels_selected)
    
    def extract(self, X_raw):
        X = numpy.empty(shape=(len(X_raw), self.num_features_all))
        for i, sample in enumerate(X_raw):
            X[i] = self.compute(sample)
        return X
    
    def extract_selected_features(self, X_raw):
        X = numpy.empty(shape=(len(X_raw), self.num_features_selected))
        for i, sample in enumerate(X_raw):
            X[i] = self.compute_selected_features(sample)
        return X
    
    def compute(self, sample):
        feature = numpy.empty(shape=(self.num_features_all))
        feature[0] = len(sample) # length
        feature[1] = numpy.max(sample) # max
        feature[2] = numpy.mean(sample) # mean
        feature[3] = numpy.median(sample) # median
        feature[4] = numpy.min(sample) # min
        feature[5] = numpy.std(sample) # standard deviation
        feature[6] = numpy.sum(sample) # sum values
        feature[7] = numpy.var(sample) # variance
        return feature
    
    def compute_selected_features(self, sample):
        feature = numpy.empty(shape=(self.num_features_selected))
        feature[0] = numpy.var(sample)
        feature[1] = numpy.std(sample)
        return feature
    
    # http://scikit-learn.org/stable/modules/feature_selection.html
    def relevance(self, classifier, X, y, n_fold=10, i=0):
        k_fold = KFold(n_splits=n_fold, shuffle=True)
        scores = pandas.DataFrame(columns=self.labels)
        for train_indices, _ in k_fold.split(X):
            clf = Classifier.get_clf(classifier)
            clf = clf.fit(X[train_indices], y[train_indices])
            scores.loc[i] = clf.feature_importances_
            i += 1
        scores = scores.sum()
        return pandas.DataFrame({"feature": scores.index, "relative_importance": scores.values})
    
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
            return ExtraTreesClassifier(n_estimators=100)
        elif clf_type == Classifier.GradientBoostingClassifier:
            return GradientBoostingClassifier()
        elif clf_type == Classifier.SVM:
            return svm.SVC(probability=True, gamma="scale")
        elif clf_type == Classifier.RandomForest:
            return RandomForestClassifier(n_estimators=100)
        elif clf_type == Classifier.NaiveBayes:
            return GaussianNB()
        elif clf_type == Classifier.AdaBoost:
            return AdaBoostClassifier(n_estimators=100)
    