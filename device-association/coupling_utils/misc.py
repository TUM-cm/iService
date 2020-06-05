import os
import logging
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

class CouplingResult:
    
    def __init__(self, result_accept, result_reject, groundtruth_accept,
                 groundtruth_reject, runtime, mac_mapping):
        self.result_accept = result_accept
        self.result_reject = result_reject
        self.groundtruth_accept = groundtruth_accept
        self.groundtruth_reject = groundtruth_reject
        self.runtime = runtime
        logging.debug("### accept results")
        self.accuracy_accept, self.precision_accept, \
            self.recall_accept, self.f1_accept = metrics(
                groundtruth_accept, result_accept, mac_mapping)
        logging.debug("### reject results")
        self.accuracy_reject, self.precision_reject, \
            self.recall_reject, self.f1_reject = metrics(
                groundtruth_reject, result_reject, mac_mapping)
    
def metrics(y_true, y_pred, mac_mapping, average="micro"):
    # Change MAC addresses to binary arrays
    binarizer = MultiLabelBinarizer()
    binarizer.fit([mac_mapping.values()])
    y_true = binarizer.transform([y_true])
    y_pred = binarizer.transform([y_pred])
    accuracy, precision, recall, f1 = -1, -1, -1, -1
    try:
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average=average)
        recall = recall_score(y_true, y_pred, average=average)
        f1 = f1_score(y_true, y_pred, average=average)
    except:
        pass
    logging.debug("accuracy: {0:.2f}".format(accuracy))
    logging.debug("precision: {0:.2f}".format(precision))
    logging.debug("recall: {0:.2f}".format(recall))
    logging.debug("f1: {0:.2f}".format(f1))
    return accuracy, precision, recall, f1

class EvaluationResult:
    
    def __init__(self, runtime_query_data, runtime_query_raw_light, runtime_query_pattern_light, runtime_query_raw_wifi, runtime_query_raw_ble, runtime_coupling,
                 localization_random_wifi, localization_filtering_wifi, localization_svm_wifi, localization_random_forest_wifi,
                 localization_random_ble, localization_filtering_ble, localization_svm_ble, localization_random_forest_ble,
                 coupling_signal_pattern_duration, coupling_signal_pattern, coupling_signal_similarity,
                 coupling_machine_learning_basic_all, coupling_machine_learning_basic_selected,
                 coupling_machine_learning_tsfresh_all, coupling_machine_learning_tsfresh_selected,
                 coupling_tvgl):
        
        self.runtime_query_data = runtime_query_data
        self.runtime_query_raw_light = runtime_query_raw_light
        self.runtime_query_pattern_light = runtime_query_pattern_light
        self.runtime_query_raw_wifi = runtime_query_raw_wifi
        self.runtime_query_raw_ble = runtime_query_raw_ble
        self.runtime_coupling = runtime_coupling
        self.localization_random_wifi = localization_random_wifi
        self.localization_filtering_wifi = localization_filtering_wifi
        self.localization_svm_wifi = localization_svm_wifi
        self.localization_random_forest_wifi = localization_random_forest_wifi
        self.localization_random_ble = localization_random_ble
        self.localization_filtering_ble = localization_filtering_ble
        self.localization_svm_ble = localization_svm_ble
        self.localization_random_forest_ble = localization_random_forest_ble
        self.coupling_signal_pattern_duration = coupling_signal_pattern_duration
        self.coupling_signal_pattern = coupling_signal_pattern
        self.coupling_signal_similarity = coupling_signal_similarity
        self.coupling_machine_learning_basic_all = coupling_machine_learning_basic_all
        self.coupling_machine_learning_basic_selected = coupling_machine_learning_basic_selected
        self.coupling_machine_learning_tsfresh_all = coupling_machine_learning_tsfresh_all
        self.coupling_machine_learning_tsfresh_selected = coupling_machine_learning_tsfresh_selected
        self.coupling_tvgl = coupling_tvgl
    