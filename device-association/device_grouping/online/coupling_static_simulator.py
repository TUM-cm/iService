import os
import sys
import logging
import datetime
import subprocess
from utils.nested_dict import nested_dict
from utils.serializer import DillSerializer
from sklearn.model_selection import ParameterGrid
import coupling.utils.vector_similarity as vector_similarity
from coupling.device_grouping.online.machine_learning_features import Classifier
from coupling.device_grouping.offline.sampling_time import get_pattern_max_sampling_period

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

coupling_compare_methods = {
    "pearson": vector_similarity.pearson,
    "spearman": vector_similarity.spearman,
    "kendall": vector_similarity.kendall
}

coupling_ml_classifiers = {
    #"SVM": Classifier.SVM,
    "Random Forest": Classifier.RandomForest,
    "Extra Trees": Classifier.ExtraTreesClassifier,
    "Gradient Boosting": Classifier.GradientBoostingClassifier
}

equalize_methods = {
    #"fill": vector_similarity.equalize_methods.fill,
    #"cut": vector_similarity.equalize_methods.cut,
    "dtw": vector_similarity.equalize_methods.dtw
}

localization_pos_in_area = [1, 2, 3, 4, 5]
path_active_parameters = os.path.join(__location__, "active_params")
path_evaluation_data = os.path.join(
    __location__, "..", "data", "simulation", "static-coupling-simulation")
evaluation_rounds = 1

def add_evaluation_data(evaluation_result):
    logging.info("add evaluation data")
    params = DillSerializer(path_active_parameters).deserialize()
    evaluation_data = DillSerializer(path_evaluation_data).deserialize()
    logging.info(params)
    evaluation_data[params["num clients"]][params["num reject clients"]] \
        [params["len light pattern"]][params["sampling period coupling"]] \
        [params["coupling compare method"]][params["coupling similarity threshold"]] \
        [params["equalize method"]][params["sampling period localization"]] \
        [params["sampling period ml train"]][params["coupling ml classifier"]].append(evaluation_result)
    DillSerializer(path_evaluation_data).serialize(evaluation_data)

class Parameter:
    
    def __init__(self):
        testbed = "vm" # server, vm
        num_clients = 10
        self.server_ip = "localhost"
        self.server_port = 10026
        self.num_clients = range(2, num_clients+1)
        self.num_reject_clients = range(num_clients-1)
        self.len_light_patterns = [2, 4, 6, 8, 10]
        self.sampling_period_couplings = [get_pattern_max_sampling_period()]
        self.coupling_compare_methods = coupling_compare_methods.keys()
        self.coupling_similarity_thresholds = [0.7]
        self.sampling_period_localizations = [5]
        self.localization_pos_in_area = localization_pos_in_area
        fingerprint_directory = os.path.join(__location__, "..", "..", "localization", "data")
        self.path_wifi_scans = os.path.join(fingerprint_directory, "wifi-fingerprints")
        self.path_ble_scans = os.path.join(fingerprint_directory, "bluetooth-fingerprints")
        self.path_ml_train_data = os.path.join(__location__, "..", "..", "simulator", "ml-train-data", testbed)
        self.sampling_period_ml_trains = [0.05]
        self.coupling_ml_classifiers = coupling_ml_classifiers.keys()
        self.equalize_methods = equalize_methods.keys()
    
def filter_params(param_grid):
    filtered_params = list()
    for param in param_grid:
        if param["num clients"] - param["num reject clients"] >= 2:
            filtered_params.append(param)
    return filtered_params

class Evaluation:
    
    def __init__(self, script, parameter, num_parameter):
        self.script = script
        self.parameter = parameter
        DillSerializer(path_evaluation_data).serialize(nested_dict(num_parameter, list))
    
    def start(self):
        logging.basicConfig(filename="static-coupling-simulation.log", level=logging.DEBUG)
        param_grid = ParameterGrid({"num clients": self.parameter.num_clients,
                                    "num reject clients": self.parameter.num_reject_clients,
                                    "len light pattern": self.parameter.len_light_patterns,
                                    "coupling ml classifier": self.parameter.coupling_ml_classifiers,
                                    "coupling compare method": self.parameter.coupling_compare_methods,
                                    "equalize method": self.parameter.equalize_methods,
                                    "coupling similarity threshold": self.parameter.coupling_similarity_thresholds,
                                    "sampling period coupling": self.parameter.sampling_period_couplings,
                                    "sampling period localization": self.parameter.sampling_period_localizations,
                                    "sampling period ml train": self.parameter.sampling_period_ml_trains})
        filtered_params = filter_params(param_grid)
        param_len = len(filtered_params)
        for i, params in enumerate(filtered_params):
            logging.info("######### start ##########")
            logging.info("Param: " + str(i+1) + "/" + str(param_len))
            logging.info("Time: " + str(datetime.datetime.now().time()))
            logging.info(params)
            DillSerializer(path_active_parameters).serialize(params)
            subprocess.check_output([sys.executable, self.script,
                                    str(self.parameter.server_ip),
                                    str(self.parameter.server_port),
                                    str(params["num clients"]),
                                    str(params["num reject clients"]),
                                    str(params["len light pattern"]),
                                    str(params["sampling period coupling"]),
                                    params["coupling compare method"],
                                    str(params["coupling similarity threshold"]),
                                    params["equalize method"],
                                    str(params["sampling period ml train"]),
                                    self.parameter.path_ml_train_data,
                                    params["coupling ml classifier"],
                                    str(params["sampling period localization"]),
                                    str(self.parameter.localization_pos_in_area),
                                    self.parameter.path_wifi_scans,
                                    self.parameter.path_ble_scans])
            logging.info("######### end ##########")
    
if __name__ == "__main__":
    num_parameter = 10
    parameter = Parameter()
    script_path = os.path.join(
        __location__, "..", "..", "simulator", "static", "coupling_simulation_round.py")
    evaluation = Evaluation(script_path, parameter, num_parameter)
    evaluation.start()
    