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

localization_room_to_pos = {1: [1,2,3,4,5], 2: [6,7,8,9], 3: [10,11], 4: [12,13],
                            5: [14,15,16,17,18,19], 6: [20,21,22,23], 7: [24,25,26,27],
                            8: [28,29,30,31], 9: [32,33,34,35], 10: [36,37,38,39]}

path_active_parameters = os.path.join(__location__, "active_params")
path_evaluation_data = os.path.join(
    __location__, "..", "data", "simulation", "dynamic-coupling-simulation")

def add_evaluation_data(evalation_result):
    logging.info("add evaluation data")
    params = DillSerializer(path_active_parameters).deserialize()
    evaluation_data = DillSerializer(path_evaluation_data).deserialize()
    logging.info(params)
    evaluation_data[params["sampling period coupling"]] \
        [params["coupling compare method"]][params["coupling similarity threshold"]] \
        [params["equalize method"]][params["sampling period localization"]] \
        [params["sampling period ml train"]][params["coupling ml classifier"]] \
        [params["num users"]][params["num rooms"]] \
        [params["simulation duration"]][params["coupling frequency"]].append(evalation_result)
    DillSerializer(path_evaluation_data).serialize(evaluation_data)
    
class Parameter:
    
    def __init__(self):
        testbed = "vm" # server, vm
        self.server_ip = "localhost"
        self.server_port = 10026
        self.sampling_period_couplings = [get_pattern_max_sampling_period()]
        self.coupling_compare_methods = coupling_compare_methods.keys()
        self.coupling_similarity_thresholds = [0.7]
        self.sampling_period_localizations = [5]
        self.path_ml_train_data = os.path.join(
            __location__, "..", "..", "simulator", "ml-train-data", testbed)
        self.path_localization_data = os.path.join(
            __location__, "..", "..", "localization", "data")
        self.localization_room_to_pos = localization_room_to_pos
        self.sampling_period_ml_trains = [0.05]
        self.coupling_ml_classifiers = coupling_ml_classifiers.keys()
        self.equalize_methods = equalize_methods.keys()
        self.coupling_frequency = [10, 20, 30]
        self.num_users = [3, 5, 10]
        self.num_rooms = range(1, 11, 1)
        self.simulation_duration = [180]
    
class Evaluation:
    
    def __init__(self, script, parameter, num_parameter):
        self.script = script
        self.parameter = parameter
        DillSerializer(path_evaluation_data).serialize(nested_dict(num_parameter, list))
    
    def start(self):
        logging.basicConfig(filename="dynamic-coupling-simulation.log", level=logging.DEBUG,
                            format="%(asctime)s %(message)s", datefmt="%I:%M:%S")
        param_grid = ParameterGrid({"num users": self.parameter.num_users,
                                    "num rooms": self.parameter.num_rooms,
                                    "simulation duration": self.parameter.simulation_duration,
                                    "coupling ml classifier": self.parameter.coupling_ml_classifiers,
                                    "coupling compare method": self.parameter.coupling_compare_methods,
                                    "equalize method": self.parameter.equalize_methods,
                                    "coupling similarity threshold": self.parameter.coupling_similarity_thresholds,
                                    "sampling period coupling": self.parameter.sampling_period_couplings,
                                    "sampling period localization": self.parameter.sampling_period_localizations,
                                    "sampling period ml train": self.parameter.sampling_period_ml_trains,
                                    "coupling frequency": self.parameter.coupling_frequency})
        param_len = len(param_grid)
        for i, params in enumerate(param_grid):
            logging.info("######### start ##########")
            logging.info("Param: " + str(i+1) + "/" + str(param_len))
            logging.info("Time: " + str(datetime.datetime.now().time()))
            logging.info(params)
            DillSerializer(path_active_parameters).serialize(params)
            subprocess.check_output([sys.executable, self.script,
                                    str(self.parameter.server_ip),
                                    str(self.parameter.server_port),
                                    str(params["sampling period coupling"]),
                                    params["coupling compare method"],
                                    str(params["coupling similarity threshold"]),
                                    params["equalize method"],
                                    str(params["sampling period ml train"]),
                                    self.parameter.path_ml_train_data,
                                    params["coupling ml classifier"],
                                    self.parameter.path_localization_data,
                                    str(self.parameter.localization_room_to_pos),
                                    str(params["sampling period localization"]),
                                    str(params["coupling frequency"]),
                                    str(params["num users"]),
                                    str(params["num rooms"]),
                                    str(params["simulation duration"])])
            logging.info("######### end ##########")
    
if __name__ == "__main__":
    num_parameter = 11
    parameter = Parameter()
    script_path = os.path.join(
        __location__, "..", "..", "simulator", "dynamic", "coupling_simulation_round.py")
    evaluation = Evaluation(script_path, parameter, num_parameter)
    evaluation.start()
    