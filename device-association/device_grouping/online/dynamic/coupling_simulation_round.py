import os
import sys

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
module_path = os.path.join(__location__, "..", "..", "..")
sys.path.append(module_path)

import ast
import logging
import collections
from mock import MagicMock
import coupling.utils.misc as misc
from collections import defaultdict
from twisted.internet import reactor
from coupling.utils.misc import create_random_mac
from coupling.utils.access_point import AccessPoint
from coupling.utils.coupling_data import DynamicCouplingResult
from coupling.utils.coupling_data import DynamicEvaluationResult
from coupling.device_grouping.online.dynamic.coupling_user import CouplingUser
from coupling.device_grouping.online.dynamic.coupling_server import ServerController
from coupling.device_grouping.online.dynamic.coupling_client import ClientController
from coupling.device_grouping.online.dynamic.coupling_testbed import CouplingTestbed
import coupling.device_grouping.online.coupling_dynamic_simulator as coupling_simulator
from coupling.device_grouping.offline.sampling_time import get_pattern_max_sampling_period

# global data
clients = dict()
mac_mapping = dict()

class SimulationData:
    
    def __init__(self, server_ip, server_port,
                 data_period_coupling, coupling_compare_method, coupling_similarity_threshold, equalize_method,
                 data_period_ml_train, path_ml_train_data, coupling_ml_classifier,
                 path_localization_data, localization_room_to_pos, data_period_localization,
                 frequency_coupling, num_users, num_rooms, simulation_duration):
        
        self.server_ip = server_ip
        self.server_port = int(server_port)
        self.data_period_coupling = float(data_period_coupling)
        self.str_coupling_compare_method = coupling_compare_method
        self.coupling_compare_method = coupling_simulator.coupling_compare_methods[self.str_coupling_compare_method]
        self.coupling_similarity_threshold = float(coupling_similarity_threshold)
        self.str_equalize_method = equalize_method
        self.equalize_method = coupling_simulator.equalize_methods[self.str_equalize_method]
        self.data_period_localization = float(data_period_localization)
        self.frequency_coupling = float(frequency_coupling)
        localization_room_to_pos = ast.literal_eval(localization_room_to_pos)        
        self.testbed = CouplingTestbed(int(num_users), int(num_rooms), int(simulation_duration),
                                       float(data_period_ml_train), path_ml_train_data, coupling_ml_classifier,
                                       path_localization_data, localization_room_to_pos)

def stop_reactor_callback():
    logging.info("stop reactor")
    reactor.stop()
    
def evaluate_callback(evaluation_coupling, evaluation_runtime):
    coupling_results = defaultdict(list)
    for mac in evaluation_coupling:
        route_per_coupling_method = dict()
        runtime_per_coupling_method = dict()
        for coupling_method in evaluation_coupling[mac]:
            results = evaluation_coupling[mac][coupling_method]
            results = collections.OrderedDict(sorted(results.items(), key=lambda t:t[0]))
            route = list()
            runtimes = list()
            for time in results:
                route.extend([entry[0] for entry in results[time] if entry[1]])
                runtimes.extend([entry[2] for entry in results[time]])
            if len(route) == 0: # empty route thus -1 for metrics
                route.append(1)
            route_per_coupling_method[coupling_method] = route
            runtime_per_coupling_method[coupling_method] = runtimes
        groundtruth_route = [room[1] for room in clients[mac].factory.get_route()]
        for coupling_method in route_per_coupling_method:
            route = route_per_coupling_method[coupling_method]
            route = misc.del_repeated_items(route, groundtruth_route)
            coupling_results[coupling_method].append(
                DynamicCouplingResult(route, groundtruth_route, runtime_per_coupling_method[coupling_method]))
    coupling_simulator.add_evaluation_data(
        DynamicEvaluationResult(coupling_results, evaluation_runtime))
    
def get_mac(identifier):
    if identifier not in mac_mapping:
        mac_mapping[identifier] = create_random_mac()
        for initial_key, client in clients.iteritems():
            if client.factory.transport:
                remote = client.factory.transport.getHost()
                if identifier == remote.host or identifier == remote.port:
                    mac = mac_mapping[identifier]
                    client.factory.set_mac(mac)
                    clients[mac] = clients.pop(initial_key)
                    break
    return mac_mapping[identifier]

def run(parameter):
    access_point = AccessPoint()
    access_point.deny_hosts = MagicMock()
    access_point.get_mac = MagicMock(side_effect=get_mac)
    for identifier, user_route in parameter.testbed.get_user_routes().iteritems():
        logging.info("route: " + str(identifier) + " - " + str(user_route))
        user = CouplingUser(identifier, user_route,
                    parameter.testbed.get_room_distances(),
                    parameter.testbed.get_client_data())
        client = ClientController(parameter.server_ip,
                                  parameter.server_port,
                                  user)
        clients[identifier] = client
    
    server = ServerController(parameter.server_port, access_point,
                              parameter.data_period_coupling, parameter.coupling_compare_method,
                              parameter.coupling_similarity_threshold, parameter.equalize_method,
                              parameter.data_period_localization,
                              len(clients), parameter.testbed.get_room_data(), parameter.frequency_coupling,                          
                              stop_reactor_callback, evaluate_callback)
    server.start()
    for client in clients.values():
        client.start()
    reactor.run()
    
def test():
    testbed = "vm" # server, vm
    server_ip = "localhost"
    server_port = 1026
    data_period_coupling = get_pattern_max_sampling_period()
    coupling_compare_method = "pearson"
    coupling_similarity_threshold = 0.7
    equalize_method = "dtw"
    data_period_ml_train = 0.05
    coupling_ml_classifier = "Random Forest"
    path_ml_train_data = os.path.join(
        __location__, "..", "ml-train-data", testbed)
    path_localization_data = os.path.join(
        __location__, "..", "..", "localization", "data")
    localization_room_to_pos = str(coupling_simulator.localization_room_to_pos)
    data_period_localization = 5
    frequency_coupling = 30
    simulation_duration = 60
    num_users = 1
    num_rooms = 10
    parameter = SimulationData(server_ip, server_port,
                               data_period_coupling, coupling_compare_method,
                               coupling_similarity_threshold, equalize_method,
                               data_period_ml_train, path_ml_train_data, coupling_ml_classifier,
                               path_localization_data, localization_room_to_pos, data_period_localization,
                               frequency_coupling, num_users, num_rooms, simulation_duration)
    run(parameter)
    
def evaluation():
    parameter = SimulationData(sys.argv[1], sys.argv[2], sys.argv[3],
                               sys.argv[4], sys.argv[5], sys.argv[6],
                               sys.argv[7], sys.argv[8], sys.argv[9],
                               sys.argv[10], sys.argv[11], sys.argv[12],
                               sys.argv[13], sys.argv[14], sys.argv[15],
                               sys.argv[16])
    run(parameter)
    
if __name__ == "__main__":
    logging.basicConfig(
        filename="dynamic-coupling-simulation.log", level=logging.DEBUG, format="%(asctime)s %(message)s", datefmt="%I:%M:%S")
    #test()
    evaluation()
    