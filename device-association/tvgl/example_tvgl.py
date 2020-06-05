import os
import time
import random
import numpy as np
import coupling.utils.misc as misc
from coupling.tvgl.TVGL import TVGL
from utils.serializer import DillSerializer
import coupling.light_grouping_pattern.light_analysis as light_analysis

# https://stats.stackexchange.com/questions/73463/what-does-the-inverse-of-covariance-matrix-say-about-data-intuitively

# Input:
# Matrix: each column one device, each row one sampling
# lengthOfSlice = num columns

# Input test data:
# Each column a different stock
# Each row one stock value per day

# Output:
# covariance: how variables co-vary with each other (non-diagonal)
# inverse covariance: how variables do not co-vary with each other
# empirical covariance (empCov)
# theta: edges between covariance matrices over time
# theta diff: how does the covariance matrices changes from t to t+1 (detect change event)

# Parameter
# lamb: sparsity                        default: 2.5
# beta: temporal consistency            default: 12
# 
# indexOfPenalty: 1 = L1, 2 = L2, 3 = Laplacian, 4 = L_inf, 5 = perturbed node    default: perturbation
#     (regularization penalty)
# 
# eps: Threshold treat output network weight as zero        default: 3e-3
# epsAbs: ADMM absolute tolerance threshold                default: 1e-3
# epsRel: ADMM relative tolerance threshold                default: 1e-3

def create_test_data(num_devices, len_light_pattern=2, testlen=3000):
    light_pattern, _ = light_analysis.load_light_pattern(len_light_pattern)
    outside_devices = np.random.choice(range(num_devices))
    inside_devices = num_devices - outside_devices
    print("outside devices: ", outside_devices)
    print("inside_devices: ", inside_devices)
    std = 1
    mean = 0
    in_counter = 0
    out_counter = 0
    start_range = 200
    test_data = []
    for _ in range(num_devices):
        start = np.random.choice(range(start_range))
        in_out = bool(random.getrandbits(1))    
        if in_out == 1 and inside_devices == in_counter:
            in_out = 0
        elif in_out == 0 and outside_devices == out_counter:
            in_out = 1
        if in_out == 0: # out
            print("out")
            datalen = testlen - start
            test_device = np.random.normal(mean, std, size=datalen).reshape(-1,1)
            out_counter += 1
        else: # in
            print("in")
            test_device = light_pattern[start:testlen].reshape(-1,1)
            in_counter += 1
        test_data.append(test_device)
    datalen = min([len(data) for data in test_data])
    test_data = [data[:datalen] for data in test_data]
    return np.concatenate(test_data, axis=1)

def run_tvgl(test_data, path_tvgl):
    numColumns = test_data.shape[1]
    lengthOfSlice = numColumns # Number of samples in each ``slice'', or timestamp
    print("Test data: ", test_data.shape)
    print("length of slice: ", lengthOfSlice)
    start_time = time.time()
    thetaSet = TVGL(test_data, lengthOfSlice, lamb=2.5, beta=12, indexOfPenalty=-1, verbose=True)
    elapsed_time = time.time() - start_time
    print("elapsed time: ", elapsed_time)
    DillSerializer(path_tvgl).serialize(thetaSet)
    
def device_grouping(thetaSet):
    thetaSet = [entry for entry in thetaSet if misc.matrix_is_diag(entry)]
    totalsum = np.sum(thetaSet, axis=0)
    values = totalsum[totalsum != 0]
    divider = np.mean(values)
    print(values)
    print(divider)
    print(values < divider)
    
if __name__ == "__main__":
    path_tvgl = "./thetaSet"
    if not os.path.exists(path_tvgl):    
        num_devices = 2
        test_data = create_test_data(num_devices)
        run_tvgl(test_data, path_tvgl)
    theta_set = DillSerializer(path_tvgl).deserialize()
    device_grouping(theta_set)
    