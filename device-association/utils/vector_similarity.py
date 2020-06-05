from __future__ import division
import os
import numpy
import random
import scipy.stats
import scipy.signal
import scipy.spatial
import utils.statistics
from utils.dtw import Dtw
from difflib import SequenceMatcher
from utils.custom_enum import enum_name
from utils.distance_correlation import distcorr

dtw = Dtw()
equalize_methods = enum_name("fill", "cut", "dtw")
__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

def equalize(x, y, method):
    x_len = x.shape[0]
    y_len = y.shape[0]
    if x_len == y_len:
        return x, y
    if method == equalize_methods.cut:        
        min_val = min(x_len, y_len)
        if x_len > min_val:
            return x[:min_val], y
        elif y_len > min_val:
            return x, y[:min_val]
    elif method == equalize_methods.fill:
        max_val = max(x_len, y_len)
        if x_len < max_val:
            x = x.copy()
            x.resize(max_val, refcheck=False)
        elif y_len < max_val:
            y = y.copy()
            y.resize(max_val, refcheck=False)
        return x, y
    elif method == equalize_methods.dtw:
        if x_len > y_len:    
            alignment = dtw.calculate_alignment(y, x)
            y = dtw.apply_warp_query(alignment, y)
        else:
            alignment = dtw.calculate_alignment(x, y)
            x = dtw.apply_warp_query(alignment, x)
        return x, y

def jaccard(x , y, equalize_method):
    x, y = equalize(x, y, equalize_method)
    return 1 - scipy.spatial.distance.jaccard(x, y)

def pearson(x, y, equalize_method):
    x, y = equalize(x, y, equalize_method)
    # map [-1, 1] to [0, 1]
    return abs(scipy.stats.pearsonr(x, y)[0])

def spearman(x, y, equalize_method):
    x, y = equalize(x, y, equalize_method)
    # map [-1, 1] to [0, 1]
    return abs(scipy.stats.spearmanr(x, y).correlation)

def kendall(x, y, equalize_method):
    x, y = equalize(x, y, equalize_method)
    result = scipy.stats.kendalltau(x, y).correlation
    return numpy.interp(result, [-1,1],[0,1])

def distance_correlation(x, y, equalize_method):
    x, y = equalize(x, y, equalize_method)
    return distcorr(x, y)

def euclidean(x, y, equalize_method):
    x, y = equalize(x, y, equalize_method)
    x = utils.statistics.linalg_norm(x)
    y = utils.statistics.linalg_norm(y)
    return 1 - scipy.spatial.distance.euclidean(x, y)

def manhattan(x, y, equalize_method):
    x, y = equalize(x, y, equalize_method)
    x = utils.statistics.sum_norm(x)
    y = utils.statistics.sum_norm(y)
    return 1 - scipy.spatial.distance.cityblock(x, y)

def minkowski(x, y, equalize_method, norm=2):
    x, y = equalize(x, y, equalize_method)
    x = utils.statistics.linalg_norm(x)
    y = utils.statistics.linalg_norm(y)
    return 1 - scipy.spatial.distance.minkowski(x, y, norm)

def cosine(x, y, equalize_method):
    x, y = equalize(x, y, equalize_method)
    return 1 - scipy.spatial.distance.cosine(x, y)

def coherence(x, y, _):
    c_xy = scipy.signal.coherence(x, y)[1]
    return numpy.mean(c_xy)

def hamming(x, y, equalize_method):
    x, y = equalize(x, y, equalize_method)
    return 1 - scipy.spatial.distance.hamming(x, y)

def dtw_distance(x, y, _):
    x = utils.statistics.sum_norm(x)
    y = utils.statistics.sum_norm(y)
    alignment = dtw.calculate_alignment(x, y)
    return 1 - dtw.get_distance(alignment)

def sequence_matcher(x, y, _):
    return SequenceMatcher(None, x, y). ratio()

similarity_methods = [
    jaccard,
    pearson,
    spearman,
    kendall,
    distance_correlation,
    euclidean,
    manhattan,
    minkowski,
    cosine,
    coherence,
    hamming,
    dtw_distance,
    #sequence_matcher
]

def test_metrics_all_one():
    client1 = numpy.random.randint(100, 500, random.randint(3000, 10000))
    client2 = client1
    for distance_method in similarity_methods:
        similarity = distance_method(client1, client2, "cut")
        print(distance_method.__name__)
        print(similarity)

def test_metrics_within_range(detailed=False):
    for distance_method in similarity_methods:
        print(distance_method.__name__)
        results = list()
        for i in range(50):
            if detailed:
                print("round: ", i)
            client1 = numpy.random.randint(100, 500, random.randint(3000, 10000))
            client2 = numpy.random.randint(100, 500, random.randint(3000, 10000))
            similarity = distance_method(client1, client2, "cut")
            results.append(similarity)
            if similarity < 0 or similarity > 1:
                print("outside of range")
                print(distance_method.__name__)
                print(results)
                import sys
                sys.exit(0)
        print(results)
    
def main():
    test_metrics_all_one()
    test_metrics_within_range()
    
if __name__ == "__main__":
    main()
