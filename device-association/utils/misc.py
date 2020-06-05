import re
import os
import time
import numpy
import random
import itertools
import threading
import matplotlib
import collections
from itertools import tee

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

# Set global matplotlib parameters
font = {'size' : 22}
matplotlib.rc('font', **font)
matplotlib.rcParams["pdf.fonttype"] = 42 # TrueType
matplotlib.rcParams["ps.fonttype"] = 42 # TrueType
matplotlib.rcParams["lines.linewidth"] = 3
matplotlib.rcParams["lines.markersize"] = 12
#matplotlib.rcParams['text.usetex'] = True # type1, different fonts

markers = ["o", "v", "^", "<", ">", "s", "p", "*", "h", "X", "D", "P" ]
hatches = ["////", 'ooo', "++", "xx", "\\", 'OO', '....', "**"]

class StopWatch:
    
    def __init__(self):
        self.start_time = -1
        self.stop_time = -1
        self.running = False
    
    def start(self):
        if not self.running:
            self.start_time = time.time()
            self.running = True
    
    def stop(self):
        if self.running:
            self.stop_time = time.time()
            self.running = False
    
    def get_elapsed_time(self):
        self.stop()
        return self.stop_time - self.start_time
    
    def reset(self):
        self.running = False

class AtomicCounter:
    
    def __init__(self, initial=0):
        self.value = initial
        self._lock = threading.RLock()
    
    def decrement(self, num=1):
        with self._lock:
            self.value -= num
            return self.value

    def increment(self, num=1):
        with self._lock:
            self.value += num
            return self.value
    
    def get(self):
        with self._lock:
            return self.value

def flatten_list(seq):
    return list(itertools.chain.from_iterable(seq))

def create_random_mac(separator=":"):
    mac = [random.randint(0, 255) for _ in range(0, 6)]
    mac[0] = (mac[0] & 0xfc) | 0x02
    return separator.join(["{0:02x}".format(x) for x in mac])

def get_valid_filename(s):
    s = str(s).strip().replace(" ", "_")
    return re.sub(r"(?u)[^-\w.]", "", s)

def matrix_is_diag(matrix):
    return numpy.all(matrix == numpy.diag(numpy.diagonal(matrix)))

def pairwise(iterable):
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)

def valid_light_pattern(light_pattern_duration, len_light_pattern=None, threshold_duration=1e3):
    light_pattern_len = numpy.array([len(pattern) for pattern in light_pattern_duration])
    if len_light_pattern:
        different_light_pattern = numpy.where(light_pattern_len != len_light_pattern)[0]
        wrong_light_pattern = len(light_pattern_len) == 0 or \
            True in (light_pattern_duration.ravel() < threshold_duration) or \
            len(different_light_pattern) > 0 or \
            len(light_pattern_duration) < len_light_pattern/2
    else:
        wrong_light_pattern = len(light_pattern_len) == 0 or \
            True in (light_pattern_duration.ravel() < threshold_duration) or \
            len(numpy.where(numpy.diff(light_pattern_len) != 0)[0]) > 0
    return not wrong_light_pattern

def __flatten(data, sep, parent_key=''):
    items = []
    for k, v in data.items():
        new_key = parent_key + sep + str(k) if parent_key else str(k)
        if isinstance(v, collections.MutableMapping):
            items.extend(__flatten(v, sep, new_key).items())
        else:
            items.append((new_key, v))
    return dict(items)

def __to_num(s):
    try:
        return int(s)
    except ValueError:
        try:
            return float(s)
        except ValueError:
            return s
    
def get_all_keys(data, level=None, sep=";"):
    unique_keys = list()
    keys = list(__flatten(data, sep).keys())
    levels = keys[0].count(sep) + 1 if level == None else level
    for level in range(levels):
        subset = set([key.split(sep)[level] for key in keys])
        subset = sorted(map(__to_num, subset))
        unique_keys.append(subset)
    return unique_keys

def chunks(data, chunk_size):
    for i in range(0, len(data), chunk_size):
        yield data[i:i+chunk_size]

def del_repeated_items(seq, truth):
    seq_count = {value: len(list(sublist)) for value, sublist in itertools.groupby(seq)}
    truth_count = {value: len(list(sublist)) for value, sublist in itertools.groupby(truth)}
    seq = dict()
    for value, count in seq_count.iteritems():
        if value in truth_count:
            count_compare = truth_count[value]
        else:
            count_compare = 1 # reduce to one element
        seq[value] = min(count, count_compare)
    result = list()
    for value, count in seq.iteritems():
        result.extend([value] * count)
    return result

def hamming_score(y_true, y_pred):
    '''
    Compute the Hamming score (a.k.a. label-based accuracy) for the multi-label case
    https://stackoverflow.com/q/32239577/395857
    '''
    acc_list = []
    for i in range(y_true.shape[0]):
        set_true = set(numpy.where(y_true[i])[0])
        set_pred = set(numpy.where(y_pred[i])[0])
        tmp_a = None
        if len(set_true) == 0 and len(set_pred) == 0:
            tmp_a = 1
        else:
            tmp_a = len(set_true.intersection(set_pred))/\
                    float( len(set_true.union(set_pred)) )
        acc_list.append(tmp_a)
    return numpy.mean(acc_list)

def main():
    eval_result = [0,0,0,1,1,2,3,4,4]
    truth = [0,0,1,2,2,4]
    print("eval result: ", eval_result)
    print("truth: ", truth)
    print("result: ", del_repeated_items(eval_result, truth))
    
if __name__ == "__main__":
    main()
    