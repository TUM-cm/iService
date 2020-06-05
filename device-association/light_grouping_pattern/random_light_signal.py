import io
import re
import time
import numpy
import pandas
import pexpect
import logging
import threading
import itertools
import matplotlib.pyplot as plt
import utils.statistics as statistics
from light_pattern import LightPattern

# Minimum amount of data that dieHarder works
# https://stackoverflow.com/questions/32954045/using-txt-file-containing-random-numbers-with-the-diehard-test-suite
# ~120MB of random binary data
# Generate test data: dieharder -o -f example.input -t 13000000
# Run dieharder test: dieharder -a -g 202 -f example.input

# http://manpages.ubuntu.com/manpages/xenial/man1/dieharder.1.html
# apt-get install dieharder

# R interface
# apt-get install r-base
# apt-get install r-recommended
# apt-get install libgsl-dev
# R as root
# apt install gawk
# git clone git://git.savannah.gnu.org/libtool.git
# install.packages("RDieHarder")

'''
Explanation of dieHarder

Test generated keys against statistical bias via dieHarder battery of statistical tests
Goal: uncover bias and dependency in the pseudo random sequence

p-value, probability that a real Random Number Generator (RNG) would produce this outcome, between 0 and 1
Good RNG features uniformly distributed p-values
p-value below a fixed significance level alpha = 0.001 indicates a failure of the PRNG with probability 1 - alpha
p-value <= 0.05 is expected 5% of the time
'''

def frange(start, stop, step, ndigits=1):
    x = start
    while x < stop:
        yield round(x, ndigits)
        x += step

class DieHarder:
    
    RESULT_LINE_DIEHARDER = "(\w+)\|(\d+)\|(\d+)\|(\d+)\|(\d+\.\d+)\|([A-Z]+)"
    
    class Parameter:
        def __init__(self, numbit, data_len, max_value):
            self.numbit = str(numbit)
            self.data_len = data_len
            self.max_value = max_value
    
    INT_32_Bit = Parameter(32, 10, 2147483647)
    INT_8_Bit = Parameter(8, 3, 255)
    
    def generate_data(self, parm, size_limit=60000000):
        numbers = list()
        while len(numbers) < size_limit:
            number = numpy.random.randint(0, parm.max_value)
            numbers.append(number)
        return numbers
    
    def write_file(self, data, parm, path="./", filename="test_data.txt", ):
        f = io.open(path + filename, 'a', newline="\n")
        f.seek(0)
        f.truncate()
        f.write(u"type: d\n")
        f.write("count: " + str(len(data)) + "\n")
        f.write("numbit: " + parm.numbit + "\n")
        for entry in data:
            f.write("{:{width}}\n".format(entry, width=parm.data_len))
        f.close()
    
    def test(self, file_test_data, rounds=21, serf=None, row=0):
        data = pandas.DataFrame(columns=["test_name", "ntup", "tsamples", "psamples", "p-value", "Assessment"])
        while _ in range(rounds):
            dieharder = pexpect.spawn("dieharder -a -g 202 -f " + file_test_data)
            while dieharder.isalive():
                dieharder.expect("\n", timeout=None)
                output = dieharder.before.strip()
                output = "".join(output.split())
                output = re.findall(DieHarder.RESULT_LINE_DIEHARDER, output)
                logging.debug(output)
                if output and len(output[0]) == 6:
                    data.loc[row] = output[0]
                    row += 1
        if serf:
            data.to_pickle(serf)
        else:
            return data
    
    def print_result(self, f):
        data = pandas.read_pickle(f)
        for col in ["ntup", "tsamples", "psamples", "p-value"]:
            data[col] = pandas.to_numeric(data[col])
            print(data.groupby("test_name").describe())
            #print(data.groupby("test_name")["p-value"].describe())
            data.boxplot(column="p-value", by="test_name", rot=45)
            plt.suptitle("") # remove automatic title
            plt.title("")
            plt.xlabel("statistical tests")
            plt.ylabel("p-value")
            plt.show()

class Entropy:
    
    def __init__(self):
        self.light_pattern = LightPattern()
    
    def generate_pattern_time(self, total_duration=1, duration_freq=0.1, pattern_len=2):
        self.run = True
        patterns = list()
        start = time.time()
        def set_new_pattern():
            self.pattern = self.light_pattern.get_pattern(pattern_len)            
            if time.time() - start < total_duration:
                t = threading.Timer(duration_freq, set_new_pattern)
                t.start()
            else:
                self.run = False
        set_new_pattern()
        while self.run:
            patterns.append(self.pattern)
        patterns = list(itertools.chain.from_iterable(patterns))
        return numpy.array(patterns)
     
    # delay influences frequency of random number generator (RNG)
    def generate_pattern_limit(self, duration_freq=0.1, pattern_len=2, len_limit=10000):
        self.run = True
        patterns = list()
        def set_new_pattern():
            self.pattern = self.light_pattern.get_pattern(pattern_len)
            if self.run:
                self.thread = threading.Timer(duration_freq, set_new_pattern)
                self.thread.start()
        set_new_pattern()
        while self.len < len_limit:
            patterns.append(self.pattern)
            self.len += len(self.pattern)
        self.run = False
        patterns = list(itertools.chain.from_iterable(patterns))
        return numpy.array(patterns)
    
    def test(self, result_file, fixed_samples=False):
        step_size = 0.1
        total_duration = 1        
        min_pattern_len = 2
        max_pattern_len = 10
        duration_freqs = list(frange(step_size, total_duration, step_size))
        pattern_lens = range(min_pattern_len, max_pattern_len+1, 2)
        data = pandas.DataFrame(index=pattern_lens, columns=duration_freqs)
        for pattern_len in pattern_lens:
            for duration_freq in duration_freqs:
                print("pattern len: {0}, duration freq: {1}".format(pattern_len, duration_freq))    
                if fixed_samples:
                    patterns = self.generate_pattern_limit(duration_freq, pattern_len)
                else:      
                    patterns = self.generate_pattern_time(pattern_len=pattern_len,
                                                          total_duration=total_duration,
                                                          duration_freq=duration_freq)
                entropy = statistics.entropy(patterns)
                print("pattern len: {}".format(len(patterns)))
                print("entropy: {}".format(entropy))
                data.loc[pattern_len, duration_freq] = entropy
        data.to_csv(result_file)
    
    def plot(self, result_file):
        data = pandas.read_csv(result_file, index_col=0)
        print("change frequency (s): {}".format(data.max(axis=0).idxmax()))
        print("pattern len: {}".format(data.max(axis=1).idxmax()))
        print("max entropy: {}".format(data.max(axis=0).max()))
        for index, row in data.iterrows():
            plt.plot(data.columns, row, label=index)
            for x, y in zip(data.columns, row):
                plt.annotate("{0:.2f}".format(y), xy=(x, y))
        plt.ylabel("Entropy")
        plt.xlabel("Change frequency of light pattern (s)")
        plt.legend(title="Pattern len", bbox_to_anchor=(0., 1.02, 1., .102),
                   ncol=5, loc=3, mode="expand", borderaxespad=0)
        plt.tight_layout()
        plt.savefig("entropy-light-signal.pdf")
        plt.show()

def test_entropy():
    patterns = list()
    light_pattern = LightPattern()
    data_len = 0
    pattern_len = 2
    len_limit = 13000000
    while data_len < len_limit:
        pattern = light_pattern.get_pattern(pattern_len)
        patterns.append(pattern)
        data_len += len(pattern)
    patterns = list(itertools.chain.from_iterable(patterns))
    patterns = numpy.array(patterns)
    #entropy = Entropy()
    #patterns = entropy.generate_pattern_time()
    
    dieharder = DieHarder()
    parm = DieHarder.INT_32_Bit
    dieharder.write_file(patterns, parm)
    
    #result_file = "entropy"
    #entropy.test(result_file)
    #entropy.plot(result_file)

def test_dieharder():
    dieharder = DieHarder()
    parm = DieHarder.INT_8_Bit
    data = dieharder.generate_data(parm, path="C:/Daten/Downloads/VM/")
    dieharder.write_file(data, parm)

def main():
    #test_dieharder()
    test_entropy()

if __name__ == "__main__":
    main()
