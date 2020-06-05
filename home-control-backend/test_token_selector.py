import timeit

# include control
# get data from local connector via callback and select token

setup = '''
import numpy

msg = "abc abc abc abc abc abc ayooabc abc abc abc abc abc abtooabc abc abc abc abc abc abmoootbc abc abc abc abc abc abc oootbc abc abc abc abc abc abtootbc abc".split(" ")
for i, a in enumerate(msg):
    if a == "abc":
        msg[i] = "ttt"
'''

execute = '''
m = max(set(msg), key=msg.count)
t = (len(m) == 6 and m.isdigit())
'''

repeat = 100000
print "set time"
print timeit.timeit(execute, setup=setup, number=repeat)

execute = '''
unique, counts = numpy.unique(msg, return_counts=True)
max_idx = numpy.where(counts == counts.max())
m = unique[max_idx]
'''

#print "numpy time"
#print timeit.timeit(execute, setup=setup, number=repeat)