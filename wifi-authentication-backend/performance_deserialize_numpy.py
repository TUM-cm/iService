# import timeit
#  
# setup = '''
# import numpy
# import pickle
# import json
#  
# test_len = 10000
# voltage = numpy.random.randint(100,1000, size=(1,test_len), dtype=numpy.int)
# # numpy.iinfo(numpy.uint).min
# max_time = numpy.iinfo(numpy.uint).max
# time = numpy.random.randint(0, max_time, size=(1,test_len), dtype=numpy.uint)
#  
# publish_data = numpy.empty(shape=(2, voltage.shape[1]))
# publish_data[0] = voltage
# publish_data[1] = time
# '''
#  
# repeat = 1000
# execute = '''
# data = pickle.dumps(publish_data)
# convert = numpy.loads(data)
# #voltage = convert[0].astype(numpy.int)
# #time = convert[1].astype(numpy.uint)
# '''
# print "pickle"
# print timeit.timeit(execute, setup=setup, number=repeat)
#  
# execute = '''
# data = numpy.fromstring(publish_data.tostring())
# data_len = data.shape[0] / 2
# convert = data.reshape(2, data_len)
# #voltage = convert[0].astype(numpy.int)
# #time = convert[1].astype(numpy.uint)
# '''
# print "fromstring"
# print timeit.timeit(execute, setup=setup, number=repeat)
#  
# execute = '''
# data = json.dumps(publish_data.tolist())
# convert = numpy.array(json.loads(data))
# #voltage = convert[0].astype(numpy.int)
# #time = convert[1].astype(numpy.uint)
# '''
# print "json"
# print timeit.timeit(execute, setup=setup, number=repeat)

import numpy
import json
import pickle
  
test_len = 10000
voltage = numpy.random.randint(100, 1000, size=(1,test_len), dtype=numpy.int)
# numpy.iinfo(numpy.uint).min
max_time = numpy.iinfo(numpy.uint).max
time = numpy.random.randint(0, max_time, size=(1,test_len), dtype=numpy.uint)

publish_data = numpy.empty(shape=(2, voltage.shape[1]))
publish_data[0] = voltage
publish_data[1] = time
  
print voltage.dtype
print time.dtype
  
data = numpy.fromstring(publish_data.tostring())
data_len = data.shape[0] / 2
convert = data.reshape(2, data_len)
voltage = convert[0].astype(numpy.int)
time = convert[1].astype(numpy.uint)

print voltage.dtype
print time.dtype
  
print convert[0].astype(numpy.int)
print convert[1].astype(numpy.uint)
 
print "json"
data = json.dumps(publish_data.tolist())
convert = numpy.array(json.loads(data))
print convert[0].dtype
print convert[1].dtype
 
 
print "pickle"
data = pickle.dumps(publish_data)
convert = numpy.loads(data)
 
print convert[0].dtype
print convert[1].dtype
